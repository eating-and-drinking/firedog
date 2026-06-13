#!/usr/bin/env python3
"""
scripts/voice_chat.py
端到端语音对话：麦克风 → 唤醒词 → ASR → LLM(流式) → TTS(流水线) → 音箱

低延迟设计
  LLM 用 TextIteratorStreamer 流式生成，按句切分后立刻送 TTS；
  TTS 合成与播放双线程流水线：播第 N 句的同时合成第 N+1 句。
  首音延迟 ≈ 首句生成时间 + 首句合成时间，而非全文生成+合成。

唤醒词逻辑
  OWW 模式（openwakeword 模型可用时）：逐 80ms chunk 实时判断
  ASR 兜底模式：VAD 端点检测 + ASR 关键词匹配

打断（barge-in）
  唤醒时注册说话人声纹；TTS 播放期间只有声纹相符的人才能打断。
  打断成功后，触发打断的那段语音直接作为下一句话的开头续接识别，
  无需重说。

使用方式：
  python scripts/voice_chat.py
  python scripts/voice_chat.py --no-wake          # 跳过唤醒词，直接对话
  python scripts/voice_chat.py --no-speaker-id    # 关闭声纹验证
  python scripts/voice_chat.py --config config/config.yaml
"""
from __future__ import annotations

import collections
import os
import queue
import re
import signal
import sys
import threading
import time
from pathlib import Path

import click
import numpy as np
import sounddevice as sd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    def _replace(m):
        var = m.group(1) or m.group(2)
        return os.environ.get(var, m.group(0))
    raw = re.sub(r"\$\{(\w+)\}|\$(\w+)", _replace, raw)
    return yaml.safe_load(raw)


def auto_detect_device(keyword: str) -> int | None:
    keyword_lower = keyword.lower()
    for i in range(len(sd.query_devices())):
        info = sd.query_devices(i)
        if keyword_lower in info["name"].lower():
            return i
    return None


def resolve_device(spec: str | None, direction: str) -> int | None:
    if spec is None or spec == "":
        return None
    if isinstance(spec, int):
        return spec
    spec_str = str(spec)
    if spec_str.startswith("auto:"):
        keyword = spec_str[5:]
        idx = auto_detect_device(keyword)
        if idx is not None:
            info = sd.query_devices(idx)
            click.echo(f"  检测到{direction}设备 [{idx}]: {info['name']}")
            return idx
        click.echo(f"  未找到 '{keyword}' 的{direction}设备，使用系统默认")
        return None
    try:
        return int(spec_str)
    except ValueError:
        click.echo(f"  无法解析设备配置 '{spec_str}'，使用系统默认")
        return None


def list_audio_devices():
    click.echo("\n系统音频设备列表：")
    click.echo("-" * 70)
    for i in range(len(sd.query_devices())):
        info = sd.query_devices(i)
        in_ch = info["max_input_channels"]
        out_ch = info["max_output_channels"]
        marker = ""
        if in_ch > 0 and "listengo" in info["name"].lower():
            marker = " <- 麦克风"
        elif out_ch > 0 and "usb audio" in info["name"].lower():
            marker = " <- 音箱"
        if in_ch > 0 or out_ch > 0:
            click.echo(
                f"  [{i:2d}] {info['name']}"
                f"  (输入:{in_ch}ch 输出:{out_ch}ch){marker}"
            )
    click.echo("-" * 70)


def drain_queue(q: "queue.Queue[np.ndarray]") -> list[np.ndarray]:
    """非阻塞取空队列，返回取出的全部帧。"""
    out: list[np.ndarray] = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def clear_queue(q: "queue.Queue[np.ndarray]") -> None:
    drain_queue(q)


def create_llm_handler(cfg: dict):
    """
    本地 Qwen2.5-Instruct LLM handler（流式）。

    返回 stream(user_text, abort_event) -> Iterator[str]：
    边生成边按句切分 yield，可通过 abort_event 中途停止生成（打断时用）。
    对话历史在生成器结束（含被提前关闭）时落账。
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )

    llm_cfg = cfg.get("llm", {})
    model_path = llm_cfg.get("model", "./Qwen2.5-3B-Instruct")
    temperature = float(llm_cfg.get("temperature", 0.7))
    max_tokens = int(llm_cfg.get("max_tokens", 512))
    timeout_s = float(llm_cfg.get("timeout_s", 60))

    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"  LLM: {model_path} ({device}) 加载中...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    llm_model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "你是一只机器狗的智能助手。请用简短、自然的口语回答，"
                "每次回答不超过两句话。不要使用 Markdown 格式。"
            ),
        }
    ]

    SENTENCE_END = "。！？!?；;\n"
    MIN_SENTENCE_CHARS = 6  # 太短的句子并入下一句，避免 TTS 频繁起停

    class _StopOnEvent(StoppingCriteria):
        def __init__(self, event: threading.Event):
            self._event = event

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return self._event.is_set()

    def _split_first_sentence(buf: str) -> tuple[str | None, str]:
        for i, ch in enumerate(buf):
            if ch in SENTENCE_END and i + 1 >= MIN_SENTENCE_CHARS:
                return buf[: i + 1], buf[i + 1:]
        return None, buf

    def stream(user_text: str, abort_event: threading.Event | None = None):
        abort_event = abort_event or threading.Event()
        messages.append({"role": "user", "content": user_text})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=timeout_s
        )

        gen_error: list[Exception] = []

        def _generate():
            try:
                with torch.no_grad():
                    llm_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList(
                            [_StopOnEvent(abort_event)]
                        ),
                    )
            except Exception as exc:
                gen_error.append(exc)

        threading.Thread(target=_generate, daemon=True, name="llm_generate").start()

        produced: list[str] = []
        buf = ""
        try:
            for piece in streamer:
                if abort_event.is_set():
                    break
                buf += piece
                while True:
                    sent, buf = _split_first_sentence(buf)
                    if sent is None:
                        break
                    if sent.strip():
                        produced.append(sent)
                        yield sent
            if buf.strip() and not abort_event.is_set():
                produced.append(buf)
                yield buf
            if gen_error:
                click.echo(f"\n  LLM 错误: {gen_error[0]}")
                if not produced:
                    fallback = "抱歉，我思考出了问题，请再说一遍。"
                    produced.append(fallback)
                    yield fallback
        except Exception as exc:  # streamer 超时等
            click.echo(f"\n  LLM 流式错误: {exc}")
        finally:
            # 消费方提前退出（打断）时也确保生成线程尽快停下
            abort_event.set()
            reply = "".join(produced).strip()
            if reply:
                messages.append({"role": "assistant", "content": reply})
            else:
                messages.pop()  # 本轮失败，回滚 user 消息
            if len(messages) > 42:
                messages[:] = [messages[0]] + messages[-40:]

    return stream


# ─────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--config", default="config/config.yaml", help="配置文件路径")
@click.option("--no-wake", is_flag=True, default=False, help="跳过唤醒词，直接对话模式")
@click.option("--no-speaker-id", is_flag=True, default=False, help="关闭声纹验证（任何人可打断）")
@click.option("--list-devices", is_flag=True, default=False, help="列出音频设备后退出")
def main(config: str, no_wake: bool, no_speaker_id: bool, list_devices: bool):
    """机器狗语音对话 — 麦克风 -> 唤醒词 -> ASR -> LLM -> TTS -> 音箱"""

    if list_devices:
        list_audio_devices()
        return

    cfg = load_config(config)
    v = cfg.get("voice", {})

    from src.utils.logger import setup_logging, get_logger
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("file"))
    log = get_logger(__name__)

    click.echo("\n" + "=" * 56)
    click.echo("  机器狗语音对话系统")
    click.echo("=" * 56)

    # ── 音频设备 ────────────────────────────────────────────────────
    click.echo("\n检测音频设备...")
    list_audio_devices()
    input_device  = resolve_device(v.get("audio", {}).get("input_device"), "麦克风输入")
    output_device = resolve_device(v.get("audio", {}).get("output_device"), "音箱输出")

    # ── 加载模块 ─────────────────────────────────────────────────────
    click.echo("\n加载模块...")

    from src.voice.asr import ASREngine, ASRConfig
    asr = ASREngine(ASRConfig(
        model_id=v.get("asr", {}).get("model_id", "./SenseVoiceSmall"),
        language=v.get("asr", {}).get("language", "zh"),
        device=v.get("asr", {}).get("device", "cuda"),
        use_itn=v.get("asr", {}).get("use_itn", True),
    ))
    click.echo("  ASR: SenseVoice-Small 就绪")

    from src.voice.tts import TTSEngine, TTSConfig
    tts = TTSEngine(TTSConfig(
        backend=v.get("tts", {}).get("backend", "cosyvoice"),
        model_dir=v.get("tts", {}).get("model_dir", "./Fun-CosyVoice3-0.5B-2512"),
        prompt_speech=v.get("tts", {}).get("prompt_speech", ""),
        prompt_text=v.get("tts", {}).get("prompt_text", ""),
        instruct=v.get("tts", {}).get("instruct", ""),
        speed=v.get("tts", {}).get("speed", 1.0),
        sample_rate=v.get("tts", {}).get("sample_rate", 24000),
        output_device=output_device,
        kokoro_voice=v.get("tts", {}).get("kokoro_voice", "zh_female_1"),
    ))
    click.echo(f"  TTS: {v.get('tts', {}).get('backend', 'cosyvoice')} 就绪")

    from src.voice.vad import SileroVAD, VADConfig
    vad = SileroVAD(VADConfig(
        threshold=v.get("vad", {}).get("threshold", 0.5),
        speech_pad_ms=v.get("vad", {}).get("speech_pad_ms", 400),
        chunk_ms=v.get("vad", {}).get("chunk_ms", 80),
    ))
    click.echo("  VAD: Silero 就绪")

    # 降噪前端（RNNoise）：在增益之后、入队之前逐 chunk 处理
    from src.voice.denoise import Denoiser, DenoiseConfig
    dn_raw = v.get("denoise", {}) or {}
    denoiser = Denoiser(DenoiseConfig(
        enabled=bool(dn_raw.get("enabled", False)),
        backend=dn_raw.get("backend", "rnnoise"),
        sample_rate=v.get("audio", {}).get("sample_rate", 16000),
    ))
    click.echo(f"  降噪: {denoiser.backend}")

    # 唤醒词检测器
    from src.voice.wake_word import WakeWordDetector, WakeWordConfig
    ww_raw = v.get("wake_word", {})
    ww_detector = WakeWordDetector(WakeWordConfig(
        model=ww_raw.get("model", "hey_jarvis"),
        threshold=float(ww_raw.get("threshold", 0.5)),
        keywords=ww_raw.get("keywords", ["你好小狗", "嘿小狗", "机器狗"]),
        fuzzy_pinyin=bool(ww_raw.get("fuzzy_pinyin", True)),
        sherpa_model_dir=ww_raw.get(
            "sherpa_model_dir",
            "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
        ),
        sherpa_threshold=float(ww_raw.get("sherpa_threshold", 0.25)),
    ))
    # 混合唤醒：KWS 与 ASR 兜底并行（KWS 在真实声学链路上可能漏检）
    wake_hybrid = bool(ww_raw.get("hybrid", True))
    click.echo(
        f"  唤醒词: 模式={ww_detector.mode}"
        f"  关键词={ww_raw.get('keywords', ['你好小狗'])}"
    )

    # 声纹验证器
    from src.voice.speaker_id import SpeakerVerifier, SpeakerIDConfig
    spk_raw = v.get("speaker_id", {})
    spk_enabled = spk_raw.get("enabled", True) and not no_speaker_id
    speaker = SpeakerVerifier(SpeakerIDConfig(
        enabled=spk_enabled,
        similarity_threshold=float(spk_raw.get("similarity_threshold", 0.75)),
        enroll_min_s=float(spk_raw.get("enroll_min_s", 1.0)),
        sample_rate=v.get("audio", {}).get("sample_rate", 16000),
    ))
    click.echo(
        f"  声纹验证: {'已启用' if spk_enabled else '已关闭'}"
        + (f" (阈值={speaker.threshold:.2f}, backend={speaker._backend})" if spk_enabled else "")
    )

    click.echo("  LLM 加载中...")
    llm_stream = create_llm_handler(cfg)
    click.echo("  LLM 就绪")

    # ── 预热 ─────────────────────────────────────────────────────────
    # CosyVoice 首次推理含 CUDA 初始化/JIT，实测比热态慢 ~10 倍；
    # 启动时各跑一次空请求，避免用户第一次对话白等
    click.echo("  预热模型...")
    _t0 = time.monotonic()
    asr.transcribe(np.zeros(int(v.get("audio", {}).get("sample_rate", 16000) * 0.5),
                            dtype=np.float32))
    tts.synthesize("你好")
    click.echo(f"  预热完成 ({time.monotonic() - _t0:.1f}s)")

    # ── 运行参数 ─────────────────────────────────────────────────────
    sample_rate          = v.get("audio", {}).get("sample_rate", 16000)
    chunk_ms             = v.get("vad", {}).get("chunk_ms", 80)
    chunk_samples        = int(sample_rate * chunk_ms / 1000)
    silence_timeout_s    = v.get("timeouts", {}).get("silence_timeout_s", 5.0)
    post_tts_cooldown_s  = v.get("timeouts", {}).get("post_tts_cooldown_ms", 800) / 1000.0
    vad_threshold        = v.get("vad", {}).get("threshold", 0.3)
    mic_gain             = float(v.get("audio", {}).get("mic_gain", 1.0))

    # 打断参数（新结构 voice.barge_in.*，兼容旧的顶层 barge_in_threshold）
    bi_raw = v.get("barge_in", {}) or {}
    barge_in_threshold = float(bi_raw.get("threshold", v.get("barge_in_threshold", 0.7)))
    verify_interval_s  = float(bi_raw.get("verify_interval_s", 0.5))
    keep_barge_audio   = bool(bi_raw.get("keep_audio", True))
    # 起音后先积累这么久的"纯用户语音"再做声纹验证；起音前的帧多为 TTS 回声，
    # 混进验证窗口会显著拉低相似度（声纹不准的主因）
    verify_audio_s     = float(bi_raw.get("verify_audio_s", 0.8))
    speaker_threshold  = speaker.threshold

    # 端点检测参数
    ep_raw = v.get("endpoint", {}) or {}
    IDLE_SILENCE_END   = int(ep_raw.get("idle_silence_chunks", 10))    # ~800ms @80ms
    LISTEN_SILENCE_END = int(ep_raw.get("listen_silence_chunks", 8))   # ~640ms
    max_utterance_s    = float(ep_raw.get("max_utterance_s", 16.0))
    max_utterance_frames = max(1, int(max_utterance_s * 1000 / chunk_ms))
    # 累计有声帧达到 min_speech_duration_ms 才确认"在说话"——
    # 孤立尖峰（键盘/咔哒/呼吸，AGC 会把它们抬到可闻电平）不再触发识别
    MIN_VOICED_CHUNKS = max(1, round(
        v.get("vad", {}).get("min_speech_duration_ms", 250) / chunk_ms
    ))
    # 单字语气词不构成有效指令（呼吸/残余回声常被 ASR 转成"嗯"）
    FILLER_CHARS = set("嗯啊呃哦哈唉诶呀嘛哼噢喔")

    # 连续对话：回答完后保持聆听（5s 内可直接追问），超时回待机
    continuous = bool(v.get("conversation", {}).get("continuous", True))

    # 唤醒前滚动缓冲（约 4s，用于声纹注册，覆盖唤醒词 + 前置语音）
    RING_MAXLEN = max(1, int(4.0 * sample_rate / chunk_samples))
    ww_ring: collections.deque[np.ndarray] = collections.deque(maxlen=RING_MAXLEN)

    # 声纹注册时批量过滤静音帧用的独立 VAD 实例
    # （主 vad 持有麦克风流的实时状态，不能拿来跑历史缓冲）
    batch_vad = SileroVAD(VADConfig(chunk_ms=chunk_ms))

    # ── 音频采集 ─────────────────────────────────────────────────────
    # 有界队列：约 40s 容量，消费停滞时丢新帧而不是无界堆积
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=512)
    running = True
    # 看门狗：回调线程每收到一帧就刷新时间戳；超时视为麦克风掉线
    watchdog_timeout_s = float(v.get("audio", {}).get("watchdog_timeout_s", 3.0))
    last_chunk_ts = [time.monotonic()]

    def audio_callback(indata, frames_, time_info, status):
        if status:
            log.warning("audio_stream_status", status=str(status))
        last_chunk_ts[0] = time.monotonic()
        chunk = indata[:, 0].copy()
        if mic_gain != 1.0:
            chunk = np.clip(chunk * mic_gain, -1.0, 1.0)
        chunk = denoiser.process(chunk)
        try:
            audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def open_input_stream(device: int | None) -> sd.InputStream:
        s = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            device=device,
            callback=audio_callback,
        )
        s.start()
        return s

    stream = open_input_stream(input_device)

    def _shutdown(sig, frame):
        nonlocal running
        running = False
        stream.stop()
        stream.close()
        if tts.is_speaking():
            tts.interrupt()
        click.echo("\n再见！")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 启动提示 ─────────────────────────────────────────────────────
    click.echo("\n" + "=" * 56)
    if no_wake:
        click.echo("  直接对话模式 — 说话即可")
    else:
        ww_words = " / ".join(ww_raw.get("keywords", ["你好小狗"]))
        if ww_detector.mode == "sherpa":
            mode_desc = "sherpa-KWS + ASR 混合" if wake_hybrid else "sherpa-KWS 流式"
            click.echo(f"  唤醒词模式 ({mode_desc}) — 说 [{ww_words}] 唤醒")
        elif ww_detector.mode == "oww":
            click.echo("  唤醒词模式 (OWW) — 说唤醒词唤醒")
        else:
            click.echo(f"  唤醒词模式 (ASR 兜底) — 说 [{ww_words}] 唤醒")
    if spk_enabled:
        click.echo(f"  声纹锁 已启用 — 只有唤醒人可打断 (阈值 {speaker_threshold:.2f})")
    else:
        click.echo("  声纹锁 已关闭")
    if continuous and not no_wake:
        click.echo("  连续对话 已启用 — 回答完后可直接追问，超时回待机")
    click.echo(f"  增益={mic_gain}x  VAD={vad_threshold}  打断={barge_in_threshold}")
    click.echo("  Ctrl+C 退出")
    click.echo("=" * 56 + "\n")

    # ═════════════════════════════════════════════════════════════════
    # 主循环
    # ═════════════════════════════════════════════════════════════════
    state = "LISTENING" if no_wake else "IDLE"
    user_text = ""
    # 打断成功后遗留的语音帧，作为下一轮 LISTENING 的开头（续接，免重说）
    preroll_frames: list[np.ndarray] = []

    # 待机 ASR 兜底：端点检测状态
    idle_buf: list[np.ndarray] = []
    idle_in_speech = False
    idle_silence_count = 0
    idle_voiced_count = 0
    # 起始前缀缓冲：捕获语音开头 320ms，避免漏掉第一个音节
    idle_pre: collections.deque[np.ndarray] = collections.deque(maxlen=4)

    while running:
        try:
            # ── 看门狗：麦克风停止供帧（USB 掉线/驱动挂死）→ 自动重连 ──
            if time.monotonic() - last_chunk_ts[0] > watchdog_timeout_s:
                click.echo("\n  [看门狗] 麦克风超时，尝试重连...")
                log.warning("mic_watchdog_restart")
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                try:
                    # 重新解析设备：拔插后索引可能变化
                    new_dev = resolve_device(
                        v.get("audio", {}).get("input_device"), "麦克风输入"
                    )
                    stream = open_input_stream(new_dev)
                    denoiser.reset()
                    vad.reset()
                    clear_queue(audio_queue)
                    click.echo("  [看门狗] 麦克风重连成功")
                    log.info("mic_watchdog_recovered")
                except Exception as exc:
                    log.error("mic_watchdog_restart_failed", error=str(exc))
                    time.sleep(2.0)
                last_chunk_ts[0] = time.monotonic()  # 防止紧密重试

            # ──────────────────────────────────────────────────────
            # IDLE：等待唤醒词
            # ──────────────────────────────────────────────────────
            if state == "IDLE":
                time.sleep(0.02)

                woken = False
                for chunk in drain_queue(audio_queue):
                    ww_ring.append(chunk)
                    if woken:
                        continue  # 已唤醒，剩余帧只补进声纹注册缓冲

                    # ─ KWS 主路（sherpa-onnx / OWW，逐 chunk 流式）────
                    if ww_detector.mode in ("sherpa", "oww"):
                        prob = ww_detector.process_chunk(chunk)
                        if prob >= ww_detector.threshold:
                            click.echo(f"\n  [唤醒] {ww_detector.mode} 分数 {prob:.2f}")
                            woken = True

                    # ─ ASR 兜底：VAD 端点检测 ──────────────────────
                    # 混合模式：与 KWS 并行跑，谁先命中谁唤醒。
                    # 既保底召回，又保留 [待机] 转写日志（KWS 漏检时可诊断）
                    if not woken and (
                        ww_detector.mode == "asr" or wake_hybrid
                    ):
                        p = vad.is_speech(chunk)
                        if p >= vad_threshold:
                            if not idle_in_speech:
                                # 语音刚开始：把前 320ms 预缓冲也加进去
                                idle_buf.extend(idle_pre)
                                idle_pre.clear()
                            idle_in_speech = True
                            idle_voiced_count += 1
                            idle_silence_count = 0
                            idle_buf.append(chunk)
                        else:
                            idle_pre.append(chunk)  # 滚动保存静音帧备用
                            if idle_in_speech:
                                idle_silence_count += 1
                                idle_buf.append(chunk)
                                if idle_silence_count >= IDLE_SILENCE_END:
                                    # 仅用语音段（idle_buf）送 ASR，避免大量静音干扰
                                    seg = np.concatenate(idle_buf)
                                    voiced = idle_voiced_count
                                    idle_buf.clear()
                                    idle_pre.clear()
                                    idle_in_speech = False
                                    idle_silence_count = 0
                                    idle_voiced_count = 0

                                    # 有声帧不足 ~240ms：孤立尖峰，不值得跑 ASR
                                    if voiced < MIN_VOICED_CHUNKS:
                                        continue

                                    text = asr.transcribe(seg)
                                    click.echo(
                                        f"\r  [待机] {text}" + " " * 10, nl=False
                                    )

                                    if ww_detector.check_text(text) > 0:
                                        click.echo(f"\n  [唤醒] 关键词: {text}")
                                        woken = True

                if woken:
                    # 声纹注册
                    clear_queue(audio_queue)
                    vad.reset()
                    idle_buf.clear()
                    idle_pre.clear()
                    idle_in_speech = False
                    idle_silence_count = 0
                    idle_voiced_count = 0

                    if speaker.enabled:
                        # 注册前用 VAD 滤掉静音/底噪帧：4s 滚动缓冲里大半是
                        # 静音，直接整段注册会让声纹 embedding 偏脏
                        ring_frames = list(ww_ring)
                        batch_vad.reset()
                        speech_frames = [
                            c for c in ring_frames
                            if batch_vad.is_speech(c) >= 0.35
                        ]
                        min_enroll = int(
                            float(spk_raw.get("enroll_min_s", 1.0)) * sample_rate
                        )
                        filtered = (
                            np.concatenate(speech_frames)
                            if speech_frames
                            else np.array([], dtype=np.float32)
                        )
                        # 过滤后不足注册时长则回退整段缓冲
                        enroll_audio = (
                            filtered
                            if len(filtered) >= min_enroll
                            else (
                                np.concatenate(ring_frames)
                                if ring_frames
                                else np.array([], dtype=np.float32)
                            )
                        )
                        if speaker.enroll(enroll_audio):
                            click.echo(
                                f"  声纹注册成功 (阈值 {speaker_threshold:.2f})"
                            )
                        else:
                            click.echo("  声纹注册失败（音频不足），任何人均可打断")
                    else:
                        click.echo("  声纹验证已关闭")

                    click.echo("  请说话...")
                    state = "LISTENING"

            # ──────────────────────────────────────────────────────
            # LISTENING：收集用户语音
            # ──────────────────────────────────────────────────────
            elif state == "LISTENING":
                # 打断续接：preroll 是触发打断的语音，作为本句开头
                frames: list[np.ndarray] = preroll_frames
                preroll_frames = []
                speech_detected = bool(frames)
                # preroll 是已通过声纹验证的打断语音，视为已确认在说话
                voiced_chunks = MIN_VOICED_CHUNKS if frames else 0
                checked_up_to = len(frames)  # preroll 在 SPEAKING 期间已做过 VAD
                listen_start = time.monotonic()
                silence_chunks = 0

                while running:
                    frames.extend(drain_queue(audio_queue))

                    new_frames = frames[checked_up_to:]
                    if not new_frames:
                        if (
                            not speech_detected
                            and time.monotonic() - listen_start > silence_timeout_s
                        ):
                            state = "IDLE" if not no_wake else "LISTENING"
                            clear_queue(audio_queue)
                            vad.reset()
                            break
                        time.sleep(0.02)
                        continue

                    for chunk in new_frames:
                        prob = vad.is_speech(chunk)
                        if prob > 0.1:
                            bar = "█" * int(prob * 20)
                            click.echo(f"\r  VAD {prob:.2f} {bar:<20}", nl=False)
                        if prob >= vad_threshold:
                            voiced_chunks += 1
                            silence_chunks = 0
                            # 累计有声 ~240ms 才确认在说话，孤立尖峰不算
                            if voiced_chunks >= MIN_VOICED_CHUNKS:
                                speech_detected = True
                        elif speech_detected:
                            silence_chunks += 1
                        else:
                            voiced_chunks = 0  # 尖峰被静音打断，清零重计

                    checked_up_to = len(frames)

                    if speech_detected and silence_chunks >= LISTEN_SILENCE_END:
                        break

                    if len(frames) >= max_utterance_frames:
                        break  # 超长强制断句

                    if time.monotonic() - listen_start > silence_timeout_s:
                        if speech_detected:
                            break
                        state = "IDLE" if not no_wake else "LISTENING"
                        clear_queue(audio_queue)
                        vad.reset()
                        break

                    time.sleep(0.02)

                if not speech_detected:
                    continue

                audio_data = np.concatenate(frames[-max_utterance_frames:])

                click.echo("  识别中...", nl=False)
                user_text = asr.transcribe(audio_data)
                # 无效输入过滤：纯标点/空（回声、呼吸常转成"。"），
                # 以及单字语气词（"嗯。"——不构成指令，接话只会自言自语）
                content = re.sub(r"[^0-9A-Za-z一-鿿]", "", user_text)
                if not content or (len(content) == 1 and content in FILLER_CHARS):
                    click.echo("\r  未识别到有效语音")
                    state = "LISTENING" if (no_wake or continuous) else "IDLE"
                    clear_queue(audio_queue)
                    vad.reset()
                    continue

                click.echo(f"\r  你说: {user_text}")
                log.info("user_said", text=user_text)

                # 用这段干净语音（无 TTS 播放时录得）微调声纹，越聊越准
                if speaker.enabled and speaker.has_speaker:
                    refined = speaker.update(audio_data)
                    if refined is not None:
                        log.debug("speaker_update_ok", similarity=round(refined, 3))

                state = "PROCESSING"

            # ──────────────────────────────────────────────────────
            # PROCESSING：LLM 流式生成 + SPEAKING：流水线播放
            # ──────────────────────────────────────────────────────
            elif state == "PROCESSING":
                click.echo("  思考中...", nl=False)
                t0 = time.monotonic()
                llm_abort = threading.Event()
                first_sentence_ts: list[float] = []

                def _echoed_sentences(
                    gen=llm_stream(user_text, llm_abort),
                    t0=t0,
                    first=first_sentence_ts,
                ):
                    # 在 TTS 合成线程中被消费；每句产出时即时回显
                    for sent in gen:
                        s = sent.strip()
                        if not s:
                            continue
                        if not first:
                            first.append(time.monotonic())
                            latency_ms = (first[0] - t0) * 1000
                            click.echo(f"\r  机器狗: {s}  (首句 {latency_ms:.0f}ms)")
                            log.info("llm_first_sentence", ms=round(latency_ms, 1))
                        else:
                            click.echo(f"          {s}")
                        yield s

                # ── SPEAKING：边生成边合成边播放，同时监听打断 ────
                state = "SPEAKING"
                clear_queue(audio_queue)
                vad.reset()
                tts.speak_stream(_echoed_sentences())

                # 打断检测：VAD 过阈值的那一帧是"起音点"。起音点之前的帧
                # 基本是 TTS 回声，绝不能混进声纹验证窗口（会拉低相似度）；
                # 起音点之后先积累 verify_audio_s 的纯用户语音再验证。
                barge_frames: list[np.ndarray] = []
                barge_onset: int | None = None
                last_reject_ts = 0.0
                interrupted = False
                ONSET_PAD = 2  # 起音点回溯 ~160ms，补上起音瞬间
                verify_audio_samples = int(verify_audio_s * sample_rate)

                while tts.is_speaking() and running:
                    for chunk in drain_queue(audio_queue):
                        prob = vad.is_speech(chunk)
                        barge_frames.append(chunk)
                        if (
                            barge_onset is None
                            and prob >= barge_in_threshold
                            and time.monotonic() - last_reject_ts
                                >= verify_interval_s
                        ):
                            barge_onset = max(
                                0, len(barge_frames) - 1 - ONSET_PAD
                            )

                    if barge_onset is not None:
                        collected = barge_frames[barge_onset:]
                        if len(collected) * chunk_samples >= verify_audio_samples:
                            sim = speaker.verify(np.concatenate(collected))
                            if sim >= speaker_threshold:
                                llm_abort.set()
                                tts.interrupt()
                                interrupted = True
                                click.echo(f"\n  打断！(声纹 {sim:.2f})")
                                log.info("barge_in", similarity=round(sim, 3))
                                break
                            click.echo(
                                f"\r  声纹不符 ({sim:.2f} < {speaker_threshold:.2f})"
                                + " " * 10,
                                nl=False,
                            )
                            last_reject_ts = time.monotonic()
                            barge_onset = None

                    time.sleep(0.05)

                if interrupted and keep_barge_audio and barge_onset is not None:
                    # 起音点之后的语音作为下一句开头续接，无需重说；
                    # 不清队列：打断后用户还在说的部分继续收
                    preroll_frames = barge_frames[barge_onset:]
                    state = "LISTENING"
                else:
                    # 冷却：丢弃播放期间积累的回声帧
                    clear_queue(audio_queue)
                    time.sleep(post_tts_cooldown_s)
                    clear_queue(audio_queue)
                    vad.reset()
                    state = "LISTENING" if (no_wake or continuous) else "IDLE"

            time.sleep(0.01)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error("voice_chat_error", error=str(e))
            click.echo(f"\n  错误: {e}")
            clear_queue(audio_queue)
            vad.reset()
            state = "LISTENING" if no_wake else "IDLE"

    stream.stop()
    stream.close()


if __name__ == "__main__":
    main()
