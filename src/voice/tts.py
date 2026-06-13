"""
src/voice/tts.py
语音合成引擎（支持 CosyVoice3 本地 / Kokoro 本地 / ElevenLabs 云端 / edge-tts 备用）

speak_stream(sentences) 接收句子迭代器（如 LLM 流式输出），
合成与播放在两个线程中流水线进行：播放第 N 句时并行合成第 N+1 句，
首句合成完即开始出声，大幅降低首音延迟。
支持打断（barge-in）：播放时可随时调用 interrupt() 立即停止。
"""
from __future__ import annotations

import io
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from src.utils.logger import get_logger, LatencyLogger
from src.utils.metrics import TTS_REQUESTS

log = get_logger(__name__)


@dataclass
class TTSConfig:
    backend: str = "cosyvoice"
    # CosyVoice3
    model_dir: str = "./Fun-CosyVoice3-0.5B-2512"
    prompt_speech: str = ""      # 零样本克隆参考音频路径
    prompt_text: str = ""        # 参考音频的文字内容（必须与音频一致）
    instruct: str = ""           # 指令控制（如 "请用开心的语气说。"）
    # 通用
    speed: float = 1.0
    sample_rate: int = 24000
    output_device: int | None = None  # 播放设备索引，None=系统默认
    # Kokoro
    kokoro_voice: str = "zh_female_1"
    # ElevenLabs
    elevenlabs_voice_id: str = ""
    elevenlabs_api_key: str = ""


class TTSEngine:
    """
    语音合成引擎。
    synthesize() 返回 bytes（WAV/PCM），
    speak() 异步播放并支持中途 interrupt()。
    """

    def __init__(self, config: TTSConfig):
        self._cfg = config
        self._playing = threading.Event()
        self._interrupt_flag = threading.Event()
        self._play_thread: threading.Thread | None = None
        # CosyVoice3 模型懒加载，只初始化一次
        self._cosyvoice_model = None

    # ------------------------------------------------------------------
    # 合成
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> bytes:
        """将文本合成为 WAV bytes。"""
        if not text.strip():
            return b""
        with LatencyLogger(log, "tts_synthesize", backend=self._cfg.backend):
            try:
                audio_bytes = self._synthesize_impl(text)
                TTS_REQUESTS.labels(backend=self._cfg.backend).inc()
                return audio_bytes
            except Exception as exc:
                log.error("tts_error", error=str(exc))
                return b""

    def _synthesize_impl(self, text: str) -> bytes:
        if self._cfg.backend == "cosyvoice":
            return self._cosyvoice_synth(text)
        elif self._cfg.backend == "kokoro":
            return self._kokoro_synth(text)
        elif self._cfg.backend == "elevenlabs":
            return self._elevenlabs_synth(text)
        elif self._cfg.backend == "edge_tts":
            return self._edge_tts_synth(text)
        raise ValueError(f"Unknown TTS backend: {self._cfg.backend}")

    def _get_cosyvoice_model(self):
        """懒加载 CosyVoice3 模型（仅首次调用时加载）。"""
        if self._cosyvoice_model is not None:
            return self._cosyvoice_model
        import sys
        cosyvoice_repo = Path(self._cfg.model_dir).parent / "CosyVoice"
        if cosyvoice_repo.exists() and str(cosyvoice_repo) not in sys.path:
            sys.path.insert(0, str(cosyvoice_repo))
        matcha_path = cosyvoice_repo / "third_party" / "Matcha-TTS"
        if matcha_path.exists() and str(matcha_path) not in sys.path:
            sys.path.insert(0, str(matcha_path))
        from cosyvoice.cli.cosyvoice import AutoModel
        log.info("cosyvoice_loading_model", model_dir=self._cfg.model_dir)
        self._cosyvoice_model = AutoModel(model_dir=self._cfg.model_dir)
        log.info("cosyvoice_model_loaded", sample_rate=self._cosyvoice_model.sample_rate)
        return self._cosyvoice_model

    def _cosyvoice_synth(self, text: str) -> bytes:
        """使用 CosyVoice3 合成语音。"""
        model = self._get_cosyvoice_model()

        # 确定参考音频路径（优先用配置，退回 asset 目录）
        ref_wav = self._cfg.prompt_speech
        if not ref_wav:
            ref_wav = str(Path(self._cfg.model_dir) / "asset" / "zero_shot_prompt.wav")
        if not Path(ref_wav).exists():
            log.warning("cosyvoice_no_prompt_speech", hint="请在配置中设置 prompt_speech 参考音频路径")
            return b""

        # 参考音频的文字内容（CosyVoice3 要求末尾必须有 <|endofprompt|>）
        ref_text = self._cfg.prompt_text or "希望你以后能够做的比我还好哟。"
        if not ref_text.endswith("<|endofprompt|>"):
            ref_text = ref_text + "<|endofprompt|>"

        if self._cfg.instruct:
            instruct_text = ref_text + self._cfg.instruct
            gen = model.inference_instruct2(
                text, instruct_text, ref_wav, stream=False
            )
        else:
            gen = model.inference_zero_shot(
                text, ref_text, ref_wav, stream=False
            )

        # 取最后一个 chunk（非流式模式只有1个）
        speech = None
        for chunk in gen:
            speech = chunk['tts_speech']

        if speech is None:
            log.error("cosyvoice_no_output")
            return b""

        # speech 是 torch.Tensor，转为 numpy 再转 WAV
        samples = speech.squeeze().cpu().numpy()
        duration_s = len(samples) / model.sample_rate
        log.info("cosyvoice_synth_done", duration_s=round(duration_s, 3), samples=len(samples))
        if duration_s < 0.1:
            log.warning("cosyvoice_short_output", text=text, duration_s=duration_s)
        return self._numpy_to_wav(samples, model.sample_rate)

    def _kokoro_synth(self, text: str) -> bytes:
        from kokoro_onnx import Kokoro
        kokoro = Kokoro("kokoro-v1.9.onnx", "voices-v1.0.bin")
        samples, sr = kokoro.create(text, voice=self._cfg.kokoro_voice, speed=self._cfg.speed, lang="zh-cn")
        return self._numpy_to_wav(samples, sr)

    def _elevenlabs_synth(self, text: str) -> bytes:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=self._cfg.elevenlabs_api_key)
        audio = client.generate(
            text=text,
            voice=self._cfg.elevenlabs_voice_id,
            model="eleven_multilingual_v2",
        )
        return b"".join(audio)

    def _edge_tts_synth(self, text: str) -> bytes:
        import asyncio
        import edge_tts

        async def _run():
            communicate = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        return asyncio.run(_run())

    # ------------------------------------------------------------------
    # 播放 & 打断
    # ------------------------------------------------------------------

    def speak(self, text: str, on_done: Callable[[], None] | None = None) -> None:
        """异步播放单段文本，支持 interrupt() 打断。"""
        self.speak_stream(iter([text]), on_done=on_done)

    def speak_stream(
        self,
        sentences: Iterable[str],
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """
        流水线播放句子序列（通常来自 LLM 流式输出）。

        合成线程逐句消费 sentences 并产出 WAV；播放线程同时消费 WAV 出声。
        speak_stream 立即返回，is_speaking() 在整个流播完/被打断前保持 True。
        interrupt() 会停止播放、跳过未播句子，并通过提前终止消费让
        sentences 生成器收到 GeneratorExit（上游可据此停止 LLM 生成）。
        """
        self._interrupt_flag.clear()
        self._playing.set()

        # 合成→播放的句间缓冲；容量小以限制打断时的浪费
        audio_q: queue.Queue[bytes | None] = queue.Queue(maxsize=4)

        def _synth_worker() -> None:
            try:
                for sentence in sentences:
                    if self._interrupt_flag.is_set() or not self._playing.is_set():
                        break
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    wav = self.synthesize(sentence)
                    while (
                        wav
                        and not self._interrupt_flag.is_set()
                        and self._playing.is_set()
                    ):
                        try:
                            audio_q.put(wav, timeout=0.2)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                log.error("tts_synth_stream_error", error=str(exc))
            finally:
                audio_q.put(None)  # sentinel：通知播放线程结束

        def _play_worker() -> None:
            import sounddevice as sd
            import soundfile as sf

            stream: sd.OutputStream | None = None
            try:
                while True:
                    wav = audio_q.get()
                    if wav is None:
                        break
                    if self._interrupt_flag.is_set():
                        continue  # 排空队列直到 sentinel，让合成线程退出
                    data, sr = sf.read(io.BytesIO(wav), dtype="float32")
                    chunk_size = int(sr * 0.05)  # 50ms 块，便于快速响应打断
                    if stream is None:
                        stream = sd.OutputStream(
                            samplerate=sr,
                            channels=1,
                            dtype="float32",
                            device=self._cfg.output_device,
                            blocksize=chunk_size,
                        )
                        stream.start()
                    for start in range(0, len(data), chunk_size):
                        if self._interrupt_flag.is_set():
                            log.debug("tts_playback_interrupted")
                            break
                        stream.write(data[start : start + chunk_size].reshape(-1, 1))
            except Exception as exc:
                log.error("tts_playback_error", error=str(exc))
                self._interrupt_flag.set()  # 释放可能阻塞在队列上的合成线程
            finally:
                if stream is not None:
                    stream.stop()
                    stream.close()
                self._playing.clear()
                self._interrupt_flag.clear()
                if on_done:
                    on_done()

        self._synth_thread = threading.Thread(
            target=_synth_worker, daemon=True, name="tts_synth"
        )
        self._play_thread = threading.Thread(
            target=_play_worker, daemon=True, name="tts_playback"
        )
        self._synth_thread.start()
        self._play_thread.start()

    def interrupt(self) -> None:
        """
        打断正在播放的语音（barge-in）。
        验收指标：用户开口 → 停止播报 ≤ 1.5s
        """
        if self._playing.is_set():
            self._interrupt_flag.set()
            log.info("tts_interrupted")

    def is_speaking(self) -> bool:
        return self._playing.is_set()

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _numpy_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        import struct
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            pcm = (samples * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
