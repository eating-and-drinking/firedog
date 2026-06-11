"""
src/voice/voice_pipeline.py
端到端双向语音管道（考核项一核心）

完整链路：
  唤醒词检测 → VAD → ASR → LLM → TTS → 播放
  支持 barge-in 打断、回声消除、多轮连续对话

验收指标：
  - 唤醒成功率 ≥ 95%，误唤醒 ≤ 1次/小时
  - ASR 字准率 ≥ 90%
  - 端到端时延 ≤ 2.5s
  - 打断响应 ≤ 1.5s
  - 长时间运行无明显卡顿/崩溃/内存泄漏
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np
import sounddevice as sd

from src.voice.asr import ASREngine, ASRConfig
from src.voice.tts import TTSEngine, TTSConfig
from src.voice.vad import SileroVAD, VADConfig
from src.voice.wake_word import WakeWordDetector, WakeWordConfig
from src.utils.logger import get_logger, LatencyLogger
from src.utils.metrics import VOICE_LATENCY_E2E, BARGE_IN_LATENCY

log = get_logger(__name__)


class PipelineState(Enum):
    IDLE = auto()           # 等待唤醒词
    LISTENING = auto()      # 检测到唤醒词，录音中
    PROCESSING = auto()     # ASR + LLM 处理中
    SPEAKING = auto()       # TTS 播放中
    INTERRUPTED = auto()    # 被打断


@dataclass
class VoicePipelineConfig:
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    sample_rate: int = 16000
    input_device: int | None = None
    output_device: int | None = None
    silence_timeout_s: float = 5.0
    # 连续对话：说完后保持监听而不回到等待唤醒状态
    continuous_conversation: bool = True
    continuous_window_s: float = 15.0


class VoicePipeline:
    """
    全双工语音管道，协调唤醒→VAD→ASR→LLM→TTS全链路。
    LLM 回调由外部注入（解耦语音与 Agent 层）。
    """

    def __init__(
        self,
        config: VoicePipelineConfig,
        llm_handler: Callable[[str], str],
    ):
        self._cfg = config
        self._llm_handler = llm_handler
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()

        # 音频队列：麦克风→唤醒词检测
        self._wake_audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)

        # 唤醒后的语音片段队列
        self._speech_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=10)

        # 子模块
        self._vad = SileroVAD(config.vad)
        self._asr = ASREngine(config.asr)
        self._tts = TTSEngine(config.tts)
        self._wake_detector = WakeWordDetector(
            config.wake_word,
            on_wake_cb=self._on_wake_detected,
        )

        self._running = False
        self._stream: sd.InputStream | None = None
        self._barge_in_start_ts: float = 0.0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动音频流与后台线程。"""
        self._running = True

        # 后台：唤醒词检测
        self._wake_detector.start(self._wake_audio_q)

        # 后台：语音处理工作者
        self._worker_thread = threading.Thread(
            target=self._processing_worker, daemon=True, name="voice_worker"
        )
        self._worker_thread.start()

        # 打开麦克风音频流
        self._stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self._cfg.sample_rate * self._cfg.vad.chunk_ms / 1000),
            device=self._cfg.input_device,
            callback=self._audio_callback,
        )
        self._stream.start()
        log.info("voice_pipeline_started")

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._wake_detector.stop()
        self._speech_q.put(None)  # sentinel
        log.info("voice_pipeline_stopped")

    # ------------------------------------------------------------------
    # 音频回调（运行在 sounddevice 线程中，须非阻塞）
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            log.warning("audio_stream_status", status=str(status))
        chunk = indata[:, 0].copy()

        # 始终送给唤醒词检测器
        try:
            self._wake_audio_q.put_nowait(chunk)
        except queue.Full:
            pass  # 背压：丢弃最旧帧

        # 如果正在监听，检查 barge-in（TTS 播放中用户开口）
        with self._state_lock:
            state = self._state

        if state == PipelineState.SPEAKING:
            prob = self._vad.is_speech(chunk)
            if prob >= self._cfg.vad.threshold:
                self._trigger_barge_in()

        elif state == PipelineState.LISTENING:
            try:
                self._speech_q.put_nowait(chunk)
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # 唤醒词回调
    # ------------------------------------------------------------------

    def _on_wake_detected(self) -> None:
        with self._state_lock:
            if self._state == PipelineState.SPEAKING:
                # 说话时唤醒 = 打断
                self._trigger_barge_in()
                return
            if self._state != PipelineState.IDLE:
                return
            self._state = PipelineState.LISTENING
        log.info("pipeline_state", state="LISTENING")

    # ------------------------------------------------------------------
    # 打断（barge-in）
    # ------------------------------------------------------------------

    def _trigger_barge_in(self) -> None:
        self._barge_in_start_ts = time.monotonic()
        self._tts.interrupt()
        with self._state_lock:
            self._state = PipelineState.LISTENING
        log.info("barge_in_triggered")

    # ------------------------------------------------------------------
    # 语音处理工作者
    # ------------------------------------------------------------------

    def _processing_worker(self) -> None:
        """
        消费语音帧队列 → VAD 分段 → ASR → LLM → TTS
        """
        while self._running:
            with self._state_lock:
                state = self._state

            if state != PipelineState.LISTENING:
                time.sleep(0.05)
                continue

            # 从队列中收集音频帧，直到 VAD 检测到完整语音片段
            audio_frames: list[np.ndarray] = []
            listen_start = time.monotonic()
            speech_detected = False

            while time.monotonic() - listen_start < self._cfg.silence_timeout_s:
                try:
                    chunk = self._speech_q.get(timeout=0.1)
                except queue.Empty:
                    if speech_detected:
                        break
                    continue

                audio_frames.append(chunk)
                prob = self._vad.is_speech(chunk)

                if prob >= self._cfg.vad.threshold:
                    speech_detected = True

                # 检测到语音结束（连续静音）
                if speech_detected:
                    silent_streak = sum(
                        1 for f in audio_frames[-8:]
                        if self._vad.is_speech(f) < self._cfg.vad.threshold
                    )
                    if silent_streak >= 6:
                        break

            if not audio_frames or not speech_detected:
                with self._state_lock:
                    self._state = PipelineState.IDLE
                log.debug("no_speech_detected_timeout")
                continue

            # 记录打断延迟
            if self._barge_in_start_ts > 0:
                barge_latency = time.monotonic() - self._barge_in_start_ts
                BARGE_IN_LATENCY.observe(barge_latency)
                log.info("barge_in_latency", ms=round(barge_latency * 1000, 1))
                self._barge_in_start_ts = 0.0

            audio = np.concatenate(audio_frames)
            e2e_start = time.monotonic()

            with self._state_lock:
                self._state = PipelineState.PROCESSING

            # ASR
            user_text = self._asr.transcribe(audio)
            if not user_text.strip():
                with self._state_lock:
                    self._state = PipelineState.IDLE
                continue

            log.info("user_said", text=user_text)

            # LLM
            try:
                response_text = self._llm_handler(user_text)
            except Exception as exc:
                log.error("llm_handler_error", error=str(exc))
                response_text = "抱歉，我处理出错了，请再说一遍。"

            if not response_text:
                with self._state_lock:
                    self._state = PipelineState.IDLE
                continue

            # 记录端到端时延（LLM 返回时）
            e2e_latency = time.monotonic() - e2e_start
            VOICE_LATENCY_E2E.observe(e2e_latency)
            log.info(
                "e2e_latency",
                ms=round(e2e_latency * 1000, 1),
                target_ms=2500,
                ok=e2e_latency <= 2.5,
            )

            with self._state_lock:
                self._state = PipelineState.SPEAKING

            # TTS 播放
            def _on_tts_done():
                with self._state_lock:
                    if self._cfg.continuous_conversation:
                        self._state = PipelineState.LISTENING
                    else:
                        self._state = PipelineState.IDLE

            self._tts.speak(response_text, on_done=_on_tts_done)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        with self._state_lock:
            return self._state
