"""
src/voice/asr.py
FasterWhisper 语音识别封装
验收指标：常用指令字准率 ≥ 90%
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger, LatencyLogger
from src.utils.metrics import ASR_REQUESTS

log = get_logger(__name__)


@dataclass
class ASRConfig:
    backend: str = "faster_whisper"
    model_size: str = "medium"
    language: str = "zh"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    sample_rate: int = 16000


class ASREngine:
    """
    语音识别引擎，默认使用 FasterWhisper（本地）。
    线程安全（内置锁，避免并发加载模型）。
    """

    def __init__(self, config: ASRConfig):
        self._cfg = config
        self._model = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self) -> None:
        if self._cfg.backend == "faster_whisper":
            self._load_faster_whisper()
        else:
            raise ValueError(f"Unsupported ASR backend: {self._cfg.backend}")

    def _load_faster_whisper(self) -> None:
        from faster_whisper import WhisperModel

        log.info(
            "asr_loading",
            backend="faster_whisper",
            model=self._cfg.model_size,
            device=self._cfg.device,
        )
        self._model = WhisperModel(
            self._cfg.model_size,
            device=self._cfg.device,
            compute_type=self._cfg.compute_type,
        )
        log.info("asr_loaded")

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        """
        将 float32 音频数组转录为文本。
        audio: shape=(N,), float32, 采样率 = config.sample_rate

        返回: 识别文本（strip 后）；识别失败返回空字符串
        """
        if audio is None or len(audio) < self._cfg.sample_rate * 0.2:
            log.debug("asr_audio_too_short", samples=len(audio) if audio is not None else 0)
            return ""

        with self._lock:
            with LatencyLogger(log, "asr_transcribe"):
                try:
                    result = self._transcribe_impl(audio)
                    ASR_REQUESTS.labels(status="success").inc()
                    log.info("asr_result", text=result[:80])
                    return result
                except Exception as exc:
                    ASR_REQUESTS.labels(status="error").inc()
                    log.error("asr_error", error=str(exc))
                    return ""

    def _transcribe_impl(self, audio: np.ndarray) -> str:
        segments, info = self._model.transcribe(
            audio,
            language=self._cfg.language,
            beam_size=self._cfg.beam_size,
            vad_filter=self._cfg.vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
        )
        text = "".join(seg.text for seg in segments).strip()
        log.debug(
            "asr_detail",
            language=info.language,
            language_prob=round(info.language_probability, 3),
            duration=round(info.duration, 2),
        )
        return text
