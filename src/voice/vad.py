"""
src/voice/vad.py
Silero VAD 封装：语音活动检测，端点检测
"""
from __future__ import annotations

import collections
import queue
import threading
from dataclasses import dataclass
from typing import Generator, Iterator

import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class VADConfig:
    threshold: float = 0.5
    speech_pad_ms: int = 400
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = 30.0
    sample_rate: int = 16000
    chunk_ms: int = 80


class SileroVAD:
    """
    Silero VAD 封装。
    支持流式逐帧推理，线程安全。
    """

    _instance: "SileroVAD | None" = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # 单例模式：模型只加载一次
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self, config: VADConfig | None = None):
        if self._initialized:
            return
        self._cfg = config or VADConfig()
        log.info("vad_loading", model="silero_vad")
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._model.eval()
        self._initialized = True
        log.info("vad_loaded")

    # ------------------------------------------------------------------
    # 核心推理
    # ------------------------------------------------------------------

    def is_speech(self, chunk: np.ndarray) -> float:
        """
        判断 chunk 是否含语音。
        chunk: float32, shape=(chunk_size,), 归一化到 [-1, 1]
        返回语音概率 [0, 1]
        """
        tensor = torch.from_numpy(chunk).float()
        with torch.no_grad():
            prob = self._model(tensor, self._cfg.sample_rate).item()
        return float(prob)

    # ------------------------------------------------------------------
    # 流式语音片段生成器
    # ------------------------------------------------------------------

    def iter_speech_segments(
        self, audio_stream: Iterator[np.ndarray]
    ) -> Generator[np.ndarray, None, None]:
        """
        消费原始音频帧流，在检测到完整语音片段后 yield 拼接的 numpy 数组。
        实现了前后 padding 与最大时长保护。
        """
        cfg = self._cfg
        chunk_samples = int(cfg.sample_rate * cfg.chunk_ms / 1000)
        pad_chunks = int(cfg.speech_pad_ms / cfg.chunk_ms)
        min_chunks = int(cfg.min_speech_duration_ms / cfg.chunk_ms)
        max_chunks = int(cfg.max_speech_duration_s * 1000 / cfg.chunk_ms)

        ring_buf: collections.deque[np.ndarray] = collections.deque(maxlen=pad_chunks)
        speech_buf: list[np.ndarray] = []
        in_speech = False
        silent_chunks = 0
        silence_trigger = pad_chunks

        for chunk in audio_stream:
            if len(chunk) != chunk_samples:
                chunk = np.resize(chunk, chunk_samples)

            prob = self.is_speech(chunk)
            is_speech = prob >= cfg.threshold

            if is_speech:
                if not in_speech:
                    # 语音起始：把 padding ring 里的帧一起加入
                    speech_buf.extend(ring_buf)
                    in_speech = True
                    silent_chunks = 0
                speech_buf.append(chunk)
                silent_chunks = 0

                if len(speech_buf) >= max_chunks:
                    log.warning("vad_max_duration_reached")
                    yield np.concatenate(speech_buf)
                    speech_buf.clear()
                    in_speech = False
            else:
                ring_buf.append(chunk)
                if in_speech:
                    speech_buf.append(chunk)
                    silent_chunks += 1
                    if silent_chunks >= silence_trigger:
                        if len(speech_buf) >= min_chunks:
                            yield np.concatenate(speech_buf)
                        speech_buf.clear()
                        in_speech = False
                        silent_chunks = 0
