"""
src/voice/vad.py
Silero VAD 封装：语音活动检测，端点检测

注意：Silero VAD 是带内部 LSTM 状态的流式模型，使用时必须遵守两条纪律：
  1. 同一实例只能按时间顺序喂帧，不能对同一帧重复推理（会污染内部状态）
  2. 不同音频流（如唤醒检测与打断检测并发时）应使用独立实例
在语音段边界（状态切换、丢弃积压音频后）调用 reset() 清空流式状态。
"""
from __future__ import annotations

import collections
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
    Silero VAD 封装（每实例独立流式状态，模型很小，可放心多实例）。

    is_speech() 内部维护残余样本缓冲：任意长度的 chunk 会被切成
    512 样本（16kHz）/ 256 样本（8kHz）的完整帧逐帧推理，
    不足一帧的尾部样本留到下一次调用拼接，不丢样本。
    """

    def __init__(self, config: VADConfig | None = None):
        self._cfg = config or VADConfig()
        log.info("vad_loading", model="silero_vad")
        from silero_vad import load_silero_vad
        self._model = load_silero_vad()
        self._model.eval()
        self._frame = 512 if self._cfg.sample_rate == 16000 else 256
        self._residual = np.empty(0, dtype=np.float32)
        self._last_prob = 0.0
        log.info("vad_loaded")

    # ------------------------------------------------------------------
    # 流式状态管理
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空流式状态（模型 LSTM 状态 + 残余样本）。在语音段边界调用。"""
        self._residual = np.empty(0, dtype=np.float32)
        self._last_prob = 0.0
        reset_fn = getattr(self._model, "reset_states", None)
        if callable(reset_fn):
            reset_fn()

    # ------------------------------------------------------------------
    # 核心推理
    # ------------------------------------------------------------------

    def is_speech(self, chunk: np.ndarray) -> float:
        """
        输入按时间顺序到达的音频 chunk（float32, [-1,1]），返回语音概率 [0,1]。
        chunk 长度任意；不足一帧时返回上一次的概率（样本已缓存，下次推理）。
        多帧时返回各帧概率的最大值。
        """
        buf = np.concatenate([self._residual, np.ascontiguousarray(chunk, dtype=np.float32)])
        n_frames = len(buf) // self._frame
        if n_frames == 0:
            self._residual = buf
            return self._last_prob

        max_prob = 0.0
        with torch.no_grad():
            for i in range(n_frames):
                sub = buf[i * self._frame:(i + 1) * self._frame]
                p = float(self._model(torch.from_numpy(sub), self._cfg.sample_rate).item())
                if p > max_prob:
                    max_prob = p
        self._residual = buf[n_frames * self._frame:]
        self._last_prob = max_prob
        return max_prob

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
        pad_chunks = max(1, int(cfg.speech_pad_ms / cfg.chunk_ms))
        min_chunks = int(cfg.min_speech_duration_ms / cfg.chunk_ms)
        max_chunks = int(cfg.max_speech_duration_s * 1000 / cfg.chunk_ms)

        ring_buf: collections.deque[np.ndarray] = collections.deque(maxlen=pad_chunks)
        speech_buf: list[np.ndarray] = []
        in_speech = False
        silent_chunks = 0
        silence_trigger = pad_chunks

        for chunk in audio_stream:
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
