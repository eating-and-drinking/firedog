"""
src/voice/denoise.py
流式降噪前端（RNNoise）

RNNoise（Xiph/Mozilla）是生产语音框架（Pipecat 等）的标准选择：
CPU 实时（实测 2.3ms/80ms chunk）、自带权重。本机实测算法延迟 ~24ms
（均匀延迟，不影响端点检测），噪声底压制 ~17dB。

接入位置：麦克风回调内，软件增益之后、入队之前：
  raw → mic_gain → denoise → audio_queue → VAD/ASR/声纹

⚠️ 默认关闭（config voice.denoise.enabled: false）。实测结论（2026-06）：
  - SenseVoice 对 0dB 白噪/电机噪声下的语音识别已全对，降噪伪影反而出错字
  - Silero VAD 对纯电机噪声 0% 误触发，降噪后反升到 2%
  仅在强广播噪声环境（户外风噪/人群）导致待机 VAD 持续误触发时开启。

pyrnnoise 按 480 样本粒度产出且带内部缓冲，输出长度与输入 chunk 不对齐；
主循环的静音计数依赖"每 chunk = 80ms"的假设，因此这里用 FIFO 把输出
重新切成与输入等长（起步阶段不足的部分左侧补零，稳态后流量守恒）。

⚠️ pyrnnoise 0.4.x 的内部重采样（sample_rate≠48000 时）是坏的——输出近乎
全零（实测 RMS 0.0001 vs 正常 0.07）。因此 RNNoise 固定跑 48kHz 原生模式，
16k↔48k 用 soxr 流式重采样器（librosa 自带依赖，有状态、无块边界伪影）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DenoiseConfig:
    enabled: bool = False           # 默认关闭，理由见模块 docstring
    backend: str = "rnnoise"        # rnnoise | none
    sample_rate: int = 16000


class Denoiser:
    """
    流式降噪器。process() 输入输出均为 float32 [-1,1]、长度相同。
    后端加载失败时自动降级为直通（backend == "none"），不阻塞启动。
    """

    def __init__(self, config: DenoiseConfig | None = None):
        self._cfg = config or DenoiseConfig()
        self._backend = "none"
        self._rn = None
        self._up = self._down = None
        self._fifo = np.empty(0, dtype=np.float32)

        if not self._cfg.enabled:
            return
        if self._cfg.backend == "none":
            return
        if self._cfg.backend != "rnnoise":
            log.warning("denoise_unknown_backend", backend=self._cfg.backend)
            return
        try:
            from pyrnnoise import RNNoise
            self._rn = RNNoise(sample_rate=48000)
            self._make_resamplers()
            self._backend = "rnnoise"
            log.info("denoise_loaded", backend="rnnoise",
                     sample_rate=self._cfg.sample_rate)
        except Exception as exc:
            log.warning("denoise_unavailable", backend="rnnoise", error=str(exc))

    def _make_resamplers(self) -> None:
        """16k↔48k 流式重采样器（sample_rate=48000 时直通）。"""
        if self._cfg.sample_rate == 48000:
            self._up = self._down = None
            return
        import soxr
        self._up = soxr.ResampleStream(
            self._cfg.sample_rate, 48000, 1, dtype="float32"
        )
        self._down = soxr.ResampleStream(
            48000, self._cfg.sample_rate, 1, dtype="float32"
        )

    @property
    def backend(self) -> str:
        return self._backend

    def reset(self) -> None:
        """清空输出对齐缓冲与重采样状态（RNNoise 自身的流式状态保持）。"""
        self._fifo = np.empty(0, dtype=np.float32)
        if self._backend == "rnnoise":
            self._make_resamplers()

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        降噪一个音频 chunk（float32 [-1,1]，任意长度）。
        返回等长的降噪结果；后端不可用或异常时原样返回。
        """
        if self._backend != "rnnoise" or len(chunk) == 0:
            return chunk
        try:
            x = np.clip(chunk, -1.0, 1.0).astype(np.float32)
            if self._up is not None:
                x = self._up.resample_chunk(x)
            pcm48 = (x * 32767).astype(np.int16)
            denoised48: list[np.ndarray] = []
            for _prob, frame in self._rn.denoise_chunk(pcm48[None, :]):
                denoised48.append(frame[0])
            if denoised48:
                y = np.concatenate(denoised48).astype(np.float32) / 32768.0
                if self._down is not None:
                    y = self._down.resample_chunk(y)
                self._fifo = np.concatenate([self._fifo, y])

            n = len(chunk)
            if len(self._fifo) >= n:
                out = self._fifo[:n]
                self._fifo = self._fifo[n:]
            else:
                # 起步阶段内部缓冲未填满：左侧补零保持节拍
                pad = np.zeros(n - len(self._fifo), dtype=np.float32)
                out = np.concatenate([pad, self._fifo])
                self._fifo = np.empty(0, dtype=np.float32)
            return out
        except Exception as exc:
            log.error("denoise_error", error=str(exc))
            return chunk
