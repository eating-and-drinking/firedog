"""降噪模块单元测试（CPU，无需模型/硬件）。"""
import numpy as np
import pytest

from src.voice.denoise import Denoiser, DenoiseConfig


@pytest.mark.unit
def test_disabled_passthrough():
    dn = Denoiser(DenoiseConfig(enabled=False))
    x = np.random.default_rng(0).normal(0, 0.05, 1280).astype(np.float32)
    assert dn.backend == "none"
    assert np.array_equal(dn.process(x), x)


@pytest.mark.unit
def test_streaming_length_conservation():
    """任意 chunk 长度下输出与输入等长（主循环按 chunk=80ms 计数静音）。"""
    dn = Denoiser(DenoiseConfig(enabled=True, backend="rnnoise", sample_rate=16000))
    if dn.backend != "rnnoise":
        pytest.skip("pyrnnoise unavailable")
    rng = np.random.default_rng(1)
    for n in (1280, 1280, 640, 1280, 320, 1280):
        out = dn.process(rng.normal(0, 0.05, n).astype(np.float32))
        assert len(out) == n
        assert out.dtype == np.float32


@pytest.mark.unit
def test_noise_floor_reduction():
    """稳态白噪应被显著压制（实测约 -17dB，这里宽松断言减半）。"""
    dn = Denoiser(DenoiseConfig(enabled=True, backend="rnnoise", sample_rate=16000))
    if dn.backend != "rnnoise":
        pytest.skip("pyrnnoise unavailable")
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.05, 16000 * 4).astype(np.float32)
    out = np.concatenate(
        [dn.process(noise[i:i + 1280]) for i in range(0, len(noise) - 1279, 1280)]
    )
    # 跳过起步补零段后比较 RMS
    tail_in = noise[16000:]
    tail_out = out[16000:]
    rms_in = np.sqrt((tail_in ** 2).mean())
    rms_out = np.sqrt((tail_out ** 2).mean())
    assert rms_out < rms_in * 0.5


@pytest.mark.unit
def test_reset_clears_alignment_buffer():
    dn = Denoiser(DenoiseConfig(enabled=True, backend="rnnoise", sample_rate=16000))
    if dn.backend != "rnnoise":
        pytest.skip("pyrnnoise unavailable")
    dn.process(np.zeros(1280, dtype=np.float32))
    dn.reset()
    assert len(dn._fifo) == 0
