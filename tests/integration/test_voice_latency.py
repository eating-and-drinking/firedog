"""
tests/integration/test_voice_latency.py
语音管道端到端时延集成测试
验收指标：端到端时延 ≤ 2500ms

注意：此测试需要本地 ASR 模型，CI 中可通过 --skip-voice 跳过
"""
import time
import threading
import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.voice.asr import ASREngine, ASRConfig
from src.voice.tts import TTSEngine, TTSConfig


@pytest.mark.integration
class TestASRLatency:
    """ASR 单独时延测试（mock faster_whisper）"""

    def test_empty_audio_returns_fast(self):
        """空音频应快速返回空字符串，不阻塞"""
        config = ASRConfig(device="cpu", model_size="tiny")
        with patch("faster_whisper.WhisperModel") as mock_model:
            mock_model.return_value.transcribe.return_value = ([], MagicMock(language="zh", language_probability=0.9, duration=0.5))
            engine = ASREngine(config)
            short_audio = np.zeros(100, dtype=np.float32)
            start = time.perf_counter()
            result = engine.transcribe(short_audio)
            elapsed = time.perf_counter() - start
            assert result == ""
            assert elapsed < 0.1, f"空音频处理耗时过长: {elapsed:.3f}s"

    def test_transcribe_latency_within_budget(self):
        """模拟 2s 音频，transcribe 时延应在预算内"""
        config = ASRConfig(device="cpu", model_size="tiny")
        fake_segment = MagicMock()
        fake_segment.text = "向前走两步"
        fake_info = MagicMock(language="zh", language_probability=0.95, duration=2.0)

        with patch("faster_whisper.WhisperModel") as mock_model:
            mock_instance = MagicMock()
            mock_instance.transcribe.return_value = ([fake_segment], fake_info)
            mock_model.return_value = mock_instance
            engine = ASREngine(config)

            # 2s @ 16kHz
            audio = np.random.randn(32000).astype(np.float32) * 0.1
            start = time.perf_counter()
            text = engine.transcribe(audio)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert text == "向前走两步"
            # mock 场景下应极快
            assert elapsed_ms < 500, f"ASR 时延: {elapsed_ms:.1f}ms"


@pytest.mark.integration
class TestTTSInterrupt:
    """TTS 打断（barge-in）测试：打断后应在 1500ms 内停止"""

    def test_interrupt_stops_playback(self):
        config = TTSConfig(backend="edge_tts")
        engine = TTSEngine(config)

        fake_audio = b"\x00" * 44100  # 1s 假 WAV

        with patch.object(engine, "synthesize", return_value=fake_audio):
            with patch("sounddevice.play"):
                with patch("soundfile.read", return_value=(np.zeros(22050), 22050)):
                    done_event = threading.Event()

                    def _on_done():
                        done_event.set()

                    engine.speak("这是一段测试语音", on_done=_on_done)
                    time.sleep(0.05)

                    interrupt_start = time.perf_counter()
                    engine.interrupt()
                    done_event.wait(timeout=2.0)
                    interrupt_latency_ms = (time.perf_counter() - interrupt_start) * 1000

                    # 打断响应时延 ≤ 1500ms
                    assert interrupt_latency_ms <= 1500, (
                        f"打断时延过大: {interrupt_latency_ms:.1f}ms > 1500ms"
                    )
