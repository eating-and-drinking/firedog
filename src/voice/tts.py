"""
src/voice/tts.py
语音合成引擎（支持 Kokoro 本地 / ElevenLabs 云端 / edge-tts 备用）
支持打断（barge-in）：播放时可随时调用 interrupt() 立即停止
"""
from __future__ import annotations

import io
import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.utils.logger import get_logger, LatencyLogger
from src.utils.metrics import TTS_REQUESTS

log = get_logger(__name__)


@dataclass
class TTSConfig:
    backend: str = "kokoro"
    voice: str = "zh_female_1"
    speed: float = 1.1
    sample_rate: int = 22050
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
        if self._cfg.backend == "kokoro":
            return self._kokoro_synth(text)
        elif self._cfg.backend == "elevenlabs":
            return self._elevenlabs_synth(text)
        elif self._cfg.backend == "edge_tts":
            return self._edge_tts_synth(text)
        raise ValueError(f"Unknown TTS backend: {self._cfg.backend}")

    def _kokoro_synth(self, text: str) -> bytes:
        from kokoro_onnx import Kokoro
        kokoro = Kokoro("kokoro-v1.9.onnx", "voices-v1.0.bin")
        samples, sr = kokoro.create(text, voice=self._cfg.voice, speed=self._cfg.speed, lang="zh-cn")
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
        """异步播放文本，支持 interrupt() 打断。"""
        audio_bytes = self.synthesize(text)
        if not audio_bytes:
            if on_done:
                on_done()
            return

        self._interrupt_flag.clear()
        self._play_thread = threading.Thread(
            target=self._playback_loop,
            args=(audio_bytes, on_done),
            daemon=True,
            name="tts_playback",
        )
        self._playing.set()
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

    def _playback_loop(
        self, audio_bytes: bytes, on_done: Callable[[], None] | None
    ) -> None:
        import sounddevice as sd
        import soundfile as sf

        try:
            buf = io.BytesIO(audio_bytes)
            data, sr = sf.read(buf, dtype="float32")
            chunk_size = int(sr * 0.05)  # 50ms 块，便于快速响应打断

            for start in range(0, len(data), chunk_size):
                if self._interrupt_flag.is_set():
                    log.debug("tts_playback_interrupted")
                    break
                chunk = data[start : start + chunk_size]
                sd.play(chunk, samplerate=sr, blocking=True)
        except Exception as exc:
            log.error("tts_playback_error", error=str(exc))
        finally:
            self._playing.clear()
            self._interrupt_flag.clear()
            if on_done:
                on_done()

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
