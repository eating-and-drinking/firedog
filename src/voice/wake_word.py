"""
src/voice/wake_word.py
OpenWakeWord 唤醒词检测
验收指标：唤醒成功率 ≥ 95%，误唤醒 ≤ 1次/小时（安静环境）
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.utils.logger import get_logger
from src.utils.metrics import WAKE_WORD_DETECTIONS

log = get_logger(__name__)


@dataclass
class WakeWordConfig:
    model_name: str = "hey_robot"
    threshold: float = 0.7
    chunk_size: int = 1280          # 80ms @ 16kHz
    sample_rate: int = 16000
    # 最小触发间隔（避免重复唤醒）
    cooldown_s: float = 2.0


class WakeWordDetector:
    """
    OpenWakeWord 封装。
    调用 start() 后在后台线程持续监听，检测到唤醒词时调用 on_wake_cb。
    """

    def __init__(
        self,
        config: WakeWordConfig,
        on_wake_cb: Callable[[], None],
    ):
        self._cfg = config
        self._on_wake = on_wake_cb
        self._running = False
        self._last_wake_ts: float = 0.0
        self._thread: threading.Thread | None = None
        self._oww = self._load_model()

    def _load_model(self):
        try:
            import openwakeword
            from openwakeword.model import Model
            log.info("wake_word_loading", model=self._cfg.model_name)
            model = Model(
                wakeword_models=[self._cfg.model_name],
                inference_framework="onnx",
            )
            log.info("wake_word_loaded", model=self._cfg.model_name)
            return model
        except Exception as exc:
            log.error("wake_word_load_failed", error=str(exc))
            raise

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def start(self, audio_queue: "queue.Queue[np.ndarray]") -> None:
        """
        启动后台检测线程。
        audio_queue: 音频生产者不断 put 的 float32 numpy 帧
        """
        self._running = True
        self._audio_q = audio_queue
        self._thread = threading.Thread(
            target=self._detection_loop, daemon=True, name="wake_word_detector"
        )
        self._thread.start()
        log.info("wake_word_started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        log.info("wake_word_stopped")

    # ------------------------------------------------------------------
    # 内部循环
    # ------------------------------------------------------------------

    def _detection_loop(self) -> None:
        import queue as q_mod

        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.5)
            except q_mod.Empty:
                continue

            # OpenWakeWord 期望 int16 输入
            chunk_int16 = (chunk * 32767).astype(np.int16)
            predictions = self._oww.predict(chunk_int16)

            score = predictions.get(self._cfg.model_name, 0.0)
            if score >= self._cfg.threshold:
                now = time.monotonic()
                if now - self._last_wake_ts >= self._cfg.cooldown_s:
                    self._last_wake_ts = now
                    WAKE_WORD_DETECTIONS.labels(result="true_positive").inc()
                    log.info("wake_word_detected", score=round(score, 3))
                    self._on_wake()
                else:
                    log.debug("wake_word_cooldown_skip", score=score)
