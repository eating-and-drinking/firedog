"""
src/voice/asr.py
SenseVoice-Small 语音识别封装（基于 FunASR）
验收指标：常用指令字准率 ≥ 90%
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger, LatencyLogger
from src.utils.metrics import ASR_REQUESTS

log = get_logger(__name__)


@dataclass
class ASRConfig:
    backend: str = "sensevoice"
    model_id: str = "iic/SenseVoice-Small"
    language: str = "zh"            # auto / zh / en / ja / ko / yue
    use_itn: bool = True            # 逆文本归一化（数字转文字等）
    device: str = "cuda"            # cuda / cpu
    sample_rate: int = 16000


class ASREngine:
    """
    语音识别引擎，使用 SenseVoice-Small（FunASR）。
    线程安全（内置锁，避免并发加载模型）。
    """

    def __init__(self, config: ASRConfig):
        self._cfg = config
        self._model = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self) -> None:
        if self._cfg.backend == "sensevoice":
            self._load_sensevoice()
        else:
            raise ValueError(f"Unsupported ASR backend: {self._cfg.backend}")

    def _load_sensevoice(self) -> None:
        from pathlib import Path
        from funasr import AutoModel

        model_id = self._cfg.model_id
        # 相对路径解析为绝对路径，避免 FunASR 把它当远程模型 ID 处理
        if model_id.startswith("./") or model_id.startswith("../"):
            model_id = str(Path(model_id).resolve())

        log.info(
            "asr_loading",
            backend="sensevoice",
            model=model_id,
            device=self._cfg.device,
        )
        self._model = AutoModel(
            model=model_id,
            device=self._cfg.device,
            disable_update=True,
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
        result = self._model.generate(
            input=audio.astype(np.float32),
            language=self._cfg.language,
            use_itn=self._cfg.use_itn,
        )

        # result 是 list，每个元素是 dict，key "text" 存识别结果
        if result and isinstance(result, list):
            text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
        else:
            text = ""

        # 去除 SenseVoice 输出的特殊标签，如 <|zh|><|NEUTRAL|><|Speech|><|withitn|>
        text = re.sub(r"<\|[^|>]+\|>", "", text)

        log.debug(
            "asr_detail",
            language=self._cfg.language,
            raw_result=str(result)[:200] if result else "",
        )
        return text.strip()
