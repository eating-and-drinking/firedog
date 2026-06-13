"""
src/voice/speaker_id.py
声纹注册与验证（说话人验证）

优先使用 resemblyzer（GE2E d-vector）。
如未安装 resemblyzer，自动降级为基于 torchaudio MFCC 的余弦相似度
（精度较低，但无需额外依赖）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SpeakerIDConfig:
    enabled: bool = True
    similarity_threshold: float = 0.75  # 允许打断的最低相似度
    enroll_min_s: float = 1.0           # 注册音频最短时长（秒）
    verify_min_s: float = 0.5           # 验证音频最短时长（秒）
    sample_rate: int = 16000


class SpeakerVerifier:
    """
    说话人验证器。

    使用流程：
        1. 唤醒时 enroll(audio) → 注册声纹
        2. 打断时 verify(audio) → 返回与注册声纹的相似度 [0, 1]
        3. 会话结束 clear() → 清除注册声纹
    """

    def __init__(self, config: SpeakerIDConfig):
        self._cfg = config
        self._enrolled: Optional[np.ndarray] = None
        self._backend = "disabled"
        self._encoder = None
        self._preprocess = None

        if not config.enabled:
            return
        self._load()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _load(self) -> None:
        # 优先 resemblyzer（语者 d-vector，较好的跨语言泛化）
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
            self._encoder = VoiceEncoder()
            self._preprocess = preprocess_wav
            self._backend = "resemblyzer"
            log.info("speaker_id_loaded", backend="resemblyzer")
            return
        except Exception as exc:
            log.info("speaker_id_resemblyzer_unavailable", reason=str(exc))

        # 降级：torchaudio MFCC 余弦相似度（无需额外依赖）
        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            self._backend = "mfcc"
            log.info("speaker_id_loaded", backend="mfcc_fallback")
        except Exception as exc:
            log.warning("speaker_id_load_failed", error=str(exc))

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def enroll(self, audio: np.ndarray) -> bool:
        """
        注册说话人声纹。
        audio: float32 numpy 数组，16kHz，至少 enroll_min_s 秒。
        返回 True 表示注册成功。
        """
        if self._backend == "disabled":
            return False
        min_samples = int(self._cfg.enroll_min_s * self._cfg.sample_rate)
        if len(audio) < min_samples:
            log.warning(
                "speaker_enroll_too_short",
                samples=len(audio),
                need=min_samples,
            )
            return False

        try:
            self._enrolled = self._extract(audio)
            log.info("speaker_enrolled", backend=self._backend)
            return True
        except Exception as exc:
            log.error("speaker_enroll_error", error=str(exc))
            return False

    def verify(self, audio: np.ndarray) -> float:
        """
        验证说话人。返回与注册声纹的余弦相似度 [0, 1]。
        若未注册或 backend 不可用，返回 1.0（放行所有人）。
        """
        if self._backend == "disabled" or self._enrolled is None:
            return 1.0
        min_samples = int(self._cfg.verify_min_s * self._cfg.sample_rate)
        if len(audio) < min_samples:
            return 1.0  # 音频太短，无法判断，放行
        try:
            emb = self._extract(audio)
            sim = float(
                np.dot(self._enrolled, emb)
                / (np.linalg.norm(self._enrolled) * np.linalg.norm(emb) + 1e-8)
            )
            return float(np.clip(sim, 0.0, 1.0))
        except Exception as exc:
            log.error("speaker_verify_error", error=str(exc))
            return 0.0

    def update(
        self,
        audio: np.ndarray,
        weight: float = 0.3,
        min_similarity: float = 0.65,
    ) -> Optional[float]:
        """
        用一段确认是当前用户的干净语音（无 TTS 播放时录得的对话轮次）
        对注册声纹做指数滑动平均微调——唤醒时的一次性注册往往偏脏，
        多轮对话后声纹会越来越准。

        与原声纹相似度低于 min_similarity 时不更新（可能是别人插话），
        返回 None；更新成功返回该语音与原声纹的相似度。
        """
        if self._backend == "disabled" or self._enrolled is None:
            return None
        if len(audio) < int(self._cfg.verify_min_s * self._cfg.sample_rate):
            return None
        try:
            emb = self._extract(audio)
            sim = float(
                np.dot(self._enrolled, emb)
                / (np.linalg.norm(self._enrolled) * np.linalg.norm(emb) + 1e-8)
            )
            if sim < min_similarity:
                log.debug("speaker_update_skipped", similarity=round(sim, 3))
                return None
            mixed = (1.0 - weight) * self._enrolled + weight * emb
            self._enrolled = mixed / (np.linalg.norm(mixed) + 1e-8)
            log.debug("speaker_refined", similarity=round(sim, 3))
            return sim
        except Exception as exc:
            log.error("speaker_update_error", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # 内部：特征提取
    # ------------------------------------------------------------------

    def _extract(self, audio: np.ndarray) -> np.ndarray:
        if self._backend == "resemblyzer":
            wav = self._preprocess(audio, source_sr=self._cfg.sample_rate)
            return self._encoder.embed_utterance(wav)

        if self._backend == "mfcc":
            return self._extract_mfcc(audio)

        raise RuntimeError(f"Unknown backend: {self._backend}")

    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        import torch
        import torchaudio

        tensor = torch.from_numpy(audio).unsqueeze(0)
        mfcc_tf = torchaudio.transforms.MFCC(
            sample_rate=self._cfg.sample_rate,
            n_mfcc=40,
            melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 40},
        )
        mfcc = mfcc_tf(tensor)  # (1, 40, T)
        mean = mfcc.mean(-1).squeeze(0)
        std = mfcc.std(-1).squeeze(0)
        emb = torch.cat([mean, std], dim=0).numpy()  # (80,)
        return emb / (np.linalg.norm(emb) + 1e-8)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """清除注册声纹（会话结束时调用）。"""
        self._enrolled = None
        log.debug("speaker_cleared")

    @property
    def has_speaker(self) -> bool:
        return self._enrolled is not None

    @property
    def enabled(self) -> bool:
        return self._backend != "disabled"

    @property
    def threshold(self) -> float:
        return self._cfg.similarity_threshold
