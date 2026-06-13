"""
src/voice/wake_word.py
唤醒词检测（优先级从高到低自动降级）
  1. sherpa-onnx KWS：中文流式关键词检测（zipformer-wenetspeech 3.3M，CPU 实时，
     ~80ms 级延迟；关键词运行时用拼音 token 定义，无需训练）
  2. OpenWakeWord：英文唤醒词模型（本项目场景基本用不上）
  3. ASR 兜底：VAD 端点检测 + SenseVoice 整段转写 + 关键词匹配
     （延迟高 = 说完 + ~800ms 静音 + ASR；待机时 GPU 常驻）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class WakeWordConfig:
    model: str = "hey_jarvis"          # openwakeword 模型名 / .onnx 文件路径
    threshold: float = 0.5             # OWW 触发阈值
    chunk_size: int = 1280             # 80ms @ 16kHz
    sample_rate: int = 16000
    # ASR 兜底关键词（不区分大小写，去空格匹配）
    keywords: list = field(
        default_factory=lambda: ["你好小狗", "嘿小狗", "机器狗", "小狗小狗"]
    )
    # 拼音模糊匹配：容忍 ASR 近音错字（"机器狗"→"一气狗"），每音节编辑距离 ≤1
    fuzzy_pinyin: bool = True
    # sherpa-onnx KWS（中文唤醒主路）
    sherpa_model_dir: str = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    sherpa_threshold: float = 0.25     # keywords_threshold，越低越灵敏（误唤醒也越多）


class WakeWordDetector:
    """
    唤醒词检测器。

    使用方式（主循环）:
        oww 模式 → 逐 chunk 调用 process_chunk(chunk)
        asr 模式 → 在 VAD 收到完整语音片段后调用 check_text(asr_result)
    """

    def __init__(self, config: WakeWordConfig):
        self._cfg = config
        self._oww: Optional[object] = None
        self._sherpa = None
        self._sherpa_stream = None
        self._mode = "asr"
        self._try_load_sherpa()
        if self._mode != "sherpa":
            self._try_load_oww()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _keyword_to_sherpa_tokens(self, word: str, token_set: set) -> Optional[str]:
        """
        汉字词 → sherpa-onnx token 序列（声母 + 带声调韵母）。
        非汉字字符（如 "小Q" 的 Q）映射为大写字母 token（WenetSpeech 词表含 A-Z）。
        任一 token 不在词表中则返回 None（该关键词不可用）。
        """
        from pypinyin import pinyin, Style
        inits = pinyin(word, style=Style.INITIALS, strict=True)
        finals = pinyin(word, style=Style.FINALS_TONE, strict=True)
        tokens: list[str] = []
        for ini, fin in zip(inits, finals):
            i, f = ini[0], fin[0]
            if i == f:
                # 非汉字字符原样返回的情况（字母/数字）
                if len(i) == 1 and i.isalpha():
                    tokens.append(i.upper())
                    continue
                return None
            if i:
                tokens.append(i)
            if f:
                tokens.append(f)
        if not tokens:
            return None
        for t in tokens:
            if t not in token_set:
                log.warning("wake_word_sherpa_token_missing", word=word, token=t)
                return None
        return " ".join(tokens)

    def _try_load_sherpa(self) -> None:
        model_dir = Path(self._cfg.sherpa_model_dir)
        if not model_dir.exists():
            log.info("wake_word_sherpa_model_missing", dir=str(model_dir))
            return
        try:
            import sherpa_onnx

            tokens_file = model_dir / "tokens.txt"
            token_set = {
                line.split()[0]
                for line in tokens_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

            lines = []
            for kw in self._cfg.keywords:
                toks = self._keyword_to_sherpa_tokens(kw, token_set)
                if toks:
                    lines.append(f"{toks} @{kw}")
                else:
                    log.warning("wake_word_sherpa_keyword_skipped", keyword=kw)
            if not lines:
                log.warning("wake_word_sherpa_no_usable_keywords")
                return

            kw_file = model_dir / "keywords_firedog.txt"
            kw_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            self._sherpa = sherpa_onnx.KeywordSpotter(
                tokens=str(tokens_file),
                encoder=str(model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
                decoder=str(model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
                joiner=str(model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
                keywords_file=str(kw_file),
                keywords_threshold=float(self._cfg.sherpa_threshold),
                sample_rate=self._cfg.sample_rate,
                num_threads=2,
                provider="cpu",
            )
            self._sherpa_stream = self._sherpa.create_stream()
            self._mode = "sherpa"
            log.info(
                "wake_word_sherpa_loaded",
                keywords=[ln.rsplit("@", 1)[-1] for ln in lines],
                threshold=self._cfg.sherpa_threshold,
            )
        except Exception as exc:
            log.info("wake_word_sherpa_unavailable", reason=str(exc))
            self._sherpa = None
            self._sherpa_stream = None

    def _try_load_oww(self) -> None:
        model_spec = self._cfg.model
        try:
            from openwakeword.model import Model  # type: ignore

            # 文件路径 or 模型名称
            model_list = [model_spec] if Path(model_spec).exists() else [model_spec]
            self._oww = Model(wakeword_models=model_list, inference_framework="onnx")
            self._mode = "oww"
            log.info("wake_word_oww_loaded", model=model_spec)
        except Exception as exc:
            log.info(
                "wake_word_oww_unavailable",
                reason=str(exc),
                fallback="asr_keyword",
            )
            self._oww = None
            self._mode = "asr"

    # ------------------------------------------------------------------
    # 主路：sherpa-onnx KWS / OpenWakeWord（逐 chunk 调用）
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> float:
        """
        输入 80ms 音频 chunk（float32, 16kHz），返回唤醒概率 [0, 1]。
        sherpa 模式命中返回 1.0；asr 模式始终返回 0。
        """
        if self._mode == "sherpa":
            stream = self._sherpa_stream
            stream.accept_waveform(
                self._cfg.sample_rate, np.ascontiguousarray(chunk, dtype=np.float32)
            )
            while self._sherpa.is_ready(stream):
                self._sherpa.decode_stream(stream)
            result = self._sherpa.get_result(stream)
            if result:
                self._sherpa.reset_stream(stream)
                log.info("wake_word_hit", keyword=result, match="sherpa_kws")
                return 1.0
            return 0.0

        if self._oww is None:
            return 0.0
        chunk_int16 = (chunk * 32767).astype(np.int16)
        pred: dict = self._oww.predict(chunk_int16)
        return float(max(pred.values())) if pred else 0.0

    # ------------------------------------------------------------------
    # 兜底：ASR 关键词匹配（精确 + 模糊）
    # ------------------------------------------------------------------

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """两个短字符串的 Levenshtein 编辑距离（音节级比较，长度都很短）。"""
        if a == b:
            return 0
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(
                    prev[j] + 1,          # 删除
                    cur[j - 1] + 1,       # 插入
                    prev[j - 1] + (ca != cb),  # 替换
                ))
            prev = cur
        return prev[-1]

    @staticmethod
    def _normalize_syllable(s: str) -> str:
        """模糊拼音归一化：卷舌/平舌、前/后鼻音在 ASR 错字中常混淆。"""
        if s.startswith(("zh", "ch", "sh")):
            s = s[0] + s[2:]
        if s.endswith("ng"):
            s = s[:-1]
        return s

    def _fuzzy_pinyin_match(self, cleaned: str, kw_c: str) -> bool:
        """
        拼音级模糊匹配：SenseVoice 对孤立短语常出近音错字
        （"机器狗"→"一气狗"/"激情狗"），汉字子串匹配会漏掉。
        规则：音节先做模糊归一化（zh→z、ng→n 等），关键词拼音序列
        在文本拼音序列上滑窗，窗口内每个音节编辑距离 ≤1 即命中。
        """
        try:
            from pypinyin import lazy_pinyin
        except ImportError:
            return False
        kw_py = [self._normalize_syllable(s) for s in lazy_pinyin(kw_c)]
        text_py = [self._normalize_syllable(s) for s in lazy_pinyin(cleaned)]
        n = len(kw_py)
        if n == 0 or len(text_py) < n:
            return False
        for start in range(len(text_py) - n + 1):
            window = text_py[start : start + n]
            if all(
                self._edit_distance(s, k) <= 1
                for s, k in zip(window, kw_py)
            ):
                return True
        return False

    def check_text(self, text: str) -> float:
        """
        对 ASR 识别结果做关键词检测。
        1) 汉字精确子串匹配 → 返回 1.0
        2) 拼音模糊匹配（容忍每音节 1 个编辑距离）→ 返回 0.9
        只保留汉字做比较（去标点/空格）。
        单字关键词误唤醒率过高（日常对话随机命中），强制要求 ≥ 2 字。
        """
        import re as _re
        # 保留汉字 + 英文字母，转小写，去空格
        cleaned = _re.sub(r'[^一-鿿a-zA-Z]', '', text).lower()
        if not cleaned:
            return 0.0
        for kw in self._cfg.keywords:
            kw_c = _re.sub(r'[^一-鿿a-zA-Z]', '', kw).lower()
            if len(kw_c) < 2:
                log.warning("wake_word_keyword_too_short", keyword=kw)
                continue
            if kw_c in cleaned:
                log.info("wake_word_hit", keyword=kw, text=text, match="exact")
                return 1.0
            if self._cfg.fuzzy_pinyin and self._fuzzy_pinyin_match(cleaned, kw_c):
                log.info("wake_word_hit", keyword=kw, text=text, match="fuzzy_pinyin")
                return 0.9
        return 0.0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """'oww' 或 'asr'"""
        return self._mode

    @property
    def threshold(self) -> float:
        return self._cfg.threshold
