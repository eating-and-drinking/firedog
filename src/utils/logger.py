"""
src/utils/logger.py
结构化日志工具，支持 JSON / 文本双模式输出
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any

import structlog


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """初始化全局日志配置（应在程序入口处调用一次）。"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=100 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format="%(message)s",
    )

    # 屏蔽第三方库过多日志
    for noisy in ("httpx", "httpcore", "openai", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    use_json = os.environ.get("LOG_JSON", "false").lower() == "true"

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
            if use_json
            else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """获取带名称绑定的结构化日志器。"""
    return structlog.get_logger(name)


class LatencyLogger:
    """上下文管理器：计量代码块耗时并写入日志。"""

    def __init__(self, logger: structlog.BoundLogger, operation: str, **ctx: Any):
        self._log = logger
        self._op = operation
        self._ctx = ctx
        self._start: float = 0.0

    def __enter__(self) -> "LatencyLogger":
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        import time
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        if exc_type is None:
            self._log.info(
                "latency",
                operation=self._op,
                elapsed_ms=round(elapsed_ms, 2),
                **self._ctx,
            )
        else:
            self._log.warning(
                "latency_with_error",
                operation=self._op,
                elapsed_ms=round(elapsed_ms, 2),
                error=str(exc_val),
                **self._ctx,
            )
