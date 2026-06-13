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
    """
    初始化全局日志配置（应在程序入口处调用一次）。

    structlog 事件必须经由 stdlib logging 分发（LoggerFactory + ProcessorFormatter），
    否则只会 print 到终端、永远进不了日志文件——排障时最需要的
    asr_result/barge_in/声纹分数等事件全部丢失（曾经的真实事故）。
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    use_json = os.environ.get("LOG_JSON", "false").lower() == "true"

    # 第三方库的 stdlib 日志也会经过这条链，补上时间戳/级别
    foreign_pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
            foreign_pre_chain=foreign_pre_chain,
        )
    )
    handlers: list[logging.Handler] = [console_handler]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer()
                if use_json
                else structlog.dev.ConsoleRenderer(colors=False),
                foreign_pre_chain=foreign_pre_chain,
            )
        )
        handlers.append(file_handler)

    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(numeric_level)

    # 屏蔽第三方库过多日志
    for noisy in ("httpx", "httpcore", "openai", "funasr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
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
