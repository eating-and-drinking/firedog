"""
src/utils/metrics.py
Prometheus 指标采集，暴露在 /metrics HTTP 端点
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.utils.logger import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------
# 语音指标
# ------------------------------------------------------------------
WAKE_WORD_DETECTIONS = Counter(
    "wake_word_detections_total",
    "唤醒词检出次数",
    ["result"],  # result: true_positive | false_positive
)

ASR_REQUESTS = Counter("asr_requests_total", "ASR 识别请求次数", ["status"])

TTS_REQUESTS = Counter("tts_requests_total", "TTS 合成请求次数", ["backend"])

VOICE_LATENCY_E2E = Histogram(
    "voice_e2e_latency_seconds",
    "用户说完 → 机器狗开始播报 端到端时延",
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
)

BARGE_IN_LATENCY = Histogram(
    "barge_in_latency_seconds",
    "用户打断 → 停止播报 时延",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0],
)

# ------------------------------------------------------------------
# Agent 指标
# ------------------------------------------------------------------
AGENT_TOOL_CALLS = Counter(
    "agent_tool_calls_total", "Agent 工具调用次数", ["tool_name", "status"]
)

AGENT_TASK_DURATION = Histogram(
    "agent_task_duration_seconds",
    "完整任务执行时长",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

# ------------------------------------------------------------------
# 机器狗状态指标
# ------------------------------------------------------------------
ROBOT_BATTERY = Gauge("robot_battery_percent", "机器狗电量百分比")
ROBOT_EMERGENCY_STOP = Gauge("robot_emergency_stop", "急停状态 (1=停止 0=正常)")


def start_metrics_server(port: int = 8000) -> None:
    """启动 Prometheus HTTP 指标服务（非阻塞）。"""
    start_http_server(port)
    log.info("metrics_server_started", port=port)
