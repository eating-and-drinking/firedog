"""
src/utils/safety.py
安全守卫：速度限制、电量检查、姿态监测、急停
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RobotState:
    battery_percent: float = 100.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    joint_temps: dict[str, float] = field(default_factory=dict)
    is_emergency_stopped: bool = False


class SafetyGuard:
    """
    集中式安全检查器。
    所有技能层写入运动指令前必须通过 check_velocity() 或 check_action()。
    电量、温度、姿态监控在后台线程中异步运行。
    """

    def __init__(
        self,
        max_linear_vel: float = 0.8,
        max_angular_vel: float = 1.5,
        battery_low: float = 15.0,
        battery_critical: float = 5.0,
        joint_temp_max: float = 80.0,
        tilt_max_deg: float = 45.0,
        emergency_stop_cb: Callable[[], None] | None = None,
    ):
        self._max_lin = max_linear_vel
        self._max_ang = max_angular_vel
        self._bat_low = battery_low
        self._bat_crit = battery_critical
        self._temp_max = joint_temp_max
        self._tilt_max = tilt_max_deg
        self._e_stop_cb = emergency_stop_cb
        self._state = RobotState()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def update_state(self, **kwargs: float | dict) -> None:
        """由 ROS 2 / SDK 监听回调调用，更新机器狗实时状态。"""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)
        self._run_checks()

    def clip_velocity(
        self, linear_x: float, linear_y: float, angular_z: float
    ) -> tuple[float, float, float]:
        """裁剪速度到安全范围，返回裁剪后的值。"""
        lin = (linear_x**2 + linear_y**2) ** 0.5
        if lin > self._max_lin:
            scale = self._max_lin / lin
            linear_x *= scale
            linear_y *= scale

        angular_z = max(-self._max_ang, min(self._max_ang, angular_z))
        return linear_x, linear_y, angular_z

    def check_action_allowed(self) -> bool:
        """
        返回 True 表示可执行动作，False 时应拒绝指令并告知用户原因。
        """
        with self._lock:
            if self._state.is_emergency_stopped:
                log.warning("safety_block", reason="emergency_stop_active")
                return False
            if self._state.battery_percent <= self._bat_crit:
                log.warning(
                    "safety_block",
                    reason="battery_critical",
                    battery=self._state.battery_percent,
                )
                return False
        return True

    def trigger_emergency_stop(self, reason: str = "manual") -> None:
        """触发急停，调用注册的回调（通常发送零速指令到 ROS 2 /cmd_vel）。"""
        with self._lock:
            self._state.is_emergency_stopped = True
        log.error("emergency_stop_triggered", reason=reason)
        if self._e_stop_cb:
            self._e_stop_cb()

    def reset_emergency_stop(self) -> None:
        with self._lock:
            self._state.is_emergency_stopped = False
        log.info("emergency_stop_reset")

    # ------------------------------------------------------------------
    # 内部检查
    # ------------------------------------------------------------------

    def _run_checks(self) -> None:
        with self._lock:
            state = self._state

        # 电量警告
        if state.battery_percent <= self._bat_crit:
            self.trigger_emergency_stop(
                f"battery_critical_{state.battery_percent:.1f}pct"
            )
            return
        if state.battery_percent <= self._bat_low:
            log.warning("battery_low", battery=state.battery_percent)

        # 温度警告
        for joint, temp in state.joint_temps.items():
            if temp >= self._temp_max:
                self.trigger_emergency_stop(f"joint_overheat_{joint}_{temp:.1f}C")
                return

        # 姿态警告（倾倒检测）
        if (
            abs(state.roll_deg) > self._tilt_max
            or abs(state.pitch_deg) > self._tilt_max
        ):
            self.trigger_emergency_stop(
                f"tilt_detected_roll={state.roll_deg:.1f}_pitch={state.pitch_deg:.1f}"
            )
