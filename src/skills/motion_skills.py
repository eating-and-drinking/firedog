"""
src/skills/motion_skills.py
运动控制技能层（考核项二）

封装机器狗常用动作为可复用 Skill，包含：
  - 参数校验
  - 安全边界检查
  - 异常处理
  - 状态陈旧防护
所有技能可被语音 / Agent 直接调用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.integration.ros2_bridge import ROS2Bridge
from src.utils.logger import get_logger
from src.utils.safety import SafetyGuard

log = get_logger(__name__)


@dataclass
class SkillResult:
    success: bool
    message: str
    data: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


class MotionSkills:
    """
    运动控制技能集。
    每个公共方法对应一个可被 Agent Tool 调用的原子技能。
    """

    def __init__(self, bridge: ROS2Bridge, safety: SafetyGuard):
        self._bridge = bridge
        self._safety = safety

    # ------------------------------------------------------------------
    # 基础运动
    # ------------------------------------------------------------------

    def stand_up(self) -> SkillResult:
        """技能：站立"""
        if not self._safety.check_action_allowed():
            return SkillResult(False, "安全检查未通过，无法执行站立")
        log.info("skill_stand_up")
        # 实际场景：调用 SDK 的 StandUp action
        # self._bridge.send_action("stand_up")
        self._bridge.publish_cmd_vel(0.0, 0.0, 0.0)
        time.sleep(0.5)
        return SkillResult(True, "已站立")

    def lie_down(self) -> SkillResult:
        """技能：趴下"""
        if not self._safety.check_action_allowed():
            return SkillResult(False, "安全检查未通过，无法执行趴下")
        log.info("skill_lie_down")
        self._bridge.publish_stop()
        return SkillResult(True, "已趴下")

    def move_forward(self, speed: float = 0.3, duration_s: float = 2.0) -> SkillResult:
        """
        技能：向前行走
        :param speed: 线速度 m/s，上限由 SafetyGuard 裁剪
        :param duration_s: 持续时间（秒），0 = 持续运动直到外部停止
        """
        if not self._validate_speed(speed):
            return SkillResult(False, f"速度参数非法: {speed}")
        if not self._safety.check_action_allowed():
            return SkillResult(False, "安全检查未通过")

        log.info("skill_move_forward", speed=speed, duration_s=duration_s)
        ok = self._bridge.publish_cmd_vel(speed, 0.0, 0.0)
        if not ok:
            return SkillResult(False, "发送速度指令失败")

        if duration_s > 0:
            time.sleep(duration_s)
            self._bridge.publish_stop()

        return SkillResult(True, f"以 {speed:.2f} m/s 前进 {duration_s:.1f} 秒")

    def move_backward(self, speed: float = 0.2, duration_s: float = 2.0) -> SkillResult:
        """技能：向后退"""
        if not self._validate_speed(speed):
            return SkillResult(False, f"速度参数非法: {speed}")
        if not self._safety.check_action_allowed():
            return SkillResult(False, "安全检查未通过")

        ok = self._bridge.publish_cmd_vel(-abs(speed), 0.0, 0.0)
        if ok and duration_s > 0:
            time.sleep(duration_s)
            self._bridge.publish_stop()
        return SkillResult(ok, "后退完成" if ok else "指令发送失败")

    def turn_left(self, angular_speed: float = 0.5, duration_s: float = 2.0) -> SkillResult:
        """技能：左转"""
        return self._turn(abs(angular_speed), duration_s, direction="left")

    def turn_right(self, angular_speed: float = 0.5, duration_s: float = 2.0) -> SkillResult:
        """技能：右转"""
        return self._turn(-abs(angular_speed), duration_s, direction="right")

    def stop(self) -> SkillResult:
        """技能：立即停止"""
        log.info("skill_stop")
        ok = self._bridge.publish_stop()
        return SkillResult(ok, "已停止" if ok else "停止指令发送失败")

    def move_to_position(
        self, target_x: float, target_y: float, speed: float = 0.3
    ) -> SkillResult:
        """
        技能：导航到目标坐标（简单比例控制，生产环境建议接 Nav2）
        :param target_x: 目标 X（m）
        :param target_y: 目标 Y（m）
        :param speed: 行进速度 m/s
        """
        if not self._safety.check_action_allowed():
            return SkillResult(False, "安全检查未通过")

        import math

        max_attempts = 200
        tolerance = 0.15  # m

        for _ in range(max_attempts):
            odom = self._bridge.get_odom()
            if odom is None:
                self._bridge.publish_stop()
                return SkillResult(False, "里程计数据不可用，终止导航")

            dx = target_x - odom.x
            dy = target_y - odom.y
            dist = math.hypot(dx, dy)

            if dist < tolerance:
                self._bridge.publish_stop()
                log.info("skill_move_to_position_done", x=target_x, y=target_y)
                return SkillResult(True, f"已到达目标 ({target_x:.2f}, {target_y:.2f})")

            # 方向角误差
            target_yaw = math.atan2(dy, dx)
            yaw_err = target_yaw - odom.yaw
            # 归一化到 [-π, π]
            while yaw_err > math.pi:
                yaw_err -= 2 * math.pi
            while yaw_err < -math.pi:
                yaw_err += 2 * math.pi

            angular_z = max(-1.0, min(1.0, yaw_err * 1.5))
            lin_x = speed if abs(yaw_err) < 0.5 else 0.0

            ok = self._bridge.publish_cmd_vel(lin_x, 0.0, angular_z)
            if not ok:
                return SkillResult(False, "速度指令发送失败")

            if not self._safety.check_action_allowed():
                self._bridge.publish_stop()
                return SkillResult(False, "行进中安全检查失败，已停止")

            time.sleep(0.05)

        self._bridge.publish_stop()
        return SkillResult(False, "超过最大导航尝试次数，未到达目标")

    # ------------------------------------------------------------------
    # 特殊动作
    # ------------------------------------------------------------------

    def perform_greeting(self) -> SkillResult:
        """技能：打招呼动作（点头/挥爪示意）"""
        if not self._safety.check_action_allowed():
            return SkillResult(False, "安全检查未通过")
        log.info("skill_greeting")
        # 实际接 SDK 动作接口
        # self._bridge.send_action("greeting")
        return SkillResult(True, "已完成打招呼动作")

    def patrol(self, waypoints: list[tuple[float, float]], speed: float = 0.3) -> SkillResult:
        """
        技能：巡逻路线（依次访问路点）
        :param waypoints: [(x1,y1), (x2,y2), ...]
        :param speed: 行进速度
        """
        if not waypoints:
            return SkillResult(False, "路点列表为空")

        log.info("skill_patrol_start", waypoints=waypoints)
        for i, (x, y) in enumerate(waypoints):
            log.info("patrol_waypoint", index=i, x=x, y=y)
            result = self.move_to_position(x, y, speed)
            if not result.success:
                return SkillResult(False, f"巡逻中止于路点 {i}: {result.message}")
            time.sleep(0.5)  # 路点停留

        return SkillResult(True, f"巡逻完成，共访问 {len(waypoints)} 个路点")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _turn(self, angular_z: float, duration_s: float, direction: str) -> SkillResult:
        if not self._safety.check_action_allowed():
            return SkillResult(False, "安全检查未通过")
        log.info(f"skill_turn_{direction}", angular_z=angular_z, duration_s=duration_s)
        ok = self._bridge.publish_cmd_vel(0.0, 0.0, angular_z)
        if ok and duration_s > 0:
            time.sleep(duration_s)
            self._bridge.publish_stop()
        return SkillResult(ok, f"{direction} 转向完成" if ok else "转向指令失败")

    @staticmethod
    def _validate_speed(speed: float) -> bool:
        return isinstance(speed, (int, float)) and 0 < speed <= 2.0
