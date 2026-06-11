"""
src/skills/skill_registry.py
技能注册表：统一管理所有可被 Agent 调用的技能
"""
from __future__ import annotations

from typing import Any, Callable

from src.skills.motion_skills import MotionSkills, SkillResult
from src.skills.sensor_skills import SensorSkills
from src.utils.logger import get_logger

log = get_logger(__name__)


class SkillRegistry:
    """
    技能注册表：维护 技能名称 → 可调用对象 的映射。
    Agent 通过 execute(name, **kwargs) 调用任意已注册技能。
    """

    def __init__(self, motion: MotionSkills, sensor: SensorSkills):
        self._skills: dict[str, Callable[..., SkillResult]] = {}
        self._register_defaults(motion, sensor)

    def _register_defaults(self, motion: MotionSkills, sensor: SensorSkills) -> None:
        # 运动技能
        self.register("stand_up", motion.stand_up, "让机器狗站立")
        self.register("lie_down", motion.lie_down, "让机器狗趴下")
        self.register("stop", motion.stop, "立即停止所有运动")
        self.register("move_forward", motion.move_forward, "向前行走，参数: speed(m/s), duration_s")
        self.register("move_backward", motion.move_backward, "向后退，参数: speed(m/s), duration_s")
        self.register("turn_left", motion.turn_left, "向左转，参数: angular_speed, duration_s")
        self.register("turn_right", motion.turn_right, "向右转，参数: angular_speed, duration_s")
        self.register("move_to_position", motion.move_to_position, "导航至坐标，参数: target_x, target_y, speed")
        self.register("patrol", motion.patrol, "按路点巡逻，参数: waypoints(list[tuple]), speed")
        self.register("greeting", motion.perform_greeting, "执行打招呼动作")

        # 传感器技能
        self.register("get_battery", sensor.get_battery_level, "查询电量")
        self.register("get_position", sensor.get_position, "查询当前坐标")
        self.register("get_imu", sensor.get_imu_status, "查询姿态")
        self.register("get_status", sensor.get_full_status, "查询综合状态")

    def register(
        self,
        name: str,
        fn: Callable[..., SkillResult],
        description: str = "",
    ) -> None:
        self._skills[name] = fn
        self._skills[name].__doc__ = description or fn.__doc__
        log.debug("skill_registered", name=name)

    def execute(self, name: str, **kwargs: Any) -> SkillResult:
        if name not in self._skills:
            available = ", ".join(self._skills.keys())
            return SkillResult(False, f"技能 '{name}' 不存在。可用技能: {available}")
        try:
            log.info("skill_executing", name=name, kwargs=kwargs)
            result = self._skills[name](**kwargs)
            log.info("skill_done", name=name, success=result.success, msg=result.message)
            return result
        except TypeError as exc:
            return SkillResult(False, f"技能参数错误: {exc}")
        except Exception as exc:
            log.error("skill_error", name=name, error=str(exc))
            return SkillResult(False, f"技能执行异常: {exc}")

    def list_skills(self) -> dict[str, str]:
        return {
            name: (fn.__doc__ or "").strip()
            for name, fn in self._skills.items()
        }
