"""
src/skills/sensor_skills.py
传感器查询技能层（考核项二）

封装 IMU、里程计、电量等传感器数据为 Agent 可调用的技能
"""
from __future__ import annotations

from src.integration.ros2_bridge import ROS2Bridge
from src.skills.motion_skills import SkillResult
from src.utils.logger import get_logger

log = get_logger(__name__)


class SensorSkills:
    """传感器数据查询技能集。"""

    def __init__(self, bridge: ROS2Bridge):
        self._bridge = bridge

    def get_battery_level(self) -> SkillResult:
        """技能：查询电量"""
        battery = self._bridge.get_battery()
        if battery is None:
            return SkillResult(False, "无法获取电量数据", data=None)
        msg = f"当前电量 {battery.percent:.1f}%"
        if battery.charging:
            msg += "（充电中）"
        return SkillResult(True, msg, data={"percent": battery.percent, "charging": battery.charging})

    def get_position(self) -> SkillResult:
        """技能：查询当前位置"""
        odom = self._bridge.get_odom()
        if odom is None:
            return SkillResult(False, "里程计数据不可用", data=None)
        return SkillResult(
            True,
            f"当前位置 X={odom.x:.2f}m, Y={odom.y:.2f}m",
            data={"x": odom.x, "y": odom.y, "yaw": odom.yaw},
        )

    def get_imu_status(self) -> SkillResult:
        """技能：查询姿态（IMU）"""
        imu = self._bridge.get_imu()
        if imu is None:
            return SkillResult(False, "IMU 数据不可用", data=None)
        return SkillResult(
            True,
            f"姿态 roll={imu.roll:.1f}° pitch={imu.pitch:.1f}°",
            data={"roll": imu.roll, "pitch": imu.pitch, "yaw": imu.yaw},
        )

    def get_full_status(self) -> SkillResult:
        """技能：查询综合状态（位置+姿态+电量）"""
        pos = self.get_position()
        imu = self.get_imu_status()
        bat = self.get_battery_level()

        parts = []
        data: dict = {}
        for result in [pos, imu, bat]:
            parts.append(result.message)
            if result.data:
                data.update(result.data)

        return SkillResult(True, "；".join(parts), data=data)
