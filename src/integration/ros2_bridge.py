"""
src/integration/ros2_bridge.py
ROS 2 通信桥：封装与机器狗本体的所有 ROS 2 通信
支持运行时降级到 mock 模式（无 ROS 2 环境时）
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.utils.logger import get_logger
from src.utils.safety import SafetyGuard

log = get_logger(__name__)

# ---------- 数据类 ----------

@dataclass
class Twist:
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0


@dataclass
class IMUData:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class OdomData:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class BatteryState:
    percent: float = 100.0
    voltage: float = 0.0
    charging: bool = False
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class JointStates:
    names: list[str] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    efforts: list[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


# 数据陈旧超时（秒）
STALE_TIMEOUT = 2.0


class ROS2Bridge:
    """
    封装全部 ROS 2 pub/sub 操作。
    - 发布：/cmd_vel、motion action
    - 订阅：/odom、/imu/data、/battery_state、/joint_states
    在没有 ROS 2 的环境中自动降级到 mock 实现。
    """

    def __init__(
        self,
        namespace: str = "/robot_dog",
        safety: SafetyGuard | None = None,
        mock: bool = False,
    ):
        self._ns = namespace
        self._safety = safety
        self._mock = mock
        self._lock = threading.Lock()

        # 缓存最新传感器数据
        self._imu: Optional[IMUData] = None
        self._odom: Optional[OdomData] = None
        self._battery: Optional[BatteryState] = None
        self._joints: Optional[JointStates] = None

        # 回调注册
        self._on_imu: list[Callable[[IMUData], None]] = []
        self._on_odom: list[Callable[[OdomData], None]] = []
        self._on_battery: list[Callable[[BatteryState], None]] = []

        if not mock:
            self._init_ros2()
        else:
            log.warning("ros2_bridge_mock_mode")
            self._start_mock_publisher()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_ros2(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist as RosTwist
            from sensor_msgs.msg import Imu, JointState
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import BatteryState as RosBattery

            rclpy.init(args=None)
            self._node = rclpy.create_node("robot_dog_agent_bridge")

            # Publisher
            self._cmd_vel_pub = self._node.create_publisher(
                RosTwist, f"{self._ns}/cmd_vel", 10
            )

            # Subscribers
            self._node.create_subscription(
                Imu, f"{self._ns}/imu/data", self._imu_cb, 10
            )
            self._node.create_subscription(
                Odometry, f"{self._ns}/odom", self._odom_cb, 10
            )
            self._node.create_subscription(
                RosBattery, f"{self._ns}/battery_state", self._battery_cb, 10
            )
            self._node.create_subscription(
                JointState, f"{self._ns}/joint_states", self._joint_cb, 10
            )

            # ROS spin 在独立线程
            self._spin_thread = threading.Thread(
                target=rclpy.spin,
                args=(self._node,),
                daemon=True,
                name="ros2_spin",
            )
            self._spin_thread.start()
            log.info("ros2_bridge_initialized", namespace=self._ns)

        except ImportError:
            log.warning("rclpy_not_found_fallback_to_mock")
            self._mock = True
            self._start_mock_publisher()
        except Exception as exc:
            log.error("ros2_init_error", error=str(exc))
            self._mock = True
            self._start_mock_publisher()

    # ------------------------------------------------------------------
    # 发布指令
    # ------------------------------------------------------------------

    def publish_cmd_vel(
        self,
        linear_x: float,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> bool:
        """
        发布速度指令到 /cmd_vel。
        经过安全裁剪后发布，返回是否发布成功。
        """
        if self._safety:
            if not self._safety.check_action_allowed():
                log.warning("cmd_vel_blocked_by_safety")
                return False
            linear_x, linear_y, angular_z = self._safety.clip_velocity(
                linear_x, linear_y, angular_z
            )

        if self._mock:
            log.debug(
                "mock_cmd_vel",
                lx=round(linear_x, 3),
                ly=round(linear_y, 3),
                az=round(angular_z, 3),
            )
            return True

        try:
            from geometry_msgs.msg import Twist as RosTwist

            msg = RosTwist()
            msg.linear.x = float(linear_x)
            msg.linear.y = float(linear_y)
            msg.angular.z = float(angular_z)
            self._cmd_vel_pub.publish(msg)
            return True
        except Exception as exc:
            log.error("cmd_vel_publish_error", error=str(exc))
            return False

    def publish_stop(self) -> bool:
        """发布零速指令（急停）。"""
        return self.publish_cmd_vel(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 传感器读取
    # ------------------------------------------------------------------

    def get_imu(self) -> Optional[IMUData]:
        with self._lock:
            data = self._imu
        if data and (time.monotonic() - data.timestamp) > STALE_TIMEOUT:
            log.warning("imu_data_stale")
            return None
        return data

    def get_odom(self) -> Optional[OdomData]:
        with self._lock:
            data = self._odom
        if data and (time.monotonic() - data.timestamp) > STALE_TIMEOUT:
            log.warning("odom_data_stale")
            return None
        return data

    def get_battery(self) -> Optional[BatteryState]:
        with self._lock:
            return self._battery

    def get_joints(self) -> Optional[JointStates]:
        with self._lock:
            return self._joints

    # ------------------------------------------------------------------
    # 回调注册
    # ------------------------------------------------------------------

    def add_imu_listener(self, cb: Callable[[IMUData], None]) -> None:
        self._on_imu.append(cb)

    def add_odom_listener(self, cb: Callable[[OdomData], None]) -> None:
        self._on_odom.append(cb)

    def add_battery_listener(self, cb: Callable[[BatteryState], None]) -> None:
        self._on_battery.append(cb)

    # ------------------------------------------------------------------
    # ROS 2 订阅回调
    # ------------------------------------------------------------------

    def _imu_cb(self, msg) -> None:
        import math
        # 四元数转欧拉角
        q = msg.orientation
        sinr = 2 * (q.w * q.x + q.y * q.z)
        cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr, cosr)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1, min(1, sinp)))

        data = IMUData(
            roll=math.degrees(roll),
            pitch=math.degrees(pitch),
            accel_x=msg.linear_acceleration.x,
            accel_y=msg.linear_acceleration.y,
            accel_z=msg.linear_acceleration.z,
        )
        with self._lock:
            self._imu = data

        if self._safety:
            self._safety.update_state(roll_deg=data.roll, pitch_deg=data.pitch)

        for cb in self._on_imu:
            try:
                cb(data)
            except Exception as exc:
                log.error("imu_listener_error", error=str(exc))

    def _odom_cb(self, msg) -> None:
        data = OdomData(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            linear_vel=msg.twist.twist.linear.x,
            angular_vel=msg.twist.twist.angular.z,
        )
        with self._lock:
            self._odom = data
        for cb in self._on_odom:
            try:
                cb(data)
            except Exception as exc:
                log.error("odom_listener_error", error=str(exc))

    def _battery_cb(self, msg) -> None:
        data = BatteryState(
            percent=msg.percentage * 100,
            voltage=msg.voltage,
            charging=msg.power_supply_status == 1,
        )
        with self._lock:
            self._battery = data
        if self._safety:
            self._safety.update_state(battery_percent=data.percent)
        for cb in self._on_battery:
            try:
                cb(data)
            except Exception as exc:
                log.error("battery_listener_error", error=str(exc))

    def _joint_cb(self, msg) -> None:
        data = JointStates(
            names=list(msg.name),
            positions=list(msg.position),
            velocities=list(msg.velocity),
            efforts=list(msg.effort),
        )
        with self._lock:
            self._joints = data

    # ------------------------------------------------------------------
    # Mock 模式
    # ------------------------------------------------------------------

    def _start_mock_publisher(self) -> None:
        """模拟传感器数据，便于在无机器狗环境下开发调试。"""
        import math

        def _mock_loop():
            t = 0.0
            while True:
                t += 0.1
                with self._lock:
                    self._imu = IMUData(
                        roll=math.sin(t) * 3,
                        pitch=math.cos(t * 0.5) * 2,
                    )
                    self._odom = OdomData(x=t * 0.1, y=0.0)
                    self._battery = BatteryState(percent=max(0, 100 - t * 0.05))
                time.sleep(0.1)

        threading.Thread(target=_mock_loop, daemon=True, name="mock_sensor").start()
        log.info("mock_sensor_started")

    def shutdown(self) -> None:
        if not self._mock:
            try:
                import rclpy
                self._node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass
        log.info("ros2_bridge_shutdown")
