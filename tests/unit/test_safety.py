"""
tests/unit/test_safety.py
SafetyGuard 单元测试
"""
import pytest
from unittest.mock import MagicMock

from src.utils.safety import SafetyGuard, RobotState


@pytest.fixture
def e_stop_cb():
    return MagicMock()


@pytest.fixture
def guard(e_stop_cb):
    return SafetyGuard(
        max_linear_vel=0.8,
        max_angular_vel=1.5,
        battery_low=15.0,
        battery_critical=5.0,
        joint_temp_max=80.0,
        tilt_max_deg=45.0,
        emergency_stop_cb=e_stop_cb,
    )


class TestVelocityClipping:
    def test_within_limits_unchanged(self, guard):
        lx, ly, az = guard.clip_velocity(0.3, 0.0, 0.5)
        assert abs(lx - 0.3) < 1e-6
        assert abs(az - 0.5) < 1e-6

    def test_linear_clipped(self, guard):
        lx, ly, az = guard.clip_velocity(2.0, 0.0, 0.0)
        total = (lx**2 + ly**2) ** 0.5
        assert total <= 0.8 + 1e-6

    def test_angular_clipped(self, guard):
        _, _, az = guard.clip_velocity(0.0, 0.0, 5.0)
        assert az <= 1.5 + 1e-6

    def test_angular_negative_clipped(self, guard):
        _, _, az = guard.clip_velocity(0.0, 0.0, -5.0)
        assert az >= -1.5 - 1e-6


class TestActionAllowed:
    def test_normal_state_allowed(self, guard):
        assert guard.check_action_allowed() is True

    def test_emergency_stop_blocks(self, guard):
        guard.trigger_emergency_stop("test")
        assert guard.check_action_allowed() is False

    def test_reset_allows_again(self, guard):
        guard.trigger_emergency_stop("test")
        guard.reset_emergency_stop()
        assert guard.check_action_allowed() is True

    def test_critical_battery_blocks(self, guard):
        guard.update_state(battery_percent=3.0)
        assert guard.check_action_allowed() is False


class TestEmergencyStop:
    def test_callback_called(self, guard, e_stop_cb):
        guard.trigger_emergency_stop("test_reason")
        e_stop_cb.assert_called_once()

    def test_tilt_triggers_estop(self, guard, e_stop_cb):
        guard.update_state(roll_deg=60.0, pitch_deg=0.0)
        e_stop_cb.assert_called()

    def test_overheat_triggers_estop(self, guard, e_stop_cb):
        guard.update_state(joint_temps={"front_left": 90.0})
        e_stop_cb.assert_called()
