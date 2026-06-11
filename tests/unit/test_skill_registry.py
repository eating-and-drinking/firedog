"""
tests/unit/test_skill_registry.py
SkillRegistry 单元测试
"""
import pytest
from unittest.mock import MagicMock, patch

from src.skills.motion_skills import SkillResult
from src.skills.skill_registry import SkillRegistry


@pytest.fixture
def mock_motion():
    m = MagicMock()
    m.stand_up.return_value = SkillResult(True, "已站立")
    m.stop.return_value = SkillResult(True, "已停止")
    m.move_forward.return_value = SkillResult(True, "前进完成")
    m.perform_greeting.return_value = SkillResult(True, "打招呼完成")
    return m


@pytest.fixture
def mock_sensor():
    s = MagicMock()
    s.get_battery_level.return_value = SkillResult(True, "电量 85.0%", data={"percent": 85.0})
    s.get_position.return_value = SkillResult(True, "X=1.0 Y=0.5", data={"x": 1.0, "y": 0.5})
    s.get_full_status.return_value = SkillResult(True, "综合状态正常")
    return s


@pytest.fixture
def registry(mock_motion, mock_sensor):
    return SkillRegistry(mock_motion, mock_sensor)


class TestSkillRegistry:
    def test_stand_up_success(self, registry, mock_motion):
        result = registry.execute("stand_up")
        assert result.success is True
        mock_motion.stand_up.assert_called_once()

    def test_unknown_skill_returns_error(self, registry):
        result = registry.execute("fly_to_moon")
        assert result.success is False
        assert "不存在" in result.message

    def test_move_forward_with_params(self, registry, mock_motion):
        result = registry.execute("move_forward", speed=0.5, duration_s=3.0)
        assert result.success is True
        mock_motion.move_forward.assert_called_once_with(speed=0.5, duration_s=3.0)

    def test_get_battery(self, registry, mock_sensor):
        result = registry.execute("get_battery")
        assert result.success is True
        assert result.data["percent"] == 85.0

    def test_list_skills_returns_dict(self, registry):
        skills = registry.list_skills()
        assert isinstance(skills, dict)
        assert "stand_up" in skills
        assert "move_forward" in skills

    def test_bad_params_returns_error(self, registry, mock_motion):
        mock_motion.move_forward.side_effect = TypeError("unexpected argument")
        result = registry.execute("move_forward", bad_param=999)
        assert result.success is False
