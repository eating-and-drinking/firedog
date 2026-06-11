"""
src/agent/tools.py
LangChain Tool 定义：将 SkillRegistry 中的技能包装为 Agent 可调用的 Tools
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.skills.skill_registry import SkillRegistry


# ------------------------------------------------------------------
# Pydantic 参数模型
# ------------------------------------------------------------------

class MoveForwardInput(BaseModel):
    speed: float = Field(default=0.3, ge=0.01, le=0.8, description="速度 m/s")
    duration_s: float = Field(default=2.0, ge=0.1, le=30.0, description="持续秒数")

class MoveBackwardInput(BaseModel):
    speed: float = Field(default=0.2, ge=0.01, le=0.8, description="速度 m/s")
    duration_s: float = Field(default=2.0, ge=0.1, le=30.0, description="持续秒数")

class TurnInput(BaseModel):
    angular_speed: float = Field(default=0.5, ge=0.1, le=1.5, description="角速度 rad/s")
    duration_s: float = Field(default=2.0, ge=0.1, le=30.0, description="持续秒数")

class MoveToPositionInput(BaseModel):
    target_x: float = Field(description="目标 X 坐标（m）")
    target_y: float = Field(description="目标 Y 坐标（m）")
    speed: float = Field(default=0.3, ge=0.01, le=0.8, description="速度 m/s")

class PatrolInput(BaseModel):
    waypoints: list[list[float]] = Field(
        description="路点列表，每个路点 [x, y]，例如 [[1.0,0.0],[2.0,1.0]]"
    )
    speed: float = Field(default=0.3, ge=0.01, le=0.8, description="速度 m/s")

class EmptyInput(BaseModel):
    pass


# ------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------

def build_tools(registry: "SkillRegistry") -> list[StructuredTool]:
    """根据 SkillRegistry 生成 LangChain StructuredTool 列表。"""

    def _wrap(name: str, description: str, args_schema, **static_kwargs):
        """闭包：把 registry.execute 包装为 Tool 函数。"""
        def _fn(**kwargs):
            kw = {**static_kwargs, **kwargs}
            result = registry.execute(name, **kw)
            return result.message

        _fn.__name__ = name
        return StructuredTool.from_function(
            func=_fn,
            name=name,
            description=description,
            args_schema=args_schema,
        )

    tools = [
        _wrap("stand_up",       "让机器狗站立",          EmptyInput),
        _wrap("lie_down",       "让机器狗趴下/休息",      EmptyInput),
        _wrap("stop",           "立即停止所有运动",        EmptyInput),
        _wrap("greeting",       "执行打招呼动作",          EmptyInput),
        _wrap("get_battery",    "查询当前电量百分比",      EmptyInput),
        _wrap("get_position",   "查询机器狗当前坐标",      EmptyInput),
        _wrap("get_imu",        "查询机器狗姿态（roll/pitch）", EmptyInput),
        _wrap("get_status",     "查询综合状态（位置+姿态+电量）", EmptyInput),
        StructuredTool.from_function(
            func=lambda speed, duration_s: registry.execute("move_forward", speed=speed, duration_s=duration_s).message,
            name="move_forward",
            description="控制机器狗向前行走，指定速度和时长",
            args_schema=MoveForwardInput,
        ),
        StructuredTool.from_function(
            func=lambda speed, duration_s: registry.execute("move_backward", speed=speed, duration_s=duration_s).message,
            name="move_backward",
            description="控制机器狗向后退",
            args_schema=MoveBackwardInput,
        ),
        StructuredTool.from_function(
            func=lambda angular_speed, duration_s: registry.execute("turn_left", angular_speed=angular_speed, duration_s=duration_s).message,
            name="turn_left",
            description="控制机器狗左转",
            args_schema=TurnInput,
        ),
        StructuredTool.from_function(
            func=lambda angular_speed, duration_s: registry.execute("turn_right", angular_speed=angular_speed, duration_s=duration_s).message,
            name="turn_right",
            description="控制机器狗右转",
            args_schema=TurnInput,
        ),
        StructuredTool.from_function(
            func=lambda target_x, target_y, speed: registry.execute(
                "move_to_position", target_x=target_x, target_y=target_y, speed=speed
            ).message,
            name="move_to_position",
            description="导航到指定坐标点",
            args_schema=MoveToPositionInput,
        ),
        StructuredTool.from_function(
            func=lambda waypoints, speed: registry.execute(
                "patrol", waypoints=[tuple(wp) for wp in waypoints], speed=speed
            ).message,
            name="patrol",
            description="按顺序巡逻多个路点，适合巡检任务",
            args_schema=PatrolInput,
        ),
    ]
    return tools
