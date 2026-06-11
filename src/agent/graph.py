"""
src/agent/graph.py
LangGraph Agent 工作流（考核项三）

架构：ReAct 风格
  用户输入 → LLM 推理 → 工具调用 → 结果反馈 → 循环直到完成

分层解耦：
  - Agent 层：自然语言理解、多步规划、Tool 调用
  - 技能层：运动/传感器指令（不感知 LLM）
  - 控制层：ROS 2 / SDK（不感知 Agent 逻辑）
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.utils.logger import get_logger
from src.utils.metrics import AGENT_TASK_DURATION, AGENT_TOOL_CALLS

log = get_logger(__name__)

SYSTEM_PROMPT = """你是一只四足机器狗的智能助手。你可以：
- 控制机器狗的运动（前进、后退、转向、导航到指定位置）
- 查询机器狗状态（电量、位置、姿态）
- 执行复合任务（巡逻、跟随、巡检后返回）

安全规则（必须遵守）：
1. 速度不超过 0.8 m/s，角速度不超过 1.5 rad/s
2. 电量低于 15% 时，警告用户并建议充电，不执行高耗能任务
3. 任何时候用户说"停止"/"急停"/"别动"，立即调用 stop 工具
4. 不确定指令意图时，先询问确认再执行

对话风格：
- 简洁、准确、拟人化（可用"我"来指代机器狗）
- 执行动作前简短确认，执行后报告结果
- 遇到错误主动解释并提供替代方案
"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task_description: str
    iteration_count: int
    is_done: bool


class RobotDogAgent:
    """
    LangGraph ReAct Agent。
    通过 handle(user_text) 接受用户指令，返回机器狗回复文本。
    """

    def __init__(
        self,
        tools: list[BaseTool],
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str = "",
        llm_base_url: str = "",
        max_iterations: int = 10,
        memory_window: int = 20,
    ):
        self._tools = tools
        self._max_iter = max_iterations
        self._memory_window = memory_window
        self._history: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        llm_kwargs: dict[str, Any] = {
            "model": llm_model,
            "temperature": 0.2,
            "max_tokens": 512,
            "timeout": 10,
        }
        if llm_api_key:
            llm_kwargs["api_key"] = llm_api_key
        if llm_base_url:
            llm_kwargs["base_url"] = llm_base_url

        self._llm = ChatOpenAI(**llm_kwargs).bind_tools(tools)
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def handle(self, user_text: str) -> str:
        """
        处理用户语音转写文本，返回机器狗回复文本。
        内部执行多轮 LLM + Tool 调用直到完成。
        """
        import time
        start = time.monotonic()

        log.info("agent_handle", user_text=user_text[:100])

        # 加入用户消息
        self._history.append(HumanMessage(content=user_text))

        # 维护对话窗口（保留 system prompt + 最近 N 条）
        windowed = [self._history[0]] + self._history[-(self._memory_window):]

        state: AgentState = {
            "messages": windowed,
            "task_description": user_text,
            "iteration_count": 0,
            "is_done": False,
        }

        final_state = self._graph.invoke(state)
        messages = final_state["messages"]

        # 找到最后一条 AI 文本消息作为回复
        reply = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
                reply = msg.content.strip()
                break

        if not reply:
            reply = "好的，已执行完成。"

        # 追加 AI 回复到历史
        self._history.append(AIMessage(content=reply))

        elapsed = time.monotonic() - start
        AGENT_TASK_DURATION.observe(elapsed)
        log.info("agent_done", elapsed_s=round(elapsed, 2), reply=reply[:80])

        return reply

    def reset_memory(self) -> None:
        """清除对话历史（保留 system prompt）。"""
        self._history = [self._history[0]]
        log.info("agent_memory_reset")

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        tool_node = ToolNode(self._tools)

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    # ------------------------------------------------------------------
    # 节点函数
    # ------------------------------------------------------------------

    def _agent_node(self, state: AgentState) -> dict:
        iter_count = state["iteration_count"] + 1

        if iter_count > self._max_iter:
            log.warning("agent_max_iterations_reached", max=self._max_iter)
            return {
                "messages": [AIMessage(content="抱歉，任务执行步骤过多，已安全停止。")],
                "iteration_count": iter_count,
                "is_done": True,
            }

        messages = list(state["messages"])
        log.debug("agent_node", iteration=iter_count, num_messages=len(messages))

        response = self._llm.invoke(messages)
        return {
            "messages": [response],
            "iteration_count": iter_count,
            "is_done": False,
        }

    @staticmethod
    def _should_continue(state: AgentState) -> str:
        if state.get("is_done"):
            return "end"
        messages = state["messages"]
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "continue"
        return "end"
