"""基于 LangChain create_agent 的 ReAct 模式。"""
from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from inspect import isawaitable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from ..config import AgentConfig
from ..contracts import AgentEvent, AgentRequest, AgentResult, ToolCapability
from ..integrations.factory import build_chat_model
from ..runtime import ExecutionOptions
from ..safety import GuardContext, GuardHistoryMessage
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """使用注入 ChatModel 或内置模型工厂执行工具驱动任务。"""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        model: BaseChatModel | None = None,
        guard_model: BaseChatModel | None = None,
    ) -> None:
        super().__init__(config)
        self._model = model
        self._guard_model = guard_model
        self._model_owned_by_sdk = False
        self._agents: dict[bool, Any] = {}

    @property
    @abstractmethod
    def instructions(self) -> str:
        """传给 ReAct runtime 的系统指令。"""

    @property
    @abstractmethod
    def langchain_tools(self) -> list[BaseTool]:
        """返回 LangChain 工具对象。"""

    @property
    def tools(self) -> list[ToolCapability]:
        return [
            ToolCapability(
                name=str(getattr(tool, "name", type(tool).__name__)),
                description=str(getattr(tool, "description", "")),
            )
            for tool in self.langchain_tools
        ]

    async def guard_model(self) -> BaseChatModel | None:
        if self._guard_model is not None:
            return self._guard_model
        return self._get_model()

    def _get_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = build_chat_model(self.config.model)
            self._model_owned_by_sdk = True
        return self._model

    async def _agent(self, options: ExecutionOptions) -> Any:
        stateful = options.context_enabled and options.profile in self.config.context.enabled_profiles
        if stateful in self._agents:
            return self._agents[stateful]
        try:
            from langchain.agents import create_agent
        except ImportError as exc:
            raise ImportError(
                "ReAct 模式的核心依赖 langchain 不可用，请重新安装 muye-multi-agent-sdk"
            ) from exc
        kwargs: dict[str, Any] = {
            "model": self._get_model(),
            "tools": self.langchain_tools,
            "system_prompt": self.instructions,
        }
        checkpointer = await self.checkpointer(options)
        if checkpointer is not None:
            kwargs["checkpointer"] = checkpointer
        self._agents[stateful] = create_agent(**kwargs)
        return self._agents[stateful]

    async def execute(self, request: AgentRequest, *, options: ExecutionOptions) -> AgentResult:
        agent = await self._agent(options)
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=request.task)]},
            config=self._runtime_config(request, options),
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        return self._result_from_messages(messages, request)

    async def _intent_guard_context(self, request: AgentRequest, options: ExecutionOptions) -> GuardContext:
        """从同一 ReAct checkpoint 提取守卫所需的有限人机文本历史。"""
        try:
            agent = await self._agent(options)
            async with asyncio.timeout(self.config.intent_guard.history_io_timeout_seconds):
                snapshot = await agent.aget_state(self._runtime_config(request, options))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Intent guard history read failed agent=%s profile=%s trace_id=%s error_type=%s",
                self.metadata.name,
                options.profile,
                request.context.trace_id,
                type(exc).__name__,
            )
            return GuardContext(read_failed=True)
        values = getattr(snapshot, "values", None)
        messages = values.get("messages") if isinstance(values, dict) else None
        guard_context = self._guard_history(messages)
        logger.debug(
            "Intent guard history loaded agent=%s profile=%s trace_id=%s message_count=%s truncated=%s",
            self.metadata.name,
            options.profile,
            request.context.trace_id,
            len(guard_context.history),
            guard_context.truncated,
        )
        return guard_context

    async def _record_guard_clarification(
        self,
        request: AgentRequest,
        options: ExecutionOptions,
        question: str,
    ) -> None:
        """将守卫提前返回的澄清消息写入 ReAct 历史，供下一轮补充使用。"""
        try:
            from langgraph.graph import START

            agent = await self._agent(options)
            async with asyncio.timeout(self.config.intent_guard.history_io_timeout_seconds):
                await agent.aupdate_state(
                    self._runtime_config(request, options),
                    {"messages": [HumanMessage(content=request.task), AIMessage(content=question)]},
                    as_node=START,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Intent guard clarification write failed agent=%s profile=%s trace_id=%s error_type=%s",
                self.metadata.name,
                options.profile,
                request.context.trace_id,
                type(exc).__name__,
            )

    def _guard_history(self, messages: object) -> GuardContext:
        """过滤 LangGraph 状态，避免工具参数、系统提示和大段历史进入守卫提示词。"""
        if not isinstance(messages, list):
            return GuardContext()
        candidates: list[GuardHistoryMessage] = []
        for message in messages:
            role: str | None = None
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            if role is None:
                continue
            content = self._message_text(message.content)
            if content:
                candidates.append(GuardHistoryMessage(role=role, content=content))

        max_messages = self.config.intent_guard.history_max_messages
        max_chars = self.config.intent_guard.history_max_chars
        selected: list[GuardHistoryMessage] = []
        remaining = max_chars
        truncated = len(candidates) > max_messages
        for message in reversed(candidates[-max_messages:]):
            if remaining <= 0:
                truncated = True
                break
            content = message.content
            if len(content) > remaining:
                content = content[-remaining:]
                truncated = True
            selected.append(GuardHistoryMessage(role=message.role, content=content))
            remaining -= len(content)
        selected.reverse()
        return GuardContext(history=tuple(selected), truncated=truncated)

    @staticmethod
    def _message_text(content: object) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        return "\n".join(
            block["text"].strip()
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip()
        )

    def _result_from_messages(self, messages: list[Any], request: AgentRequest) -> AgentResult:
        """从 LangChain 消息列表构造 SDK 统一结果。"""
        final_message = next((item for item in reversed(messages) if isinstance(item, AIMessage)), None)
        content = final_message.content if final_message is not None else ""
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
        text = str(content).strip()
        if not text:
            return AgentResult.failure("EMPTY_RESULT", "ReAct Agent 未返回可展示结果。", recoverable=True, trace_id=request.context.trace_id)
        tool_names = [str(getattr(message, "name", "")) for message in messages if getattr(message, "type", "") == "tool"]
        return AgentResult.success({"markdown": text}, tool_calls_made=[name for name in tool_names if name], trace_id=request.context.trace_id)

    async def stream_events(self, request: AgentRequest, *, options: ExecutionOptions) -> AsyncIterator[AgentEvent]:
        """先投影工具生命周期，再以统一终态结果收尾。"""
        agent = await self._agent(options)
        final_messages: list[Any] = []
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=request.task)]},
            config=self._runtime_config(request, options),
            version="v2",
        ):
            event_name = event.get("event")
            if event_name == "on_tool_start":
                yield AgentEvent.tool(str(event.get("name", "tool")), "start", input=event.get("data", {}).get("input", {}))
            elif event_name == "on_tool_end":
                yield AgentEvent.tool(str(event.get("name", "tool")), "complete")
            elif event_name == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict) and isinstance(output.get("messages"), list):
                    final_messages = output["messages"]
        yield AgentEvent.completed(self._result_from_messages(final_messages, request))

    async def aclose(self) -> None:
        """清除运行时，并仅关闭 SDK 工厂创建的模型。

        ``model`` 和 ``guard_model`` 都允许调用方注入并跨 Agent 共享，因此 SDK 不接管
        它们的关闭责任。只有 ``_get_model()`` 通过内置工厂创建的主模型会在应用关闭时
        被关闭。
        """
        self._agents.clear()
        if self._model_owned_by_sdk and self._model is not None:
            close = getattr(self._model, "aclose", None)
            if close is not None:
                result = close()
                if isawaitable(result):
                    await result
        self._model = None
        self._model_owned_by_sdk = False
        await super().aclose()
