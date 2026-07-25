"""LangGraph 模式。"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ..contracts import AgentEvent, AgentRequest, AgentResult
from ..runtime import ExecutionOptions
from .base import BaseAgent


class GraphAgent(BaseAgent):
    """托管 StateGraph 编译、上下文与节点进度映射的模式基类。"""

    @abstractmethod
    def build_graph(self) -> Any:
        """返回未编译的 LangGraph StateGraph。"""

    @abstractmethod
    async def result_from_state(self, state: dict[str, Any], request: AgentRequest) -> AgentResult:
        """将最终图状态转换为统一结果。"""

    def initial_state(self, request: AgentRequest) -> dict[str, Any]:
        """默认将任务及安全上下文快照传入图。"""
        return {"task": request.task, "context": request.context.model_dump()}

    def node_progress(self, node_name: str) -> tuple[str, int]:
        """返回节点的人类可读进度信息。"""
        return node_name, 0

    async def _compiled(self, options: ExecutionOptions) -> Any:
        graph = self.build_graph()
        checkpointer = await self.checkpointer(options)
        return graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()

    async def execute(self, request: AgentRequest, *, options: ExecutionOptions) -> AgentResult:
        graph = await self._compiled(options)
        state = await graph.ainvoke(self.initial_state(request), config=self._runtime_config(request, options))
        return await self.result_from_state(dict(state), request)

    async def stream_events(self, request: AgentRequest, *, options: ExecutionOptions) -> AsyncIterator[AgentEvent]:
        graph = await self._compiled(options)
        state: dict[str, Any] = {}
        async for update in graph.astream(
            self.initial_state(request),
            config=self._runtime_config(request, options),
            stream_mode="updates",
        ):
            if not isinstance(update, dict):
                continue
            for node_name, node_state in update.items():
                if isinstance(node_state, dict):
                    state.update(node_state)
                description, progress = self.node_progress(str(node_name))
                yield AgentEvent.tool(str(node_name), "running", log=description, progress=progress)
        yield AgentEvent.completed(await self.result_from_state(state, request))
