"""GraphAgent 最小可运行示例。"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from muye_multi_agent_sdk import AgentMetadata, AgentRequest, AgentResult, GraphAgent, create_app


class State(TypedDict, total=False):
    task: str
    normalized: str


class NormalizeAgent(GraphAgent):
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="normalize-agent", version="1.0.0", description="Graph 模式示例")

    def build_graph(self) -> Any:
        graph = StateGraph(State)
        graph.add_node("normalize", lambda state: {"normalized": state["task"].strip()})
        graph.add_edge(START, "normalize")
        graph.add_edge("normalize", END)
        return graph

    async def result_from_state(self, state: dict[str, Any], request: AgentRequest) -> AgentResult:
        return AgentResult.success({"markdown": state.get("normalized", "")}, trace_id=request.context.trace_id)


app = create_app(NormalizeAgent())
