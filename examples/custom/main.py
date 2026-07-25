"""CustomAgent 最小可运行示例。"""
from __future__ import annotations

from muye_multi_agent_sdk import AgentMetadata, AgentRequest, AgentResult, CustomAgent, create_app


class EchoAgent(CustomAgent):
    """确定性 Agent：适合外部 API 适配或无需 LLM 的流程。"""

    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="echo-agent", version="1.0.0", description="Custom 模式示例")

    async def execute(self, request: AgentRequest, **_kwargs: object) -> AgentResult:
        return AgentResult.success({"markdown": f"已处理：{request.task}"}, trace_id=request.context.trace_id)


app = create_app(EchoAgent())
