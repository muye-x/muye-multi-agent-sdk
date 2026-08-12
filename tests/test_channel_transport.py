"""Channel 协议与可选 Agent endpoint 回归测试。"""
from __future__ import annotations

import asyncio

import httpx

from muye_multi_agent_sdk import (
    AgentConfig,
    AgentMetadata,
    AgentRequest,
    AgentResult,
    ChannelInvokeRequest,
    CustomAgent,
    create_app,
)


class ChannelTestAgent(CustomAgent):
    """记录通道映射结果的最小 Agent。"""

    def __init__(self) -> None:
        super().__init__(AgentConfig())
        self.request: AgentRequest | None = None

    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="channel-test", version="1.0.0", description="Channel transport test")

    async def execute(self, request: AgentRequest, **_kwargs: object) -> AgentResult:
        self.request = request
        return AgentResult.success({"markdown": "已收到"}, trace_id=request.context.trace_id)


async def _post(app: object, payload: dict[str, object], *, token: str = "token") -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/internal/v1/channels/invoke", json=payload, headers={"Authorization": f"Bearer {token}"})


def _payload() -> dict[str, object]:
    return ChannelInvokeRequest(
        channel="wechat",
        user_id="usr_1",
        session_id="wechat_session_1",
        trace_id="trace-0001",
        message_id="message-0001",
        message={"type": "text", "content": "  你好  "},
    ).model_dump(mode="json")


def test_channel_endpoint_is_opt_in_and_hides_provider_data() -> None:
    agent = ChannelTestAgent()
    app = create_app(agent, channel_request_verifier=lambda request: request.headers.get("authorization") == "Bearer token")

    response = asyncio.run(_post(app, _payload()))

    assert response.status_code == 200
    assert response.json() == {"status": "success", "trace_id": "trace-0001", "message": {"type": "text", "content": "已收到"}, "error": None}
    assert agent.request is not None
    assert agent.request.context.extra == {"channel": "wechat", "channel_message_id": "message-0001"}


def test_channel_endpoint_rejects_invalid_service_token_before_execution() -> None:
    agent = ChannelTestAgent()
    app = create_app(agent, channel_request_verifier=lambda _: False)

    response = asyncio.run(_post(app, _payload()))

    assert response.status_code == 401
    assert agent.request is None
