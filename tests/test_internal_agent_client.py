"""SDK internal Agent client 的协议契约测试。"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from muye_multi_agent_sdk import AgentRequest
from muye_multi_agent_sdk.integrations import InternalAgentClient, InternalAgentClientError


def _client_factory(handler):
    return lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


def test_stream_negotiates_and_filters_child_envelope() -> None:
    paths: list[str] = []
    stream_body = "".join(
        [
            'event: session_start\ndata: {"event":"session_start"}\n\n',
            'event: thinking\ndata: {"event":"thinking","data":{"content":"计划"}}\n\n',
            'event: done\ndata: {"event":"done"}\n\n',
            'event: session_end\ndata: {"event":"session_end"}\n\n',
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"internal_protocol_version": "muye-agent-internal/3.0", "api_profiles": ["internal"], "supports_streaming": True})
        return httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})

    async def run() -> list[dict]:
        client = InternalAgentClient(_client_factory(handler))
        return [
            event
            async for event in client.stream(
                base_url="http://travel.test",
                timeout_seconds=2,
                request=AgentRequest(task="成都三日游", context={"user_id": "u1", "session_id": "s1"}),
                service_name="travel",
            )
        ]

    assert asyncio.run(run()) == [{"event": "thinking", "data": {"content": "计划"}}]
    assert paths == ["/capabilities", "/invoke/stream"]


def test_invoke_and_cancel_validate_standard_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"internal_protocol_version": "muye-agent-internal/3.0", "api_profiles": ["internal"], "supports_streaming": True})
        if request.url.path == "/invoke":
            return httpx.Response(200, json={"status": "success", "payload": {"result_data": {"markdown": "ok"}}})
        return httpx.Response(200, json={"status": "cancelled", "message": "已取消"})

    async def run() -> tuple[dict, dict]:
        client = InternalAgentClient(_client_factory(handler))
        result = await client.invoke(base_url="http://travel.test", timeout_seconds=2, request=AgentRequest(task="测试"))
        cancelled = await client.cancel(base_url="http://travel.test", timeout_seconds=2, user_id="u1", session_id="s1")
        return result, cancelled

    assert asyncio.run(run()) == (
        {"status": "success", "payload": {"result_data": {"markdown": "ok"}}},
        {"status": "cancelled", "message": "已取消"},
    )


def test_stream_rejects_invalid_capabilities() -> None:
    with pytest.raises(InternalAgentClientError, match="internal v3"):
        InternalAgentClient.validate_capabilities({"api_profiles": ["internal"]}, require_streaming=True)
