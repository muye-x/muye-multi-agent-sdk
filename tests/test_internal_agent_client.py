"""SDK internal Agent client 的协议契约测试。"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from pydantic import SecretStr

from muye_multi_agent_sdk import AgentIdentity, AgentRequest
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
    with pytest.raises(InternalAgentClientError, match="协议版本"):
        InternalAgentClient.validate_capabilities({"api_profiles": ["internal"]}, require_streaming=True)


def test_invoke_requires_expected_identity_and_rejects_expired_deadline() -> None:
    expected_identity = AgentIdentity(
        agent_id="agent_product_handbook",
        agent_version="1.0.0",
        descriptor_checksum="a" * 64,
        source_tree_checksum="b" * 64,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/capabilities":
            return httpx.Response(
                200,
                json={
                    "internal_protocol_version": "muye-agent-internal/3.0",
                    "api_profiles": ["internal"],
                    "supports_streaming": True,
                    "identity": expected_identity.model_dump(mode="json"),
                },
            )
        return httpx.Response(200, json={"status": "success", "payload": {"result_data": {"markdown": "ok"}}})

    client = InternalAgentClient(_client_factory(handler))
    result = asyncio.run(
        client.invoke(
            base_url="http://knowledge.test",
            timeout_seconds=2,
            request=AgentRequest(task="退款"),
            expected_identity=expected_identity,
        )
    )
    assert result["status"] == "success"

    with pytest.raises(InternalAgentClientError, match="deadline"):
        asyncio.run(
            client.invoke(
                base_url="http://knowledge.test",
                timeout_seconds=2,
                request=AgentRequest(task="退款"),
                deadline_monotonic=time.monotonic() - 1,
            )
        )


def test_client_forwards_service_token_and_cross_process_deadline() -> None:
    received_headers: dict[str, dict[str, str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received_headers[request.url.path] = dict(request.headers)
        if request.url.path == "/capabilities":
            return httpx.Response(
                200,
                json={
                    "internal_protocol_version": "muye-agent-internal/3.0",
                    "api_profiles": ["internal"],
                    "supports_streaming": True,
                    "features": ["trusted_deadline"],
                },
            )
        return httpx.Response(200, json={"status": "success", "payload": {"result_data": {"markdown": "ok"}}})

    result = asyncio.run(
        InternalAgentClient(_client_factory(handler)).invoke(
            base_url="http://knowledge.test",
            timeout_seconds=2,
            request=AgentRequest(task="退款"),
            deadline_monotonic=time.monotonic() + 5,
            service_token=SecretStr("test-service-token"),
        )
    )

    assert result["status"] == "success"
    for headers in received_headers.values():
        assert headers["authorization"] == "Bearer test-service-token"
        assert int(headers["x-muye-deadline-unix-ms"]) > int(time.time() * 1000)


def test_capabilities_reject_unexpected_protocol_profile_or_deadline_feature() -> None:
    valid = {
        "internal_protocol_version": "muye-agent-internal/3.0",
        "api_profiles": ["internal"],
        "supports_streaming": True,
        "features": ["trusted_deadline"],
    }

    with pytest.raises(InternalAgentClientError, match="协议版本"):
        InternalAgentClient.validate_capabilities(
            valid,
            require_streaming=False,
            expected_protocol_version="muye-agent-internal/3.1",
        )
    with pytest.raises(InternalAgentClientError, match="预期 API profile"):
        InternalAgentClient.validate_capabilities(valid, require_streaming=False, expected_profile="public")
    with pytest.raises(InternalAgentClientError, match="所需能力"):
        InternalAgentClient.validate_capabilities(
            {**valid, "features": []},
            require_streaming=False,
            required_features={"trusted_deadline"},
        )
