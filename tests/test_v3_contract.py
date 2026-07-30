"""独立 SDK 的公共契约回归测试。"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from muye_multi_agent_sdk import (
    AgentConfig,
    AgentCapabilities,
    AgentIdentity,
    ContextConfig,
    AgentMetadata,
    AgentRequest,
    AgentResult,
    ApiConfig,
    CustomAgent,
    SDK_VERSION,
    __version__,
    create_app,
)


class EchoAgent(CustomAgent):
    """不依赖网络的 Custom 模式测试 Agent。"""

    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="echo-agent", version="1.0.0", description="测试 Agent")

    async def execute(self, request: AgentRequest, **_kwargs: object) -> AgentResult:
        return AgentResult.success(
            {"markdown": f"已处理：{request.task}"},
            origin_data={"private": True},
            prompt_data="直接展示结果。",
            trace_id=request.context.trace_id,
        )


class FailingAgent(EchoAgent):
    """用于验证 public profile 不透传领域异常。"""

    async def execute(self, request: AgentRequest, **_kwargs: object) -> AgentResult:
        return AgentResult.failure("UPSTREAM_FAILURE", "内部连接串错误", trace_id=request.context.trace_id)


class IdentifiedAgent(EchoAgent):
    """携带 v2 部署身份的最小 Agent。"""

    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="identified-agent",
            version="1.0.0",
            description="身份测试",
            identity=AgentIdentity(
                agent_id="agent_product_handbook",
                agent_version="1.0.0",
                descriptor_checksum="a" * 64,
                source_tree_checksum="b" * 64,
            ),
        )


def test_sdk_version_is_2_0_0() -> None:
    """包公共版本与构建使用的版本常量必须一致。"""
    assert SDK_VERSION == __version__ == "2.0.0"


def test_custom_agent_is_directly_invokable_without_runtime_context() -> None:
    """直接调用不要求调用方手工设置 ContextVar。"""
    result = asyncio.run(EchoAgent().invoke(AgentRequest(task="测试")))

    assert result.status == "success"
    assert result.result_data == {"markdown": "已处理：测试"}


def test_internal_http_contract_keeps_three_field_payload() -> None:
    """internal HTTP 契约保持主 Agent 所需的三字段 payload。"""
    app = create_app(EchoAgent())
    response = asyncio.run(_post(app, "/invoke", {"task": "测试", "context": {"user_id": "u", "session_id": "s"}}))

    assert response.status_code == 200
    assert response.json()["payload"] == {
        "origin_data": {"private": True},
        "result_data": {"markdown": "已处理：测试"},
        "prompt_data": "直接展示结果。",
    }


def test_stream_has_single_sse_owner_and_terminal_envelope() -> None:
    """transport 必须独占 SSE 信封并保证 done 与 session_end。"""
    app = create_app(EchoAgent())
    response = asyncio.run(_post(app, "/invoke/stream", {"task": "测试", "context": {"user_id": "u", "session_id": "s"}}))

    assert response.status_code == 200
    assert response.text.count("event: session_start") == 1
    assert response.text.count("event: done") == 1
    assert response.text.count("event: session_end") == 1
    assert "origin_data" in response.text


def test_public_profile_projects_markdown_without_internal_data() -> None:
    """public profile 不得泄漏 internal origin_data 或 prompt_data。"""
    app = create_app(EchoAgent(AgentConfig(api=ApiConfig(profiles={"internal", "public"}, public_path="/api/v1/echo"))))
    response = asyncio.run(_post(app, "/api/v1/echo/invoke", {"user_input": "测试", "user_id": "u", "session_id": "s"}))

    assert response.status_code == 200
    assert response.json()["markdown"] == "已处理：测试"
    assert "origin_data" not in response.text
    assert "prompt_data" not in response.text


def test_capabilities_declares_one_consistent_protocol_version() -> None:
    """运行时能力声明必须使用唯一版本常量。"""
    response = asyncio.run(_get(create_app(EchoAgent()), "/capabilities"))

    assert response.status_code == 200
    assert response.json()["internal_protocol_version"] == "muye-agent-internal/3.0"


def test_ready_endpoint_declares_initialized_sdk_transport() -> None:
    response = asyncio.run(_get(create_app(EchoAgent()), "/ready"))

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["agent"] == "echo-agent"


def test_capabilities_expose_optional_v2_identity_and_features() -> None:
    response = asyncio.run(_get(create_app(IdentifiedAgent()), "/capabilities"))

    assert response.json()["identity"]["agent_id"] == "agent_product_handbook"
    assert set(response.json()["features"]) == {"cancel", "citation_blocks", "sse", "trusted_deadline"}


def test_identity_version_must_match_metadata_and_capabilities() -> None:
    identity = AgentIdentity(
        agent_id="agent_product_handbook",
        agent_version="1.0.0",
        descriptor_checksum="a" * 64,
        source_tree_checksum="b" * 64,
    )

    with pytest.raises(ValueError, match="identity.agent_version"):
        AgentMetadata(name="test", version="2.0.0", description="test", identity=identity)
    with pytest.raises(ValueError, match="identity.agent_version"):
        AgentCapabilities(
            agent_name="test",
            version="2.0.0",
            description="test",
            internal_protocol_version="muye-agent-internal/3.0",
            identity=identity,
        )


def test_internal_verifier_protects_capabilities_and_trusted_deadline() -> None:
    def verifier(request: object) -> bool:
        return getattr(request, "headers").get("authorization") == "Bearer test-service-token"

    app = create_app(EchoAgent(), internal_request_verifier=verifier)
    unauthorized = asyncio.run(_get(app, "/capabilities"))
    assert unauthorized.status_code == 401

    headers = {
        "Authorization": "Bearer test-service-token",
        "X-Muye-Deadline-Unix-Ms": str(int((time.time() - 1) * 1000)),
    }
    response = asyncio.run(_post(app, "/invoke", {"task": "测试"}, headers=headers))

    assert response.status_code == 200
    assert response.json()["error"]["code"] == "DEADLINE_EXCEEDED"


def test_cors_is_disabled_by_default() -> None:
    """未配置来源时不应向浏览器声明跨域访问权限。"""
    response = asyncio.run(
        _get(
            create_app(EchoAgent()),
            "/health",
            headers={"Origin": "https://app.example.com"},
        )
    )

    assert "access-control-allow-origin" not in response.headers


def test_empty_cors_origins_does_not_enable_wildcard_access() -> None:
    """显式空列表表示禁用 CORS，不能回退为通配符。"""
    response = asyncio.run(
        _get(
            create_app(EchoAgent(), cors_origins=[]),
            "/health",
            headers={"Origin": "https://app.example.com"},
        )
    )

    assert "access-control-allow-origin" not in response.headers


def test_explicit_cors_origin_is_allowed() -> None:
    """只有调用方显式声明的来源才应获得跨域响应头。"""
    origin = "https://app.example.com"
    response = asyncio.run(
        _get(
            create_app(EchoAgent(), cors_origins=[origin]),
            "/health",
            headers={"Origin": origin},
        )
    )

    assert response.headers["access-control-allow-origin"] == origin


def test_public_profile_redacts_agent_error_messages() -> None:
    """public HTTP 响应不能包含业务 Agent 的内部错误消息。"""
    app = create_app(FailingAgent(AgentConfig(api=ApiConfig(profiles={"public"}, public_path="/api/v1/failing"))))
    response = asyncio.run(_post(app, "/api/v1/failing/invoke", {"user_input": "测试", "user_id": "u", "session_id": "s"}))

    assert response.status_code == 200
    assert "内部连接串错误" not in response.text
    assert response.json()["markdown"] == "当前请求暂时无法完成，请稍后重试。"


def test_context_enabled_internal_request_requires_explicit_identity() -> None:
    """短期上下文不得让缺少身份的 internal 请求共享默认 checkpoint。"""
    app = create_app(EchoAgent(AgentConfig(context=ContextConfig(enabled_profiles={"internal"}))))
    response = asyncio.run(_post(app, "/invoke", {"task": "测试"}))

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["error"]["code"] == "CONTEXT_IDENTITY_REQUIRED"


async def _post(
    app: object,
    path: str,
    payload: dict[str, object],
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """通过 ASGI transport 调用路由，避免同步 TestClient 的版本耦合。"""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        return await client.post(path, json=payload, headers=headers)


async def _get(
    app: object,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """通过 ASGI transport 读取无副作用端点。"""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        return await client.get(path, headers=headers)
