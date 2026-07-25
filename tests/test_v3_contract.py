"""独立 SDK 的公共契约回归测试。"""
from __future__ import annotations

import asyncio

import httpx

from muye_multi_agent_sdk import (
    AgentConfig,
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


def test_sdk_version_is_1_0_0() -> None:
    """包公共版本与构建使用的版本常量必须一致。"""
    assert SDK_VERSION == __version__ == "1.0.0"


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


async def _post(app: object, path: str, payload: dict[str, object]) -> httpx.Response:
    """通过 ASGI transport 调用路由，避免同步 TestClient 的版本耦合。"""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        return await client.post(path, json=payload)


async def _get(
    app: object,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """通过 ASGI transport 读取无副作用端点。"""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        return await client.get(path, headers=headers)
