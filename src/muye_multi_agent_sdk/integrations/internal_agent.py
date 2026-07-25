"""可信 internal v3 Agent 服务的异步 HTTP 客户端。"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from ..contracts import AgentRequest


class InternalAgentClientError(RuntimeError):
    """internal Agent 的网络、协议或能力协商失败。"""


class InternalAgentClient:
    """调用已由调用方信任和 allowlist 的 internal v3 Agent 服务。

    此类不接受模型生成的 URL，也不承担业务路由决策。调用方必须提供已经过
    自身配置校验的 ``base_url``。流式调用仅返回可嵌入父会话的中间事件，子服务
    自己的 session envelope 和终态事件会被过滤。
    """

    def __init__(self, client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient) -> None:
        self._client_factory = client_factory

    async def invoke(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        request: AgentRequest,
        service_name: str = "internal Agent",
    ) -> dict[str, Any]:
        """协商能力后调用非流式 internal 接口并返回原始标准响应。"""
        try:
            async with self._client_factory(timeout=timeout_seconds) as client:
                await self._negotiate(client, base_url, require_streaming=False)
                response = await client.post(self._url(base_url, "/invoke"), json=request.model_dump())
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise InternalAgentClientError(f"{service_name} 调用超时") from exc
        except httpx.HTTPError as exc:
            raise InternalAgentClientError(f"{service_name} 不可用") from exc
        return self._response_body(response, service_name)

    async def stream(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        request: AgentRequest,
        service_name: str = "internal Agent",
    ) -> AsyncIterator[dict[str, Any]]:
        """调用流式 internal 接口并返回可转发的中间 SSE 事件。"""
        try:
            async with self._client_factory(timeout=timeout_seconds) as client:
                await self._negotiate(client, base_url, require_streaming=True)
                async with client.stream(
                    "POST",
                    self._url(base_url, "/invoke/stream"),
                    json=request.model_dump(),
                ) as response:
                    response.raise_for_status()
                    async for event in self._iter_sse_events(response):
                        if event["event"] not in {"session_start", "done", "session_end"}:
                            yield event
        except httpx.TimeoutException as exc:
            raise InternalAgentClientError(f"{service_name} 流式调用超时") from exc
        except httpx.HTTPError as exc:
            raise InternalAgentClientError(f"{service_name} 流式调用不可用") from exc

    async def cancel(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        user_id: str,
        session_id: str,
        trace_id: str = "",
        service_name: str = "internal Agent",
    ) -> dict[str, Any]:
        """请求取消指定会话，并校验标准取消响应。"""
        payload = {"user_id": user_id, "session_id": session_id, "trace_id": trace_id}
        try:
            async with self._client_factory(timeout=timeout_seconds) as client:
                response = await client.post(self._url(base_url, "/cancel"), json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise InternalAgentClientError(f"{service_name} 取消超时") from exc
        except httpx.HTTPError as exc:
            raise InternalAgentClientError(f"{service_name} 取消不可用") from exc
        body = self._json_object(response, service_name)
        if body.get("status") not in {"cancelled", "not_found"}:
            raise InternalAgentClientError(f"{service_name} 返回无效取消响应")
        return body

    @staticmethod
    def _url(base_url: str, path: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise InternalAgentClientError("internal Agent 地址必须使用 HTTP(S)")
        return f"{normalized}{path}"

    async def _negotiate(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        *,
        require_streaming: bool,
    ) -> None:
        response = await client.get(self._url(base_url, "/capabilities"))
        response.raise_for_status()
        self.validate_capabilities(self._json_object(response, "internal Agent"), require_streaming=require_streaming)

    @staticmethod
    def validate_capabilities(capabilities: object, *, require_streaming: bool) -> None:
        """验证向后兼容的 internal v3 最小能力声明。"""
        if not isinstance(capabilities, dict):
            raise InternalAgentClientError("internal Agent capabilities 协议无效")
        protocol_version = capabilities.get("internal_protocol_version")
        profiles = capabilities.get("api_profiles")
        if not isinstance(protocol_version, str) or not protocol_version.startswith("muye-agent-internal/3."):
            raise InternalAgentClientError("internal Agent 不兼容 internal v3 协议")
        if not isinstance(profiles, list) or "internal" not in profiles:
            raise InternalAgentClientError("internal Agent 未声明 internal profile")
        if require_streaming and capabilities.get("supports_streaming") is not True:
            raise InternalAgentClientError("internal Agent 不支持流式调用")

    @staticmethod
    def _response_body(response: httpx.Response, service_name: str) -> dict[str, Any]:
        body = InternalAgentClient._json_object(response, service_name)
        if body.get("status") not in {"success", "error", "interrupted", "clarification_needed"}:
            raise InternalAgentClientError(f"{service_name} 返回协议无效")
        return body

    @staticmethod
    def _json_object(response: httpx.Response, service_name: str) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise InternalAgentClientError(f"{service_name} 返回非 JSON 响应") from exc
        if not isinstance(body, dict):
            raise InternalAgentClientError(f"{service_name} 返回协议无效")
        return body

    @staticmethod
    async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        event_name = ""
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if not line:
                if data_lines:
                    yield InternalAgentClient._parse_sse_event(event_name, data_lines)
                event_name = ""
                data_lines = []
            elif line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
        if data_lines:
            yield InternalAgentClient._parse_sse_event(event_name, data_lines)

    @staticmethod
    def _parse_sse_event(event_name: str, data_lines: list[str]) -> dict[str, Any]:
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError as exc:
            raise InternalAgentClientError("internal Agent 返回无效 SSE JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("event"), str):
            raise InternalAgentClientError("internal Agent 返回无效 SSE 事件")
        if event_name and event_name != payload["event"]:
            raise InternalAgentClientError("internal Agent SSE event 与 payload 不一致")
        return payload
