"""可信 internal v3 Agent 服务的异步 HTTP 客户端。"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from pydantic import SecretStr

from ..contracts import AgentIdentity, AgentRequest
from ..version import INTERNAL_PROTOCOL_VERSION


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
        expected_identity: AgentIdentity | None = None,
        expected_protocol_version: str = INTERNAL_PROTOCOL_VERSION,
        expected_profile: str = "internal",
        deadline_monotonic: float | None = None,
        service_token: str | SecretStr | None = None,
    ) -> dict[str, Any]:
        """协商能力后调用非流式 internal 接口并返回原始标准响应。"""
        try:
            headers = self._request_headers(deadline_monotonic, service_token)
            async with self._client_factory(timeout=self._effective_timeout(timeout_seconds, deadline_monotonic)) as client:
                await self._negotiate(
                    client,
                    base_url,
                    require_streaming=False,
                    expected_identity=expected_identity,
                    expected_protocol_version=expected_protocol_version,
                    expected_profile=expected_profile,
                    required_features={"trusted_deadline"} if deadline_monotonic is not None else None,
                    headers=headers,
                )
                self._ensure_deadline(deadline_monotonic)
                response = await client.post(self._url(base_url, "/invoke"), json=request.model_dump(), headers=headers)
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
        expected_identity: AgentIdentity | None = None,
        expected_protocol_version: str = INTERNAL_PROTOCOL_VERSION,
        expected_profile: str = "internal",
        deadline_monotonic: float | None = None,
        service_token: str | SecretStr | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """调用流式 internal 接口并返回可转发的中间 SSE 事件。"""
        try:
            headers = self._request_headers(deadline_monotonic, service_token)
            async with self._client_factory(timeout=self._effective_timeout(timeout_seconds, deadline_monotonic)) as client:
                await self._negotiate(
                    client,
                    base_url,
                    require_streaming=True,
                    expected_identity=expected_identity,
                    expected_protocol_version=expected_protocol_version,
                    expected_profile=expected_profile,
                    required_features={"trusted_deadline"} if deadline_monotonic is not None else None,
                    headers=headers,
                )
                self._ensure_deadline(deadline_monotonic)
                async with client.stream(
                    "POST",
                    self._url(base_url, "/invoke/stream"),
                    json=request.model_dump(),
                    headers=headers,
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
        deadline_monotonic: float | None = None,
        service_token: str | SecretStr | None = None,
    ) -> dict[str, Any]:
        """请求取消指定会话，并校验标准取消响应。"""
        payload = {"user_id": user_id, "session_id": session_id, "trace_id": trace_id}
        try:
            headers = self._request_headers(deadline_monotonic, service_token)
            async with self._client_factory(timeout=self._effective_timeout(timeout_seconds, deadline_monotonic)) as client:
                self._ensure_deadline(deadline_monotonic)
                response = await client.post(self._url(base_url, "/cancel"), json=payload, headers=headers)
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
        expected_identity: AgentIdentity | None,
        expected_protocol_version: str,
        expected_profile: str,
        required_features: set[str] | None,
        headers: dict[str, str],
    ) -> None:
        response = await client.get(self._url(base_url, "/capabilities"), headers=headers)
        response.raise_for_status()
        self.validate_capabilities(
            self._json_object(response, "internal Agent"),
            require_streaming=require_streaming,
            expected_identity=expected_identity,
            expected_protocol_version=expected_protocol_version,
            expected_profile=expected_profile,
            required_features=required_features,
        )

    @staticmethod
    def validate_capabilities(
        capabilities: object,
        *,
        require_streaming: bool,
        expected_identity: AgentIdentity | None = None,
        expected_protocol_version: str = INTERNAL_PROTOCOL_VERSION,
        expected_profile: str = "internal",
        required_features: set[str] | None = None,
    ) -> None:
        """验证调用方预期的 internal protocol、profile、identity 与 features。"""
        if not isinstance(capabilities, dict):
            raise InternalAgentClientError("internal Agent capabilities 协议无效")
        protocol_version = capabilities.get("internal_protocol_version")
        profiles = capabilities.get("api_profiles")
        if protocol_version != expected_protocol_version:
            raise InternalAgentClientError("internal Agent 协议版本不匹配")
        if not isinstance(profiles, list) or expected_profile not in profiles:
            raise InternalAgentClientError("internal Agent 未声明预期 API profile")
        if require_streaming and capabilities.get("supports_streaming") is not True:
            raise InternalAgentClientError("internal Agent 不支持流式调用")
        features = capabilities.get("features", [])
        if required_features:
            if not isinstance(features, list) or not required_features.issubset(set(features)):
                raise InternalAgentClientError("internal Agent 未声明调用所需能力")
        if expected_identity is not None:
            try:
                identity = AgentIdentity.model_validate(capabilities.get("identity"))
            except Exception as exc:
                raise InternalAgentClientError("internal Agent 未声明可验证身份") from exc
            if identity != expected_identity:
                raise InternalAgentClientError("internal Agent 身份或源码校验和不匹配")

    @staticmethod
    def _effective_timeout(timeout_seconds: float, deadline_monotonic: float | None) -> float:
        """将下游 HTTP timeout 限制在 trusted caller 的剩余总 deadline 内。"""
        if deadline_monotonic is None:
            return timeout_seconds
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise InternalAgentClientError("internal Agent 调用已超过 deadline")
        return min(timeout_seconds, remaining)

    @staticmethod
    def _ensure_deadline(deadline_monotonic: float | None) -> None:
        """避免 capabilities 协商耗尽 deadline 后仍继续发起 invoke/cancel。"""
        if deadline_monotonic is not None and deadline_monotonic <= time.monotonic():
            raise InternalAgentClientError("internal Agent 调用已超过 deadline")

    @staticmethod
    def _request_headers(
        deadline_monotonic: float | None,
        service_token: str | SecretStr | None,
    ) -> dict[str, str]:
        """为受信任调用投影短期服务凭据和跨进程 deadline。"""
        headers: dict[str, str] = {}
        if service_token is not None:
            token = service_token.get_secret_value() if isinstance(service_token, SecretStr) else service_token
            normalized_token = token.strip()
            if not normalized_token:
                raise InternalAgentClientError("internal Agent 服务凭据不能为空")
            headers["Authorization"] = f"Bearer {normalized_token}"
        if deadline_monotonic is not None:
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise InternalAgentClientError("internal Agent 调用已超过 deadline")
            headers["X-Muye-Deadline-Unix-Ms"] = str(int((time.time() + remaining) * 1000))
        return headers

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
