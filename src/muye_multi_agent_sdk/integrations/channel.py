"""第三方通道服务调用 Agent 的受认证 HTTP 客户端。"""
from __future__ import annotations

import math
from typing import Any

import httpx

from ..contracts import ChannelInvokeRequest, ChannelInvokeResponse


class ChannelAgentClientError(RuntimeError):
    """通道调用端点不可用或不符合 Channel 协议。"""


class ChannelAgentClient:
    """调用标准 Channel endpoint；外部写操作的重试由 provider 服务自行决定。"""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 60,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._token = token.strip()
        if not self._base_url.startswith(("http://", "https://")) or not self._token:
            raise ValueError("Channel Agent client 必须配置 HTTP(S) 地址和服务凭据")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Channel Agent client timeout_seconds 必须是有限正数")
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def invoke(self, request: ChannelInvokeRequest) -> ChannelInvokeResponse:
        """发送一次调用，不重试以避免重复触发 Agent 工具。"""
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self._base_url}/internal/v1/channels/invoke",
                headers={"Authorization": f"Bearer {self._token}"},
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            payload: Any = response.json()
            return ChannelInvokeResponse.model_validate(payload)
        except (httpx.HTTPError, ValueError) as exc:
            raise ChannelAgentClientError("Channel Agent 调用失败") from exc

    async def aclose(self) -> None:
        """仅关闭由本客户端创建的连接池。"""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds, trust_env=False)
        return self._client
