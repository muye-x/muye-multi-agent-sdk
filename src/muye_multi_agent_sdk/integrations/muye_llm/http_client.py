"""muye-llm 网关调用使用的 HTTP client helper。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...config.models import normalize_model_base_url
from .constants import MuyeLlmApi, MuyeLlmStreamEventName
from .errors import MuyeLlmError
from .events import MuyeLlmStreamEvent


class MuyeLlmHttpClient:
    """封装 muye-llm chat 和 stream 端点的轻量异步 HTTP client。"""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        trust_env: bool,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = normalize_model_base_url(base_url)
        self.timeout = timeout
        self.trust_env = trust_env
        self.client = client

    async def post_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """发送非流式 chat 请求并返回解析后的 JSON。"""

        if self.client is not None:
            response = await self.client.post(
                f"{self.base_url}{MuyeLlmApi.CHAT_PATH}",
                json=body,
            )
            response.raise_for_status()
            return response.json()
        async with httpx.AsyncClient(
            timeout=self.timeout,
            trust_env=self.trust_env,
        ) as scoped_client:
            response = await scoped_client.post(
                f"{self.base_url}{MuyeLlmApi.CHAT_PATH}",
                json=body,
            )
            response.raise_for_status()
            return response.json()

    async def stream_events(self, body: dict[str, Any]) -> AsyncIterator[MuyeLlmStreamEvent]:
        """发送流式 chat 请求并产出结构化 muye-llm 事件。"""

        if self.client is not None:
            async with self.client.stream(
                "POST",
                f"{self.base_url}{MuyeLlmApi.STREAM_PATH}",
                json=body,
            ) as response:
                async for event in iter_stream_response(response):
                    yield event
            return

        async with httpx.AsyncClient(
            timeout=self.timeout,
            trust_env=self.trust_env,
        ) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}{MuyeLlmApi.STREAM_PATH}",
                json=body,
            ) as response:
                async for event in iter_stream_response(response):
                    yield event


async def iter_stream_response(
    response: httpx.Response,
) -> AsyncIterator[MuyeLlmStreamEvent]:
    """将 muye-llm SSE 行协议解析为结构化事件。"""

    response.raise_for_status()
    event_type = ""
    async for line in response.aiter_lines():
        if not line:
            event_type = ""
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[5:].strip())
        except json.JSONDecodeError as exc:
            raise MuyeLlmError("muye-llm流式响应格式无效") from exc
        if event_type == MuyeLlmStreamEventName.ERROR:
            message = str(payload.get("message") or "muye-llm流式调用失败")
            yield MuyeLlmStreamEvent(
                event=MuyeLlmStreamEventName.ERROR,
                message=message,
                raw=payload,
            )
            return
        if event_type == MuyeLlmStreamEventName.DONE:
            yield MuyeLlmStreamEvent(event=MuyeLlmStreamEventName.DONE, raw=payload)
            return
        if event_type == MuyeLlmStreamEventName.TOOL_CALLS:
            tool_calls = payload.get("tool_calls")
            if not isinstance(tool_calls, list) or not all(isinstance(item, dict) for item in tool_calls):
                raise MuyeLlmError("muye-llm工具调用响应格式无效")
            yield MuyeLlmStreamEvent(
                event=MuyeLlmStreamEventName.TOOL_CALLS,
                tool_calls=tool_calls,
                raw=payload,
            )
            continue
        token = str(payload.get("content") or "")
        if token:
            yield MuyeLlmStreamEvent(
                event=MuyeLlmStreamEventName.TOKEN,
                content=token,
                raw=payload,
            )
