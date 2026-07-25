"""Muye LLM 适配器的流式协议与连接生命周期测试。"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
from langchain_core.messages import HumanMessage
from pydantic import PrivateAttr

from muye_multi_agent_sdk.integrations.muye_llm import MuyeLlmChatModel


class DelayedStreamModel(MuyeLlmChatModel):
    async def stream_events(self, _messages: object, **_kwargs: object):
        from muye_multi_agent_sdk.integrations.muye_llm.events import MuyeLlmStreamEvent

        yield MuyeLlmStreamEvent(event="token", content="first")
        await asyncio.sleep(0.2)
        yield MuyeLlmStreamEvent(event="token", content="second")


class BlockingStreamModel(MuyeLlmChatModel):
    _closed: threading.Event = PrivateAttr(default_factory=threading.Event)

    @property
    def closed(self) -> threading.Event:
        return self._closed

    async def stream_events(self, _messages: object, **_kwargs: object):
        from muye_multi_agent_sdk.integrations.muye_llm.events import MuyeLlmStreamEvent

        try:
            yield MuyeLlmStreamEvent(event="token", content="first")
            await asyncio.Event().wait()
        finally:
            self._closed.set()


class RecordingRunManager:
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str, **_kwargs: object) -> None:
        self.tokens.append(token)


def test_async_stream_reads_structured_tool_call_event() -> None:
    body = (
        'event: tool_calls\n'
        'data: {"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\\"id\\":1}"}}]}\n\n'
        'event: done\n'
        'data: {}\n\n'
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async def collect() -> list[object]:
        model = MuyeLlmChatModel(base_url="http://llm.test")
        model.set_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        chunks = [chunk async for chunk in model._astream([HumanMessage(content="test")])]
        await model._client.aclose()  # type: ignore[union-attr]
        return chunks

    chunks = asyncio.run(collect())

    assert chunks[0].message.tool_calls[0]["name"] == "lookup"
    assert chunks[0].message.tool_calls[0]["args"] == {"id": 1}


def test_request_body_omits_unconfigured_model_selection() -> None:
    model = MuyeLlmChatModel(base_url="http://llm.test")

    body = model.build_request_body([{"role": "user", "content": "test"}])

    assert "model" not in body
    assert "enable_thinking" not in body


def test_non_stream_and_stream_forward_static_model_selection() -> None:
    captured_bodies: list[dict[str, object]] = []
    stream_body = 'event: done\ndata: {}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        if request.url.path.endswith("/stream"):
            return httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"success": True, "data": {"content": "ok"}})

    async def run() -> None:
        model = MuyeLlmChatModel(
            base_url="http://llm.test",
            model_name="reasoning",
            enable_thinking=False,
        )
        model.set_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        assert await model.chat_text([{"role": "user", "content": "test"}]) == "ok"
        events = [event async for event in model.stream_events([{"role": "user", "content": "test"}])]
        assert [event.event for event in events] == ["done"]
        await model._client.aclose()  # type: ignore[union-attr]

    asyncio.run(run())

    assert len(captured_bodies) == 2
    assert all(body["model"] == "reasoning" for body in captured_bodies)
    assert all(body["enable_thinking"] is False for body in captured_bodies)


def test_bind_tools_preserves_static_model_selection() -> None:
    model = MuyeLlmChatModel(
        base_url="http://llm.test",
        model_name="reasoning",
        enable_thinking=True,
    )

    bound_model = model.bind_tools(
        [{"type": "function", "function": {"name": "lookup", "description": "lookup", "parameters": {}}}]
    )
    body = bound_model.build_request_body([{"role": "user", "content": "test"}])

    assert body["model"] == "reasoning"
    assert body["enable_thinking"] is True
    asyncio.run(model.aclose())


def test_sync_stream_yields_before_async_stream_finishes() -> None:
    model = DelayedStreamModel(base_url="http://llm.test")

    started_at = time.monotonic()
    iterator = model._stream([HumanMessage(content="test")])
    first = next(iterator)
    elapsed = time.monotonic() - started_at
    remaining = list(iterator)

    assert first.message.content == "first"
    assert [chunk.message.content for chunk in remaining] == ["second"]
    assert elapsed < 0.15


def test_sync_stream_notifies_run_manager_for_each_text_chunk() -> None:
    model = DelayedStreamModel(base_url="http://llm.test")
    manager = RecordingRunManager()

    chunks = list(model._stream([HumanMessage(content="test")], run_manager=manager))

    assert [chunk.message.content for chunk in chunks] == ["first", "second"]
    assert manager.tokens == ["first", "second"]


def test_sync_stream_close_cancels_background_producer() -> None:
    model = BlockingStreamModel(base_url="http://llm.test")
    iterator = model._stream([HumanMessage(content="test")])

    assert next(iterator).message.content == "first"
    iterator.close()

    assert model.closed.wait(timeout=1)


def test_sdk_owned_http_client_is_reused_and_closed() -> None:
    async def run() -> None:
        model = MuyeLlmChatModel(base_url="http://llm.test")
        first = model._http_client()
        second = model._http_client()
        assert first is second
        await model.aclose()
        assert first.is_closed

    asyncio.run(run())
