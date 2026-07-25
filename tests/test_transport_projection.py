"""public HTTP 与 SSE 展示字段投影回归测试。"""
from __future__ import annotations

import asyncio
import json

import httpx

from muye_multi_agent_sdk import AgentConfig, AgentEvent, AgentMetadata, AgentRequest, AgentResult, ApiConfig, CustomAgent, create_app
from muye_multi_agent_sdk.transport.sse import SseEmitter


class ProjectingAgent(CustomAgent):
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="projecting-agent", version="1", description="投影测试")

    async def execute(self, request: AgentRequest, **_kwargs: object) -> AgentResult:
        if request.task == "missing":
            data = {"markdown": "   "}
        else:
            data = {"markdown": "   ", "summary": "  fallback  "}
        return AgentResult.success(data, trace_id=request.context.trace_id)


async def _post(app: object, path: str, body: dict[str, object]) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        return await client.post(path, json=body)


def _app() -> object:
    return create_app(ProjectingAgent(AgentConfig(api=ApiConfig(profiles={"public"}, public_path="/api/v1/project"))))


def test_public_http_and_sse_use_the_same_markdown_fallback() -> None:
    app = _app()
    payload = {"user_input": "fallback", "user_id": "u", "session_id": "s"}

    response = asyncio.run(_post(app, "/api/v1/project/invoke", payload))
    stream = asyncio.run(_post(app, "/api/v1/project/invoke/stream", payload))

    assert response.json()["markdown"] == "fallback"
    assert '"delta": "fallback"' in stream.text


def test_public_http_and_sse_reject_missing_markdown_consistently() -> None:
    app = _app()
    payload = {"user_input": "missing", "user_id": "u", "session_id": "s"}

    response = asyncio.run(_post(app, "/api/v1/project/invoke", payload))
    stream = asyncio.run(_post(app, "/api/v1/project/invoke/stream", payload))

    assert response.json()["error"]["code"] == "INVALID_AGENT_RESPONSE"
    assert '"code": "INVALID_AGENT_RESPONSE"' in stream.text


def test_sse_emitter_preserves_start_metadata_and_counts_delta_block_once() -> None:
    emitter = SseEmitter(session_id="s", user_id="u")

    start = emitter.session_start(metadata={"model": "muye-model"})
    emitter.agent_event(AgentEvent(kind="block", data={"id": "b1", "type": "markdown", "delta": "你"}), public=False)
    emitter.agent_event(AgentEvent(kind="block", data={"id": "b1", "type": "markdown", "delta": "好"}), public=False)
    done = emitter.done()

    start_envelope = json.loads(start.split("data: ", 1)[1])
    done_envelope = json.loads(done.split("data: ", 1)[1])
    assert start_envelope["data"] == {"model": "muye-model"}
    assert done_envelope["data"]["totalBlocks"] == 1


def test_sse_emitter_counts_public_terminal_fallback_as_a_block() -> None:
    emitter = SseEmitter(session_id="s", user_id="u")

    frame = emitter.result_frames(AgentResult.failure("UPSTREAM_ERROR", "上游失败"), public=True)[0]
    done = emitter.done()

    envelope = json.loads(frame.split("data: ", 1)[1])
    done_envelope = json.loads(done.split("data: ", 1)[1])
    assert envelope["event"] == "block"
    assert envelope["data"]["id"] == "b1"
    assert done_envelope["data"]["totalBlocks"] == 1
