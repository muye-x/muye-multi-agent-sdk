"""FastAPI transport，保留现有 internal/public HTTP 与 SSE wire contract。"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from inspect import isawaitable
import re
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..contracts import AgentContext, AgentRequest, AgentResult, CancelRequest
from ..modes import BaseAgent
from ..runtime import ExecutionOptions
from ..version import SDK_VERSION
from .projection import public_markdown
from .sse import SseEmitter

PublicRequestAdapter = Callable[[dict[str, Any]], AgentRequest]
InternalRequestVerifier = Callable[[Request], object]
_DEADLINE_HEADER = "X-Muye-Deadline-Unix-Ms"


def create_app(
    agent: BaseAgent,
    *,
    public_request_adapter: PublicRequestAdapter | None = None,
    cors_origins: list[str] | None = None,
    internal_request_verifier: InternalRequestVerifier | None = None,
) -> FastAPI:
    """创建标准 Agent ASGI 应用；内部认证由部署层注入 verifier。"""
    from fastapi.middleware.cors import CORSMiddleware

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await agent.aclose()

    app = FastAPI(title=agent.metadata.name, version=agent.metadata.version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[] if cors_origins is None else cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "agent": agent.metadata.name, "version": agent.metadata.version, "sdk_version": SDK_VERSION}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        """声明 SDK transport 已初始化；外部依赖探针由部署服务另行负责。"""
        return {"status": "ready", "agent": agent.metadata.name, "version": agent.metadata.version}

    @app.get("/capabilities")
    async def capabilities(request: Request) -> dict[str, Any]:
        await _verify_internal_request(request, internal_request_verifier)
        result = agent.capabilities().model_dump()
        result["sdk_version"] = SDK_VERSION
        return result

    if "internal" in agent.config.api.profiles:
        @app.post("/invoke")
        async def invoke(request: AgentRequest, raw_request: Request) -> dict[str, Any]:
            await _verify_internal_request(raw_request, internal_request_verifier)
            return _internal_response(
                await agent.invoke(
                    request,
                    options=_internal_options(agent, raw_request, internal_request_verifier),
                )
            )

        @app.post("/invoke/stream")
        async def invoke_stream(request: AgentRequest, raw_request: Request) -> StreamingResponse:
            await _verify_internal_request(raw_request, internal_request_verifier)
            return _streaming_response(
                agent,
                request,
                profile="internal",
                options=_internal_options(agent, raw_request, internal_request_verifier),
            )

        @app.post("/cancel")
        async def cancel(request: CancelRequest, raw_request: Request) -> dict[str, Any]:
            await _verify_internal_request(raw_request, internal_request_verifier)
            return (await agent.cancel(user_id=request.user_id, session_id=request.session_id, profile="internal", run_id=request.run_id)).model_dump()

    if "public" in agent.config.api.profiles:
        public_path = (agent.config.api.public_path or f"/api/v1/{agent.metadata.name.replace('_', '-')}").rstrip("/")

        @app.post(f"{public_path}/invoke")
        async def public_invoke(payload: dict[str, Any]) -> dict[str, Any]:
            request = _public_request(payload, public_request_adapter)
            return _public_response(await agent.invoke(request, options=ExecutionOptions(profile="public", context_enabled="public" in agent.config.context.enabled_profiles, public_route=True)))

        @app.post(f"{public_path}/invoke/stream")
        async def public_stream(payload: dict[str, Any]) -> StreamingResponse:
            return _streaming_response(
                agent,
                _public_request(payload, public_request_adapter),
                profile="public",
                options=ExecutionOptions(
                    profile="public",
                    context_enabled="public" in agent.config.context.enabled_profiles,
                    public_route=True,
                ),
            )

        @app.post(f"{public_path}/cancel")
        async def public_cancel(request: CancelRequest) -> dict[str, Any]:
            return (await agent.cancel(user_id=request.user_id, session_id=request.session_id, profile="public", run_id=request.run_id)).model_dump()

    return app


def _public_request(payload: dict[str, Any], adapter: PublicRequestAdapter | None) -> AgentRequest:
    if adapter is not None:
        return adapter(payload)
    user_input = payload.get("user_input")
    if not isinstance(user_input, str) or not user_input.strip():
        raise HTTPException(status_code=422, detail="user_input 为必填字段")
    known = {"user_input", "user_id", "session_id", "trace_id", "turn_id", "files", "user_location", "extra"}
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    extra = {**extra, **{key: value for key, value in payload.items() if key not in known}}
    return AgentRequest(
        task=user_input,
        context=AgentContext(
            user_id=str(payload.get("user_id") or "default_user"),
            session_id=str(payload.get("session_id") or "default_session"),
            trace_id=str(payload.get("trace_id") or "") or AgentContext().trace_id,
            turn_id=str(payload["turn_id"]) if payload.get("turn_id") else None,
            files=payload.get("files") if isinstance(payload.get("files"), list) else [],
            user_location=payload.get("user_location") if isinstance(payload.get("user_location"), dict) else None,
            extra=extra,
        ),
    )


def _internal_response(result: AgentResult) -> dict[str, Any]:
    body: dict[str, Any] = {"status": result.status, "trace_id": result.trace_id, "tool_calls_made": result.tool_calls_made}
    if result.status == "success":
        body["payload"] = {"origin_data": result.origin_data, "result_data": result.result_data, "prompt_data": result.prompt_data}
        if result.citations:
            body["citations"] = [citation.model_dump(mode="json") for citation in result.citations]
    if result.error is not None:
        body["error"] = result.error.model_dump()
    if result.clarification_question:
        body["clarification_question"] = result.clarification_question
    return body


def _public_response(result: AgentResult) -> dict[str, Any]:
    if result.status != "success":
        markdown = result.clarification_question if result.status == "clarification_needed" else "当前请求暂时无法完成，请稍后重试。"
        error = {"code": result.error.code, "message": "请求未完成", "recoverable": result.error.recoverable} if result.error else None
        return {"status": result.status, "markdown": markdown, "error": error, "trace_id": result.trace_id}
    result_data = result.result_data or {}
    markdown = public_markdown(result_data)
    if markdown is None:
        return {
            "status": "error",
            "markdown": "当前请求暂时无法完成，请稍后重试。",
            "error": {"code": "INVALID_AGENT_RESPONSE", "message": "请求未完成", "recoverable": False},
            "trace_id": result.trace_id,
        }
    body = {
        "status": "success",
        "markdown": markdown,
        "chart": result_data.get("chart") or result_data.get("charts"),
        "json_data": result_data.get("json_data"),
        "trace_id": result.trace_id,
    }
    if result.citations:
        body["citations"] = [citation.model_dump(mode="json") for citation in result.citations]
    return body


async def _verify_internal_request(
    request: Request,
    verifier: InternalRequestVerifier | None,
) -> None:
    """调用部署注入的认证器；未配置时保持旧 internal 网络部署兼容。"""
    if verifier is None:
        return
    verified = verifier(request)
    if isawaitable(verified):
        verified = await verified
    if verified is False:
        raise HTTPException(status_code=401, detail={"code": "AUTHENTICATION_ERROR", "message": "internal 服务认证失败"})


def _internal_options(
    agent: BaseAgent,
    request: Request,
    verifier: InternalRequestVerifier | None,
) -> ExecutionOptions:
    """仅在 internal verifier 成功后接纳跨进程 deadline，避免请求体伪造预算。"""
    deadline_header = request.headers.get(_DEADLINE_HEADER) if verifier is not None else None
    deadline_monotonic: float | None = None
    if deadline_header is not None:
        if re.fullmatch(r"[0-9]{1,16}", deadline_header) is None:
            raise HTTPException(status_code=422, detail={"code": "VALIDATION_ERROR", "message": "internal deadline 格式无效"})
        deadline_monotonic = time.monotonic() + (int(deadline_header) / 1000 - time.time())
    return ExecutionOptions(
        profile="internal",
        context_enabled="internal" in agent.config.context.enabled_profiles,
        deadline_monotonic=deadline_monotonic,
    )


def _streaming_response(
    agent: BaseAgent,
    request: AgentRequest,
    *,
    profile: str,
    options: ExecutionOptions,
) -> StreamingResponse:
    async def source():
        emitter = SseEmitter(session_id=request.context.session_id, user_id=request.context.user_id)
        yield emitter.session_start()
        try:
            async for event in agent.stream(request, options=options):
                for frame in emitter.agent_event(event, public=profile == "public"):
                    yield frame
        finally:
            yield emitter.done()
            yield emitter.session_end()

    return StreamingResponse(source(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
