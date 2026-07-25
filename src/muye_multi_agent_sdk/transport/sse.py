"""SSE 信封创建和唯一序列化位置。"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ..contracts import AgentEvent, AgentResult
from .projection import public_markdown


class SseEmitter:
    """为一个请求生成严格递增的 Block Stream 信封。"""

    def __init__(self, *, session_id: str, user_id: str, stream_id: str | None = None) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.stream_id = stream_id or f"stream_{uuid.uuid4().hex}"
        self.sequence = 0
        self.started_at = int(time.time() * 1000)
        self.block_count = 0
        self._block_ids: set[str] = set()

    def frame(self, event: str, data: dict[str, Any]) -> str:
        envelope = self.envelope(event, data)
        return f"event: {event}\ndata: {json.dumps(envelope, ensure_ascii=False)}\n\n"

    def envelope(self, event: str, data: dict[str, Any]) -> dict[str, Any]:
        """创建严格递增的结构化信封，供非 HTTP transport 的适配层复用。"""
        self.sequence += 1
        return {
            "event": event,
            "sessionId": self.session_id,
            "streamId": self.stream_id,
            "userId": self.user_id,
            "seq": self.sequence,
            "timestamp": int(time.time() * 1000),
            "data": data,
        }

    def duration_ms(self) -> int:
        """返回从 emitter 创建到当前时刻的毫秒数。"""
        return int(time.time() * 1000) - self.started_at

    def register_block(self, block_id: str) -> None:
        """按稳定 block ID 计数，重复 delta 不重复累计。"""
        self._count_block(block_id)

    def session_start(self, run_id: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        """发射会话首帧；metadata 用于模型等稳定的启动信息。"""
        data: dict[str, Any] = dict(metadata or {})
        if run_id:
            data["run_id"] = run_id
        return self.frame("session_start", data)

    def agent_event(self, event: AgentEvent, *, public: bool) -> list[str]:
        if event.kind == "thinking":
            return [self.frame("thinking", event.data)]
        if event.kind == "tool":
            return [self.frame("tool", event.data)]
        if event.kind == "block":
            data = dict(event.data)
            block_id = data.get("id")
            if not isinstance(block_id, str) or not block_id:
                block_id = self._next_block_id()
                data["id"] = block_id
            self._count_block(block_id)
            return [self.frame("block", data)]
        if event.result is not None:
            return self.result_frames(event.result, public=public)
        return []

    def result_frames(self, result: AgentResult, *, public: bool) -> list[str]:
        if result.status == "success":
            block_id = self._next_block_id()
            if public:
                markdown = public_markdown(result.result_data)
                if markdown is None:
                    return [self.frame("error", {"code": "INVALID_AGENT_RESPONSE", "message": "Agent 公开响应缺少 Markdown 内容"})]
                return [self.frame("block", {"id": block_id, "type": "markdown", "delta": markdown})]
            return [
                self.frame(
                    "block",
                    {
                        "id": block_id,
                        "type": "json",
                        "content": {
                            "json_id": f"{uuid.uuid4().hex}",
                            "data": {
                                "origin_data": result.origin_data,
                                "result_data": result.result_data,
                                "prompt_data": result.prompt_data,
                            },
                        },
                    },
                )
            ]
        if result.status == "clarification_needed":
            event_name = "clarification_needed"
            message = result.clarification_question or "请补充任务信息。"
        elif result.status == "interrupted":
            event_name = "interrupted"
            message = result.error.message if result.error else "任务已中断。"
        else:
            event_name = "error"
            message = result.error.message if result.error else "Agent 执行失败。"
        if public:
            public_message = message if result.status == "clarification_needed" else "当前请求暂时无法完成，请稍后重试。"
            return [self.frame("block", {"id": self._next_block_id(), "type": "markdown", "delta": public_message})]
        return [self.frame(event_name, {"id": f"b{self.block_count}", "type": "json", "result_data": {"code": result.error.code if result.error else event_name, "message": message}})]

    def _next_block_id(self) -> str:
        candidate = self.block_count + 1
        while f"b{candidate}" in self._block_ids:
            candidate += 1
        block_id = f"b{candidate}"
        self._count_block(block_id)
        return block_id

    def _count_block(self, block_id: str) -> None:
        if block_id not in self._block_ids:
            self._block_ids.add(block_id)
            self.block_count += 1

    def done(self) -> str:
        return self.frame("done", {"totalBlocks": self.block_count, "totalEvents": self.sequence, "duration": self.duration_ms()})

    def session_end(self) -> str:
        return self.frame("session_end", {"totalBlocks": self.block_count, "duration": self.duration_ms()})
