"""模式层与 transport 层之间使用的结构化事件。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import AgentResult


EventKind = Literal["thinking", "tool", "block", "result"]


class AgentEvent(BaseModel):
    """不含 SSE 信封的 Agent 运行事件。"""

    model_config = ConfigDict(extra="forbid")

    kind: EventKind
    data: dict[str, Any] = Field(default_factory=dict)
    result: AgentResult | None = None

    @classmethod
    def thinking(cls, content: str, *, event_id: str = "thinking") -> "AgentEvent":
        return cls(kind="thinking", data={"id": event_id, "content": content, "collapsed": True})

    @classmethod
    def tool(cls, name: str, status: str, **data: Any) -> "AgentEvent":
        return cls(kind="tool", data={"name": name, "status": status, **data})

    @classmethod
    def block(cls, block_type: str, content: Any, *, block_id: str | None = None) -> "AgentEvent":
        data: dict[str, Any] = {"type": block_type, "content": content}
        if block_id:
            data["id"] = block_id
        return cls(kind="block", data=data)

    @classmethod
    def completed(cls, result: AgentResult) -> "AgentEvent":
        return cls(kind="result", result=result)
