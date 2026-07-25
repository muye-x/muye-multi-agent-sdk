"""muye-llm 流式事件 DTO。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MuyeLlmStreamEvent(BaseModel):
    """muye-llm 标准流式事件。"""

    event: str = Field(description="事件类型，当前支持 token、tool_calls、error、done")
    content: str = Field(default="", description="token 事件中的文本增量")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="tool_calls 事件中的 OpenAI 工具调用")
    message: str = Field(default="", description="error 事件中的错误信息")
    raw: dict[str, Any] = Field(default_factory=dict, description="原始 data payload")
