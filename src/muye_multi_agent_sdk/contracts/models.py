"""SDK 对外稳定的数据契约。"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


JsonObject = dict[str, Any]
ResultStatus = Literal["success", "error", "interrupted", "clarification_needed"]


class AgentContext(BaseModel):
    """一次请求的调用方身份和领域扩展信息。"""

    model_config = ConfigDict(extra="forbid")

    user_id: str = "default_user"
    session_id: str = "default_session"
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: str | None = None
    user_location: JsonObject | None = None
    files: list[JsonObject] = Field(default_factory=list)
    extra: JsonObject = Field(default_factory=dict)


class AgentRequest(BaseModel):
    """SDK 内部和 internal HTTP profile 共用的任务请求。"""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=2000)
    context: AgentContext = Field(default_factory=AgentContext)

    @model_validator(mode="after")
    def normalize_task(self) -> "AgentRequest":
        self.task = self.task.strip()
        if not self.task:
            raise ValueError("task 不能为空")
        return self


class AgentError(BaseModel):
    """面向调用方的结构化错误。"""

    code: str
    message: str
    recoverable: bool = False
    retry_suggestion: str | None = None


class AgentResult(BaseModel):
    """三种模式统一返回的结果，保留现有 HTTP 三字段 payload 语义。"""

    model_config = ConfigDict(extra="forbid")

    status: ResultStatus
    origin_data: JsonObject | None = None
    result_data: JsonObject | None = None
    prompt_data: str | None = None
    error: AgentError | None = None
    clarification_question: str | None = None
    tool_calls_made: list[str] = Field(default_factory=list)
    trace_id: str | None = None

    @classmethod
    def success(
        cls,
        result_data: JsonObject,
        *,
        origin_data: JsonObject | None = None,
        prompt_data: str | None = None,
        tool_calls_made: list[str] | None = None,
        trace_id: str | None = None,
    ) -> "AgentResult":
        return cls(
            status="success",
            origin_data=origin_data,
            result_data=result_data,
            prompt_data=prompt_data,
            tool_calls_made=tool_calls_made or [],
            trace_id=trace_id,
        )

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        recoverable: bool = False,
        trace_id: str | None = None,
    ) -> "AgentResult":
        return cls(
            status="error",
            error=AgentError(code=code, message=message, recoverable=recoverable),
            trace_id=trace_id,
        )

    @classmethod
    def clarification(cls, question: str, *, trace_id: str | None = None) -> "AgentResult":
        return cls(status="clarification_needed", clarification_question=question, trace_id=trace_id)

    @classmethod
    def interrupted(
        cls,
        code: str,
        message: str,
        *,
        recoverable: bool = False,
        trace_id: str | None = None,
    ) -> "AgentResult":
        return cls(
            status="interrupted",
            error=AgentError(code=code, message=message, recoverable=recoverable),
            trace_id=trace_id,
        )

    @model_validator(mode="after")
    def validate_status(self) -> "AgentResult":
        if self.status == "success" and not self.result_data:
            raise ValueError("success 结果必须包含非空 result_data")
        if self.status == "error" and self.error is None:
            raise ValueError("error 结果必须包含 error")
        if self.status == "interrupted" and self.error is None:
            raise ValueError("interrupted 结果必须包含 error")
        if self.status == "clarification_needed" and not self.clarification_question:
            raise ValueError("clarification_needed 结果必须包含 clarification_question")
        return self


class AgentMetadata(BaseModel):
    """Agent 的稳定身份信息，由三种模式共用。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=500)
    supported_intents: list[str] = Field(default_factory=list)


class ToolCapability(BaseModel):
    """对外可调用工具能力；Graph 内部节点不应放入该字段。"""

    name: str
    description: str
    parameters: JsonObject = Field(default_factory=dict)


class AgentCapabilities(BaseModel):
    """`/capabilities` 的统一响应模型。"""

    agent_name: str
    version: str
    description: str
    supported_intents: list[str] = Field(default_factory=list)
    tools: list[ToolCapability] = Field(default_factory=list)
    api_profiles: list[str] = Field(default_factory=lambda: ["internal"])
    supports_streaming: bool = True
    max_task_length: int = 2000
    internal_protocol_version: str
    public_protocol_version: str | None = None


class CancelRequest(BaseModel):
    """取消正在运行任务的会话定位信息。"""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    session_id: str
    trace_id: str | None = None
    run_id: str | None = None


class CancelResponse(BaseModel):
    """取消请求的标准响应。"""

    status: Literal["cancelled", "not_found"]
    message: str
    run_id: str | None = None
    trace_id: str | None = None
