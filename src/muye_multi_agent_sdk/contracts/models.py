"""SDK 对外稳定的数据契约。"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JsonObject = dict[str, Any]
ResultStatus = Literal["success", "error", "interrupted", "clarification_needed"]
AGENT_ID_PATTERN = r"^agent_[a-z0-9][a-z0-9_-]{2,63}$"
CHECKSUM_PATTERN = r"^[a-f0-9]{64}$"
IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$"
SEMVER_PATTERN = (
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-(?:(?:0|[1-9]\d*)|(?:\d*[A-Za-z-][0-9A-Za-z-]*))"
    r"(?:\.(?:(?:0|[1-9]\d*)|(?:\d*[A-Za-z-][0-9A-Za-z-]*)))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class AgentIdentity(BaseModel):
    """已部署 Agent 的稳定身份和可审计源版本。

    该对象只能由已校验的 descriptor/build 输入构造。它可选地出现在 v3
    capabilities 中，允许旧服务渐进迁移而不会把目录名或 URL 当作身份。
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    agent_id: str = Field(pattern=AGENT_ID_PATTERN)
    agent_version: str = Field(pattern=SEMVER_PATTERN)
    descriptor_checksum: str = Field(pattern=CHECKSUM_PATTERN)
    source_tree_checksum: str = Field(pattern=CHECKSUM_PATTERN)


class CitationBlock(BaseModel):
    """知识回答可安全公开的最小引用块，不包含物理库表或检索内部参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    citation_id: str = Field(pattern=IDENTIFIER_PATTERN)
    title: str = Field(min_length=1, max_length=512)
    source: str = Field(min_length=1, max_length=2048)
    locator: str | None = Field(default=None, max_length=1024)
    excerpt: str | None = Field(default=None, max_length=4000)


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


class ChannelTextMessage(BaseModel):
    """第三方通道可安全传递给 Agent 的文本消息。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["text"] = "text"
    content: str = Field(min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("channel message content 不能为空")
        return normalized


class ChannelInvokeRequest(BaseModel):
    """受信任通道服务到 Agent 的标准化调用。

    ``user_id`` 必须由经过认证的通道服务派生。上游 provider 的凭据、原始用户
    标识和消息关联令牌不属于该契约，避免进入 Agent 上下文或记忆后端。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    channel: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    user_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
    session_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
    trace_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{7,127}$")
    message_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
    message: ChannelTextMessage


class ChannelInvokeResponse(BaseModel):
    """Agent 对通道服务的最小文本响应，不包含内部执行数据。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: ResultStatus
    trace_id: str
    message: ChannelTextMessage | None = None
    error: "AgentError" | None = None


class AgentError(BaseModel):
    """面向调用方的结构化错误。"""

    code: str
    message: str
    recoverable: bool = False
    retry_suggestion: str | None = None


ChannelInvokeResponse.model_rebuild()


class AgentResult(BaseModel):
    """三种模式统一返回的结果，保留现有 HTTP 三字段 payload 语义。"""

    model_config = ConfigDict(extra="forbid")

    status: ResultStatus
    origin_data: JsonObject | None = None
    result_data: JsonObject | None = None
    prompt_data: str | None = None
    error: AgentError | None = None
    clarification_question: str | None = None
    citations: list[CitationBlock] = Field(default_factory=list, max_length=50)
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
        citations: list[CitationBlock] | None = None,
        trace_id: str | None = None,
    ) -> "AgentResult":
        return cls(
            status="success",
            origin_data=origin_data,
            result_data=result_data,
            prompt_data=prompt_data,
            citations=citations or [],
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
    identity: AgentIdentity | None = None

    @model_validator(mode="after")
    def validate_identity_version(self) -> "AgentMetadata":
        """防止 capabilities 把一个构建版本冒充为另一个已部署版本。"""
        if self.identity is not None and self.identity.agent_version != self.version:
            raise ValueError("identity.agent_version 必须与 AgentMetadata.version 一致")
        return self


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
    identity: AgentIdentity | None = None
    features: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("features")
    @classmethod
    def validate_features(cls, values: list[str]) -> list[str]:
        """能力名称必须稳定、去重，避免调用方依据非结构化文本协商协议。"""
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 128 for value in normalized):
            raise ValueError("features 的每一项必须是 1 至 128 个字符")
        if len(set(normalized)) != len(normalized):
            raise ValueError("features 不能重复")
        return normalized

    @model_validator(mode="after")
    def validate_identity_version(self) -> "AgentCapabilities":
        """确保协商到的版本与可审计部署身份来自同一构建。"""
        if self.identity is not None and self.identity.agent_version != self.version:
            raise ValueError("identity.agent_version 必须与 capabilities.version 一致")
        return self


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
