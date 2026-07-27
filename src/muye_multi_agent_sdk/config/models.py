"""独立 SDK 的显式配置与环境加载。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


DEFAULT_MUYE_LLM_BASE_URL = "http://127.0.0.1:9850"
DEFAULT_MUYE_DATA_BASE_URL = "http://127.0.0.1:9840"


def normalize_model_base_url(value: str) -> str:
    """规范化并校验模型服务地址，供所有公开构造边界复用。"""
    normalized = value.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("模型 base_url 必须以 http:// 或 https:// 开头")
    return normalized


def normalize_data_base_url(value: str) -> str:
    """规范化并校验可信内网 muye-data 地址。"""
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("数据服务 base_url 必须是有效的 HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("数据服务 base_url 必须以 http:// 或 https:// 开头")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("数据服务 base_url 不能包含凭据")
    return normalized


def _optional_env_text(values: dict[str, str], *names: str) -> str | None:
    """按优先级读取可选文本配置，空值显式表示未设置。"""
    for name in names:
        if name not in values:
            continue
        value = values[name].strip()
        return value or None
    return None


def _optional_env_bool(values: dict[str, str], name: str) -> bool | None:
    """读取可选布尔配置；未设置时保留网关默认语义。"""
    raw_value = values.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是布尔值")


class ModelConfig(BaseModel):
    """内置聊天模型工厂的连接、采样与模型选择配置。

    本类由应用在启动时构造，并由 ``build_chat_model()`` 消费；不在构造过程中读取
    进程环境。``provider="muye"`` 时，模型 alias 与 thinking 能力由 muye-llm
    网关校验；未指定的可选项会保留给网关默认配置处理。
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["muye", "openai_compatible"] = Field(
        default="muye",
        description="内置模型适配器类型；muye 通过内部网关调用，openai_compatible 直接使用兼容接口。",
    )
    model: str | None = Field(
        default=None,
        description="模型标识；muye 时为网关注册 alias，None 表示使用网关默认模型；OpenAI-compatible 时必填。",
    )
    base_url: str = Field(
        default=DEFAULT_MUYE_LLM_BASE_URL,
        description="模型服务基础地址；会规范化末尾斜杠并要求使用 http 或 https。",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI-compatible 服务的可选访问密钥；muye 网关模式通常不由 SDK 持有上游密钥。",
    )
    enable_thinking: bool | None = Field(
        default=None,
        description="muye 请求的 thinking 覆盖；None 时使用网关默认值，True/False 时显式覆盖。",
    )
    temperature: float = Field(
        default=0.1,
        ge=0,
        le=2,
        description="生成随机性；值越低越稳定，值越高越发散。",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="单次模型响应允许生成的最大 token 数。",
    )
    timeout_seconds: float = Field(
        default=30,
        gt=0,
        le=300,
        description="SDK 等待一次模型 HTTP 调用完成的超时时间，单位为秒。",
    )

    @model_validator(mode="after")
    def validate_model_selection(self) -> "ModelConfig":
        self.base_url = normalize_model_base_url(self.base_url)
        if self.model is not None:
            self.model = self.model.strip()
            if not self.model:
                raise ValueError("模型名不能为空")
        if self.provider == "openai_compatible":
            if self.model is None:
                raise ValueError("OpenAI-compatible provider 必须配置模型名")
            if self.enable_thinking is not None:
                raise ValueError("enable_thinking 仅支持 muye provider")
        return self


class DataConfig(BaseModel):
    """SDK 到 muye-data 的只读 HTTP 客户端配置。

    配置只包含可信服务地址和请求预算；数据库连接、物理表名与索引参数只能由
    muye-data 部署配置持有，不能通过 Agent 或单次请求传入。
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(
        default=DEFAULT_MUYE_DATA_BASE_URL,
        description="muye-data 服务基础地址，只允许 HTTP(S)。",
    )
    timeout_seconds: float = Field(
        default=15,
        gt=0,
        le=300,
        allow_inf_nan=False,
        description="一次完整 retrieve 请求的 HTTP 超时，单位为秒。",
    )
    max_retries: int = Field(
        default=0,
        ge=0,
        le=1,
        description="完整只读请求的额外重试次数；默认不重试以避免放大下游模型调用。",
    )

    @model_validator(mode="after")
    def normalize_base_url(self) -> "DataConfig":
        self.base_url = normalize_data_base_url(self.base_url)
        return self


class ContextConfig(BaseModel):
    """短期会话上下文与 LangGraph checkpointer 配置。

    上下文默认关闭；仅当请求 profile 同时出现在 ``enabled_profiles`` 时，运行时才会
    初始化指定后端。SQLite 和 Postgres 连接信息只在对应后端实际启用时使用。
    """

    model_config = ConfigDict(extra="forbid")

    enabled_profiles: set[Literal["internal", "public"]] = Field(
        default_factory=set,
        description="允许保存和恢复短期上下文的 API profile 集合；空集合表示完全禁用。",
    )
    backend: Literal["memory", "sqlite", "postgres"] = Field(
        default="memory",
        description="启用上下文时使用的 checkpointer 后端类型。",
    )
    sqlite_path: str = Field(
        default="agent_checkpoints.db",
        description="backend 为 sqlite 时的本地 checkpoint 数据库路径。",
    )
    postgres_uri: SecretStr | None = Field(
        default=None,
        description="backend 为 postgres 时的连接 URI；None 时该后端无法初始化。",
    )
    postgres_pool_min_size: int = Field(
        default=1,
        ge=1,
        description="Postgres 连接池保持的最小连接数。",
    )
    postgres_pool_max_size: int = Field(
        default=10,
        ge=1,
        description="Postgres 连接池允许的最大连接数。",
    )
    postgres_pool_timeout_seconds: float = Field(
        default=30,
        gt=0,
        description="连接池耗尽时等待可用连接的最长时间。",
    )
    postgres_pool_max_idle_seconds: float = Field(
        default=300,
        gt=0,
        description="超过最小连接数的空闲连接保留时间。",
    )
    postgres_pool_max_lifetime_seconds: float = Field(
        default=3600,
        gt=0,
        description="连接在池中的最长生命周期。",
    )

    @model_validator(mode="after")
    def validate_postgres_pool(self) -> "ContextConfig":
        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError("Postgres 连接池 min_size 不能大于 max_size")
        return self


class IntentGuardConfig(BaseModel):
    """请求进入 Agent 前的可选意图安全分类配置。

    守卫关闭时其余字段不会参与执行。分类失败采用 fail-open，避免模型依赖故障阻断
    正常请求；因此该组件不应作为唯一的安全边界。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="是否在每个请求执行前运行意图分类；默认关闭。",
    )
    timeout_seconds: float = Field(
        default=5,
        gt=0,
        le=60,
        description="意图分类模型调用的独立超时时间，单位为秒。",
    )
    context_aware: bool = Field(
        default=True,
        description="上下文启用时是否让守卫参考同会话最近消息；关闭时保持单轮分类行为。",
    )
    history_max_messages: int = Field(
        default=6,
        ge=1,
        le=20,
        description="守卫可读取的最近人机文本消息上限；只计 HumanMessage 与 AIMessage。",
    )
    history_max_chars: int = Field(
        default=2000,
        ge=128,
        le=10000,
        description="守卫历史文本的总字符上限，超出时优先保留最近内容。",
    )
    history_io_timeout_seconds: float = Field(
        default=1,
        gt=0,
        le=60,
        description="读取或写入守卫会话历史的最长等待时间，单位为秒；超时不会改变当前澄清响应。",
    )
    meaningful_definition: str = Field(
        default="",
        description="追加到守卫提示词的业务有效输入定义，例如允许补充已追问的业务字段。",
    )
    meaningless_definition: str = Field(
        default="",
        description="追加到守卫提示词的业务无意义输入定义；不能覆盖 SDK 固定的违规安全规则。",
    )


class ApiConfig(BaseModel):
    """SDK HTTP transport 暴露的 profile 与 public 路由配置。

    profile 决定 ``create_app()`` 注册哪些端点；public profile 会额外执行安全的输出
    投影，不应假定与 internal 返回相同的数据结构。
    """

    model_config = ConfigDict(extra="forbid")

    profiles: set[Literal["internal", "public"]] = Field(
        default_factory=lambda: {"internal"},
        description="需要注册的 API profile；默认仅暴露 internal SDK 协议端点。",
    )
    public_path: str | None = Field(
        default=None,
        description="public profile 的可选基础路由；None 时按 Agent metadata 自动生成。",
    )


class AgentConfig(BaseModel):
    """Agent 运行时的聚合配置根对象。

    该对象将模型、上下文、意图守卫和 HTTP transport 配置集中注入各运行时组件。
    直接构造保持纯粹；只有 ``from_env()`` 负责从 shell 与可选 `.env` 收集值。
    """

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="内置聊天模型的连接、选择和采样配置。",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="按需创建的 muye-data 只读客户端配置。",
    )
    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="按 profile 启用的短期会话上下文配置。",
    )
    intent_guard: IntentGuardConfig = Field(
        default_factory=IntentGuardConfig,
        description="请求前意图分类和 fail-open 行为配置。",
    )
    api: ApiConfig = Field(
        default_factory=ApiConfig,
        description="HTTP API profile、public 路由与输出边界配置。",
    )
    request_timeout_seconds: float = Field(
        default=60,
        gt=0,
        le=600,
        description="整个 Agent 请求生命周期的超时上限，包含守卫、执行和流式处理，单位为秒。",
    )

    @classmethod
    def from_env(cls, env_file: Path | str | None = Path(".env")) -> "AgentConfig":
        """从 shell 与可选 `.env` 读取配置，shell 优先且不修改环境。"""
        values: dict[str, str] = {}
        if env_file and Path(env_file).is_file():
            from dotenv import dotenv_values

            values.update({key: value for key, value in dotenv_values(env_file).items() if value is not None})
        values.update(os.environ)
        provider = values.get("MUYE_SDK_MODEL_PROVIDER", "muye")
        profiles = {item.strip() for item in values.get("MUYE_SDK_API_PROFILES", "internal").split(",") if item.strip()}
        context_profiles = {item.strip() for item in values.get("MUYE_SDK_CONTEXT_PROFILES", "").split(",") if item.strip()}
        return cls(
            model=ModelConfig(
                provider=provider,
                model=_optional_env_text(values, "MUYE_SDK_MODEL", "MUYE_LLM_MODEL"),
                base_url=values.get(
                    "MUYE_SDK_MODEL_BASE_URL",
                    values.get("MUYE_LLM_BASE_URL", DEFAULT_MUYE_LLM_BASE_URL),
                ),
                api_key=values.get("MUYE_SDK_MODEL_API_KEY") or None,
                enable_thinking=_optional_env_bool(values, "MUYE_SDK_MODEL_ENABLE_THINKING"),
                temperature=float(values.get("MUYE_SDK_MODEL_TEMPERATURE", "0.1")),
                max_tokens=int(values.get("MUYE_SDK_MODEL_MAX_TOKENS", "4096")),
                timeout_seconds=float(values.get("MUYE_SDK_MODEL_TIMEOUT_SECONDS", "30")),
            ),
            data=DataConfig(
                base_url=values.get("MUYE_SDK_DATA_BASE_URL", DEFAULT_MUYE_DATA_BASE_URL),
                timeout_seconds=float(values.get("MUYE_SDK_DATA_TIMEOUT_SECONDS", "15")),
                max_retries=int(values.get("MUYE_SDK_DATA_MAX_RETRIES", "0")),
            ),
            context=ContextConfig(
                enabled_profiles=context_profiles,
                backend=values.get("MUYE_SDK_CONTEXT_BACKEND", "memory"),
                sqlite_path=values.get("MUYE_SDK_CONTEXT_SQLITE_PATH", "agent_checkpoints.db"),
                postgres_uri=values.get("MUYE_SDK_CONTEXT_POSTGRES_URI") or None,
                postgres_pool_min_size=int(values.get("MUYE_SDK_CONTEXT_POSTGRES_POOL_MIN_SIZE", "1")),
                postgres_pool_max_size=int(values.get("MUYE_SDK_CONTEXT_POSTGRES_POOL_MAX_SIZE", "10")),
                postgres_pool_timeout_seconds=float(values.get("MUYE_SDK_CONTEXT_POSTGRES_POOL_TIMEOUT_SECONDS", "30")),
                postgres_pool_max_idle_seconds=float(values.get("MUYE_SDK_CONTEXT_POSTGRES_POOL_MAX_IDLE_SECONDS", "300")),
                postgres_pool_max_lifetime_seconds=float(values.get("MUYE_SDK_CONTEXT_POSTGRES_POOL_MAX_LIFETIME_SECONDS", "3600")),
            ),
            intent_guard=IntentGuardConfig(
                enabled=values.get("MUYE_SDK_INTENT_GUARD", "false").lower() == "true",
                timeout_seconds=float(values.get("MUYE_SDK_INTENT_GUARD_TIMEOUT_SECONDS", "5")),
                context_aware=values.get("MUYE_SDK_INTENT_GUARD_CONTEXT_AWARE", "true").lower() == "true",
                history_max_messages=int(values.get("MUYE_SDK_INTENT_GUARD_HISTORY_MAX_MESSAGES", "6")),
                history_max_chars=int(values.get("MUYE_SDK_INTENT_GUARD_HISTORY_MAX_CHARS", "2000")),
                history_io_timeout_seconds=float(values.get("MUYE_SDK_INTENT_GUARD_HISTORY_IO_TIMEOUT_SECONDS", "1")),
                meaningful_definition=values.get("MUYE_SDK_INTENT_MEANINGFUL_DEFINITION", ""),
                meaningless_definition=values.get("MUYE_SDK_INTENT_MEANINGLESS_DEFINITION", ""),
            ),
            api=ApiConfig(profiles=profiles, public_path=values.get("MUYE_SDK_PUBLIC_PATH") or None),
            request_timeout_seconds=float(values.get("MUYE_SDK_REQUEST_TIMEOUT_SECONDS", "60")),
        )
