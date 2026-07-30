"""标准 Agent 模板的受限配置加载与协议断言辅助函数。"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .contracts import AgentCapabilities, AgentIdentity
from .version import INTERNAL_PROTOCOL_VERSION


ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


class TemplateRuntimeError(RuntimeError):
    """模板启动前配置或契约失败的稳定分类错误。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CapabilityProvider(Protocol):
    """供模板契约测试消费的最小 Agent 能力接口。"""

    def capabilities(self) -> AgentCapabilities:
        """返回当前运行时实际公开的 capabilities。"""


def load_yaml_config(
    path: Path,
    model_type: type[ConfigModel],
    *,
    max_bytes: int = 1_000_000,
) -> ConfigModel:
    """安全读取有限大小的本地 YAML 并交给严格 Pydantic 模型验证。

    模板只把版本控制中的配置作为启动输入，不跟随 symlink，也不接受任意 Python
    object tag。调用方必须提供 ``extra='forbid'`` 的模型以防止配置漂移。
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes 必须是正整数")
    if path.is_symlink() or not path.is_file():
        raise TemplateRuntimeError("TEMPLATE_CONFIG_NOT_FOUND", f"模板配置不可用：{path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TemplateRuntimeError("TEMPLATE_CONFIG_READ_FAILED", f"无法读取模板配置：{path}") from exc
    if len(payload) > max_bytes:
        raise TemplateRuntimeError("TEMPLATE_CONFIG_TOO_LARGE", "模板配置超过大小限制")
    try:
        loaded = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TemplateRuntimeError("TEMPLATE_CONFIG_INVALID", "模板配置不是有效 UTF-8 YAML") from exc
    if not isinstance(loaded, dict):
        raise TemplateRuntimeError("TEMPLATE_CONFIG_INVALID", "模板配置根节点必须是对象")
    try:
        return model_type.model_validate(loaded)
    except ValidationError as exc:
        raise TemplateRuntimeError("TEMPLATE_CONFIG_INVALID", "模板配置不符合运行时契约") from exc


def assert_agent_contract(
    agent: CapabilityProvider,
    *,
    expected_identity: AgentIdentity,
    required_features: set[str] | None = None,
) -> AgentCapabilities:
    """断言模板 Agent 的 capabilities 与 descriptor/build 身份一致。

    此函数用于生成 Agent 的离线契约测试，不发起 HTTP、模型或数据服务请求。
    """
    capabilities = agent.capabilities()
    if capabilities.internal_protocol_version != INTERNAL_PROTOCOL_VERSION:
        raise TemplateRuntimeError("CAPABILITIES_MISMATCH", "模板 Agent 的 internal 协议版本不匹配")
    if capabilities.identity != expected_identity:
        raise TemplateRuntimeError("CAPABILITIES_MISMATCH", "模板 Agent 的身份或源码校验和不匹配")
    missing_features = (required_features or set()) - set(capabilities.features)
    if missing_features:
        raise TemplateRuntimeError(
            "CAPABILITIES_MISMATCH",
            f"模板 Agent 缺少必要能力：{', '.join(sorted(missing_features))}",
        )
    return capabilities
