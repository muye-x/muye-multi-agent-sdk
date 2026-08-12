"""模型与外部系统集成的公共接口。"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .factory import build_chat_model

if TYPE_CHECKING:
    from .internal_agent import InternalAgentClient, InternalAgentClientError
    from .muye_data import DataClient, DataClientError


def __getattr__(name: str) -> Any:
    """仅在使用 internal client 时加载其 HTTP 实现。"""
    module_name = None
    if name in {"InternalAgentClient", "InternalAgentClientError"}:
        module_name = ".internal_agent"
    elif name in {"DataClient", "DataClientError"}:
        module_name = ".muye_data"
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if exc.name == "httpx" or (exc.name or "").startswith("httpx."):
            raise ImportError(
                f"{name} 的核心依赖 httpx 不可用，请重新安装 muye-multi-agent-sdk"
            ) from exc
        raise
    client_type = getattr(module, name)
    globals()[name] = client_type
    return client_type


__all__ = [
    "DataClient",
    "DataClientError",
    "InternalAgentClient",
    "InternalAgentClientError",
    "build_chat_model",
]
from .channel import ChannelAgentClient, ChannelAgentClientError

__all__ += ["ChannelAgentClient", "ChannelAgentClientError"]
