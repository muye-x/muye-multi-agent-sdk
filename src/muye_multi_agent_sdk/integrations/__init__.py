"""模型与外部系统集成的公共接口。"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .factory import build_chat_model

if TYPE_CHECKING:
    from .internal_agent import InternalAgentClient, InternalAgentClientError


def __getattr__(name: str) -> Any:
    """仅在使用 internal client 时加载其 HTTP 实现。"""
    if name not in {"InternalAgentClient", "InternalAgentClientError"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(".internal_agent", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == "httpx" or (exc.name or "").startswith("httpx."):
            raise ImportError(
                "InternalAgentClient 的核心依赖 httpx 不可用，请重新安装 muye-multi-agent-sdk"
            ) from exc
        raise
    client_type = getattr(module, name)
    globals()[name] = client_type
    return client_type


__all__ = ["InternalAgentClient", "InternalAgentClientError", "build_chat_model"]
