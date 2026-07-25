"""Muye Multi-Agent SDK 的公共接口。"""

from .config import AgentConfig, ApiConfig, ContextConfig, IntentGuardConfig, ModelConfig
from .contracts import (
    AgentCapabilities,
    AgentContext,
    AgentError,
    AgentEvent,
    AgentMetadata,
    AgentRequest,
    AgentResult,
    CancelRequest,
    CancelResponse,
    ToolCapability,
)
from .modes import BaseAgent, CustomAgent, GraphAgent, ReActAgent
from .transport import create_app
from .version import INTERNAL_PROTOCOL_VERSION, PUBLIC_PROTOCOL_VERSION, SDK_VERSION

__version__ = SDK_VERSION

__all__ = [
    "AgentCapabilities", "AgentConfig", "AgentContext", "AgentError", "AgentEvent", "AgentMetadata",
    "AgentRequest", "AgentResult", "ApiConfig", "BaseAgent", "CancelRequest", "CancelResponse",
    "ContextConfig", "CustomAgent", "GraphAgent", "INTERNAL_PROTOCOL_VERSION", "IntentGuardConfig",
    "ModelConfig", "PUBLIC_PROTOCOL_VERSION", "ReActAgent",
    "SDK_VERSION", "ToolCapability", "create_app",
]
