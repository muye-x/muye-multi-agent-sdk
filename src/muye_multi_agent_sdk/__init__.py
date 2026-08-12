"""Muye Multi-Agent SDK 的公共接口。"""

from .config import AgentConfig, ApiConfig, ContextConfig, DataConfig, IntentGuardConfig, ModelConfig
from .contracts import (
    AgentCapabilities,
    AgentContext,
    AgentError,
    AgentEvent,
    AgentIdentity,
    AgentMetadata,
    AgentRequest,
    AgentResult,
    CancelRequest,
    CancelResponse,
    CitationBlock,
    ChannelInvokeRequest,
    ChannelInvokeResponse,
    ChannelTextMessage,
    ToolCapability,
)
from .modes import BaseAgent, CustomAgent, GraphAgent, ReActAgent
from .template_support import TemplateRuntimeError, assert_agent_contract, load_yaml_config
from .integrations.channel import ChannelAgentClient, ChannelAgentClientError
from .transport import create_app
from .version import CHANNEL_PROTOCOL_VERSION, INTERNAL_PROTOCOL_VERSION, PUBLIC_PROTOCOL_VERSION, SDK_VERSION

__version__ = SDK_VERSION

__all__ = [
    "AgentCapabilities", "AgentConfig", "AgentContext", "AgentError", "AgentEvent", "AgentIdentity",
    "AgentMetadata", "AgentRequest", "AgentResult", "ApiConfig", "BaseAgent", "CancelRequest", "CancelResponse",
    "CitationBlock", "ChannelAgentClient", "ChannelAgentClientError", "ChannelInvokeRequest", "ChannelInvokeResponse", "ChannelTextMessage",
    "ContextConfig", "CustomAgent", "DataConfig", "GraphAgent", "INTERNAL_PROTOCOL_VERSION", "IntentGuardConfig",
    "ModelConfig", "PUBLIC_PROTOCOL_VERSION", "ReActAgent",
    "SDK_VERSION", "CHANNEL_PROTOCOL_VERSION", "ToolCapability", "create_app",
    "TemplateRuntimeError", "assert_agent_contract", "load_yaml_config",
]
