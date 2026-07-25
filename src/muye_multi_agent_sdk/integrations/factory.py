"""内置模型工厂和外部 ChatModel 注入边界。"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import ModelConfig


def build_chat_model(config: ModelConfig) -> BaseChatModel:
    """按内置 provider 配置创建 LangChain ChatModel。"""
    if config.provider == "muye":
        try:
            from .muye_llm import MuyeLlmChatModel
        except ModuleNotFoundError as exc:
            if exc.name == "httpx" or (exc.name or "").startswith("httpx."):
                raise ImportError(
                    "muye provider 的核心依赖 httpx 不可用，请重新安装 muye-multi-agent-sdk"
                ) from exc
            raise
        return MuyeLlmChatModel(
            base_url=config.base_url,
            model_name=config.model,
            enable_thinking=config.enable_thinking,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
        )
    if config.provider == "openai_compatible":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI-compatible provider 的核心依赖 langchain-openai 不可用，"
                "请重新安装 muye-multi-agent-sdk"
            ) from exc
        return ChatOpenAI(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
        )
    raise ValueError(f"不支持的内置模型 provider: {config.provider}")
