"""muye-llm 的 LangChain 适配器。"""

from .chat_model import MuyeLlmChatModel
from .errors import MuyeLlmError

__all__ = ["MuyeLlmChatModel", "MuyeLlmError"]
