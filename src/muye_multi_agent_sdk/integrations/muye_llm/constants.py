"""muye-llm HTTP 协议常量。"""


class MuyeLlmApi:
    """muye-llm 网关端点。"""

    CHAT_PATH = "/api/v2/chat"
    STREAM_PATH = "/api/v2/chat/stream"


class MuyeLlmStreamEventName:
    """muye-llm SSE 事件名称。"""

    TOKEN = "token"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"
    DONE = "done"

__all__ = ["MuyeLlmApi", "MuyeLlmStreamEventName"]
