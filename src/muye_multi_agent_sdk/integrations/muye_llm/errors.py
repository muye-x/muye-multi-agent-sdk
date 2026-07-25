"""muye-llm 适配器异常。"""


class MuyeLlmError(RuntimeError):
    """muye-llm 网关调用、响应解析或流式事件处理失败。"""
