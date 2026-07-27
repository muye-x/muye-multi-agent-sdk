"""muye-data SDK 客户端的稳定错误类型。"""
from __future__ import annotations


class DataClientError(RuntimeError):
    """网络、服务错误或响应协议不兼容。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        recoverable: bool,
        status_code: int | None = None,
        trace_id: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable
        self.status_code = status_code
        self.trace_id = trace_id

