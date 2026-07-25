"""public profile 的安全展示字段投影。"""
from __future__ import annotations

from typing import Any

_MARKDOWN_FIELDS = ("markdown", "summary", "answer", "result")


def public_markdown(result_data: dict[str, Any] | None) -> str | None:
    """返回首个非空展示字段；缺失时由调用方投影为统一错误。"""
    for field in _MARKDOWN_FIELDS:
        value = (result_data or {}).get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
