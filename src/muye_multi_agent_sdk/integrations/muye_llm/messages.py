"""LangChain message、tool 和结构化输出转换 helper。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool



def content_to_text(content: Any) -> str:
    """将 LangChain content block 转换为 muye-llm 可接收的纯文本。"""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def to_message_payload(message: BaseMessage) -> dict[str, Any]:
    """将 LangChain message 转换为 muye-llm/OpenAI 兼容结构。"""

    content = content_to_text(message.content)
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": content}
    if isinstance(message, AIMessage):
        payload: dict[str, Any] = {"role": "assistant", "content": content or ""}
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", ""),
                        "arguments": json.dumps(
                            tool_call.get("args", {}),
                            ensure_ascii=False,
                        ),
                    },
                }
                for tool_call in tool_calls
                if isinstance(tool_call, dict)
            ]
            payload["content"] = None
        return payload
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": message.tool_call_id,
        }
    return {
        "role": getattr(message, "type", "user"),
        "content": content,
    }


def parse_tool_calls(raw: str) -> list[dict[str, Any]] | None:
    """解析 muye-llm 返回的文本形式 tool calls。"""

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict) and isinstance(data.get("tool_calls"), list):
        return data["tool_calls"]
    return None


def tool_calls_to_langchain(raw_tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 OpenAI function-calling payload 转换为 LangChain tool calls。"""

    converted: list[dict[str, Any]] = []
    for tool_call in raw_tool_calls:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            args = {"raw": arguments}
        converted.append(
            {
                "name": function.get("name", ""),
                "args": args if isinstance(args, dict) else {"value": args},
                "id": tool_call.get("id", ""),
                "type": "tool_call",
            }
        )
    return converted


def tool_to_openai(tool: BaseTool | dict[str, Any]) -> dict[str, Any] | None:
    """将 LangChain tool 转换为 OpenAI function-calling tool 定义。"""

    if isinstance(tool, dict):
        return tool
    if hasattr(tool, "as_openai_tool"):
        return tool.as_openai_tool()
    if hasattr(tool, "tool_call_schema"):
        schema = tool.tool_call_schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        }
    return None


def extract_json_payload(raw: str) -> Any:
    """从模型文本中提取 JSON object 或 array。"""

    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        raise ValueError("模型未返回 JSON 内容")
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return json.loads(text[start : idx + 1])
    raise ValueError("模型返回的 JSON 内容不完整")
