"""固定作用域 muye-data LangChain 工具测试。"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from muye_multi_agent_sdk.integrations import DataClientError
from muye_multi_agent_sdk.integrations.muye_data import RetrievalHit, RetrievalResponse
from muye_multi_agent_sdk.tools import create_data_retrieval_tool


class _FakeClient:
    def __init__(self, result: RetrievalResponse | DataClientError) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, **kwargs: Any) -> RetrievalResponse:
        self.calls.append(kwargs)
        if isinstance(self.result, DataClientError):
            raise self.result
        return self.result


def _response(content: str = "reference") -> RetrievalResponse:
    return RetrievalResponse(
        resource="docs",
        pipeline="hybrid",
        trace_id="trace-1",
        took_ms=2,
        partial=False,
        warnings=[],
        hits=[RetrievalHit(id="1", content=content, score=0.9, fields={"title": "A"})],
    )


def test_tool_exposes_only_query_and_preserves_fixed_scope() -> None:
    client = _FakeClient(_response())
    tool = create_data_retrieval_tool(
        client,  # type: ignore[arg-type]
        resource="docs",
        pipeline="hybrid",
        fixed_filter={"op": "eq", "field": "tenant_id", "value": "t1"},
        return_fields=["title"],
    )

    result = asyncio.run(tool.ainvoke({"query": "refund"}))

    assert set(tool.args) == {"query"}
    assert client.calls[0]["resource"] == "docs"
    assert client.calls[0]["pipeline"] == "hybrid"
    assert client.calls[0]["filter"].field == "tenant_id"
    assert result["content_trust"] == "untrusted_reference"


def test_tool_returns_structured_safe_error() -> None:
    client = _FakeClient(
        DataClientError("DATA_TIMEOUT", "timeout", recoverable=True, trace_id="t1")
    )
    tool = create_data_retrieval_tool(client, resource="docs")  # type: ignore[arg-type]

    result = asyncio.run(tool.ainvoke({"query": "refund"}))

    assert result == {
        "ok": False,
        "error": {"code": "DATA_TIMEOUT", "recoverable": True},
        "trace_id": "t1",
    }


def test_tool_truncates_large_content() -> None:
    client = _FakeClient(_response("x" * 2_000))
    tool = create_data_retrieval_tool(
        client,  # type: ignore[arg-type]
        resource="docs",
        max_output_chars=256,
    )

    result = asyncio.run(tool.ainvoke({"query": "refund"}))

    assert result["truncated"] is True
    assert len(result["hits"][0]["content"]) < 2_000


def test_tool_budget_uses_serialized_json_length() -> None:
    client = _FakeClient(_response('"' * 2_000))
    tool = create_data_retrieval_tool(
        client,  # type: ignore[arg-type]
        resource="docs",
        max_output_chars=256,
    )

    result = asyncio.run(tool.ainvoke({"query": "refund"}))

    serialized_hits = json.dumps(result["hits"], ensure_ascii=False, separators=(",", ":"))
    assert result["truncated"] is True
    assert len(serialized_hits) <= 256


def test_tool_rejects_whitespace_query_before_client_call() -> None:
    client = _FakeClient(_response())
    tool = create_data_retrieval_tool(client, resource="docs")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        asyncio.run(tool.ainvoke({"query": "   "}))

    assert client.calls == []


def test_tool_validates_fixed_scope_when_created() -> None:
    client = _FakeClient(_response())

    with pytest.raises(ValidationError):
        create_data_retrieval_tool(client, resource="invalid resource")  # type: ignore[arg-type]
