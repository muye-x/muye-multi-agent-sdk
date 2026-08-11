"""muye-data SDK 客户端和 Agent 生命周期契约测试。"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from muye_multi_agent_sdk import (
    AgentConfig,
    AgentMetadata,
    AgentRequest,
    AgentResult,
    CustomAgent,
    DataConfig,
)
from muye_multi_agent_sdk.integrations import DataClient, DataClientError
from muye_multi_agent_sdk.integrations import muye_data as muye_data_integration
from muye_multi_agent_sdk.integrations.muye_data import DataAccessContext, RetrievalRequest
from muye_multi_agent_sdk import AgentIdentity


def _client(handler: Any, *, max_retries: int = 0) -> tuple[DataClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://data.test",
    )
    return (
        DataClient(
            DataConfig(base_url="http://data.test", max_retries=max_retries),
            http_client=http_client,
        ),
        http_client,
    )


def test_retrieve_serializes_database_independent_contract() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(
            200,
            json={
                "resource": "docs",
                "pipeline": "hybrid",
                "trace_id": "trace-1",
                "took_ms": 12,
                "partial": False,
                "warnings": [],
                "hits": [
                    {"id": "1", "content": "answer", "score": 0.8, "fields": {"title": "A"}}
                ],
            },
        )

    client, http_client = _client(handler)
    result = asyncio.run(
        client.retrieve(
            resource="docs",
            query="refund",
            pipeline="hybrid",
            filter={"op": "eq", "field": "tenant_id", "value": "t1"},
            return_fields=["title"],
            trace_id="trace-1",
        )
    )

    assert result.hits[0].id == "1"
    assert captured["filter"] == {"op": "eq", "field": "tenant_id", "value": "t1"}
    assert "backend" not in captured
    assert "collection" not in captured
    asyncio.run(client.aclose())
    assert not http_client.is_closed
    asyncio.run(http_client.aclose())


def test_capabilities_rejects_invalid_response() -> None:
    client, http_client = _client(lambda _request: httpx.Response(200, json=[]))

    with pytest.raises(DataClientError, match="协议无效") as exc_info:
        asyncio.run(client.capabilities("docs"))

    assert exc_info.value.code == "DATA_PROTOCOL_ERROR"
    asyncio.run(http_client.aclose())


def test_service_error_preserves_safe_code_and_trace() -> None:
    client, http_client = _client(
        lambda _request: httpx.Response(
            404,
            json={
                "error_code": "RESOURCE_NOT_FOUND",
                "message": "资源不存在",
                "recoverable": False,
                "trace_id": "t1",
            },
        )
    )

    with pytest.raises(DataClientError) as exc_info:
        asyncio.run(client.retrieve(resource="missing", query="q"))

    assert exc_info.value.code == "RESOURCE_NOT_FOUND"
    assert exc_info.value.trace_id == "t1"
    assert not exc_info.value.recoverable
    asyncio.run(http_client.aclose())


def test_retry_is_bounded_to_configured_read_only_attempt() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                503,
                json={
                    "error_code": "DATA_UNAVAILABLE",
                    "message": "暂不可用",
                    "recoverable": True,
                    "trace_id": "",
                },
            )
        return httpx.Response(
            200,
            json={
                "resource": "docs",
                "pipeline": "dense",
                "trace_id": "",
                "took_ms": 1,
                "partial": False,
                "warnings": [],
                "hits": [],
            },
        )

    client, http_client = _client(handler, max_retries=1)
    result = asyncio.run(client.retrieve(resource="docs", query="q"))

    assert result.hits == []
    assert attempts == 2
    asyncio.run(http_client.aclose())


def test_nonrecoverable_service_error_is_not_retried() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            502,
            json={
                "error_code": "BACKEND_PROTOCOL_ERROR",
                "message": "数据库响应无效",
                "recoverable": False,
                "trace_id": "trace-1",
            },
        )

    client, http_client = _client(handler, max_retries=1)

    with pytest.raises(DataClientError) as exc_info:
        asyncio.run(client.retrieve(resource="docs", query="q", trace_id="trace-1"))

    assert exc_info.value.code == "BACKEND_PROTOCOL_ERROR"
    assert not exc_info.value.recoverable
    assert attempts == 1
    asyncio.run(http_client.aclose())


def test_invalid_error_response_is_protocol_error_without_boolean_coercion() -> None:
    client, http_client = _client(
        lambda _request: httpx.Response(
            503,
            json={
                "error_code": "DATA_UNAVAILABLE",
                "message": "暂不可用",
                "recoverable": "false",
                "trace_id": "trace-1",
            },
        ),
        max_retries=1,
    )

    with pytest.raises(DataClientError) as exc_info:
        asyncio.run(client.retrieve(resource="docs", query="q", trace_id="trace-1"))

    assert exc_info.value.code == "DATA_PROTOCOL_ERROR"
    assert not exc_info.value.recoverable
    asyncio.run(http_client.aclose())


def test_capabilities_sends_trace_header_and_timeout_preserves_trace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Trace-Id"] == "trace-1"
        return httpx.Response(
            200,
            json={
                "resource": "docs",
                "default_pipeline": "dense",
                "pipelines": [{"name": "dense", "type": "dense", "rerank": False}],
                "returnable_fields": [],
                "filterable_fields": [],
                "filter_operators": [],
                "max_top_k": 100,
            },
        )

    client, http_client = _client(handler)
    result = asyncio.run(client.capabilities("docs", trace_id="trace-1"))
    assert result.resource == "docs"
    asyncio.run(http_client.aclose())

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    timeout_client, timeout_http_client = _client(timeout_handler)
    with pytest.raises(DataClientError) as exc_info:
        asyncio.run(
            timeout_client.retrieve(resource="docs", query="q", trace_id="trace-timeout")
        )
    assert exc_info.value.trace_id == "trace-timeout"
    asyncio.run(timeout_http_client.aclose())


def test_retrieve_forwards_only_trusted_access_context_headers() -> None:
    access_context = DataAccessContext(
        service_id="agent-service",
        deployment_id="deploy-handbook-v1",
        agent=AgentIdentity(
            agent_id="agent_product_handbook",
            agent_version="1.0.0",
            descriptor_checksum="a" * 64,
            source_tree_checksum="b" * 64,
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Muye-Service-Id"] == "agent-service"
        assert request.headers["X-Muye-Deployment-Id"] == "deploy-handbook-v1"
        assert request.headers["X-Muye-Agent-Id"] == "agent_product_handbook"
        assert request.headers["X-Muye-Agent-Version"] == "1.0.0"
        assert request.headers["X-Muye-Descriptor-Checksum"] == "a" * 64
        assert request.headers["X-Muye-Source-Checksum"] == "b" * 64
        assert "access_context" not in __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "resource": "docs",
                "pipeline": "hybrid",
                "trace_id": "",
                "took_ms": 1,
                "partial": False,
                "warnings": [],
                "hits": [],
            },
        )

    client, http_client = _client(handler)
    asyncio.run(client.retrieve(resource="docs", query="refund", access_context=access_context))
    asyncio.run(http_client.aclose())


def test_sdk_filter_contract_matches_service_budgets() -> None:
    invalid_filters: list[dict[str, Any]] = [
        {"op": "eq", "field": "score", "value": float("nan")},
        {"op": "eq", "field": "text", "value": "x" * 4097},
        {
            "op": "and",
            "conditions": [
                {"op": "eq", "field": "category", "value": index}
                for index in range(50)
            ],
        },
    ]
    deep_filter: dict[str, Any] = {"op": "eq", "field": "category", "value": "a"}
    for _ in range(8):
        deep_filter = {"op": "not", "condition": deep_filter}
    invalid_filters.append(deep_filter)

    for invalid_filter in invalid_filters:
        with pytest.raises(ValidationError):
            RetrievalRequest(resource="docs", query="q", filter=invalid_filter)


@pytest.mark.parametrize("base_url", ["http://", "http://user:password@data.test"])
def test_data_config_rejects_invalid_or_credentialed_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        DataConfig(base_url=base_url)


def test_data_config_rejects_non_finite_timeout() -> None:
    with pytest.raises(ValidationError):
        DataConfig(timeout_seconds=float("inf"))


def test_data_client_propagates_cancellation() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    client, http_client = _client(handler)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(client.retrieve(resource="docs", query="q"))

    asyncio.run(http_client.aclose())


class _LifecycleAgent(CustomAgent):
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="data-test", version="1", description="test")

    async def execute(self, request: AgentRequest, **_: Any) -> AgentResult:
        return AgentResult.success({"markdown": request.task})


class _CloseableDataClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_agent_does_not_close_injected_data_client() -> None:
    injected = _CloseableDataClient()
    agent = _LifecycleAgent(data_client=injected)  # type: ignore[arg-type]

    asyncio.run(agent.aclose())

    assert not injected.closed


def test_agent_closes_sdk_owned_data_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[_CloseableDataClient] = []

    def build_client(_config: DataConfig) -> _CloseableDataClient:
        client = _CloseableDataClient()
        created.append(client)
        return client

    monkeypatch.setattr(muye_data_integration, "DataClient", build_client)
    agent = _LifecycleAgent(config=AgentConfig())

    assert agent.data_client is created[0]
    asyncio.run(agent.aclose())

    assert created[0].closed
