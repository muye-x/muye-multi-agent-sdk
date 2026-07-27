"""可信内网 muye-data 服务的异步只读客户端。"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ...config import DataConfig
from .errors import DataClientError
from .models import (
    ErrorResponse,
    FilterExpression,
    ResourceCapabilities,
    RetrievalRequest,
    RetrievalResponse,
)

logger = logging.getLogger(__name__)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class DataClient:
    """复用 HTTP 连接池调用 muye-data，不接受数据库地址或物理表名。"""

    _RETRYABLE_STATUS_CODES = {408, 429, 502, 503, 504}

    def __init__(
        self,
        config: DataConfig | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or DataConfig()
        self._client = http_client
        self._owns_client = http_client is None

    async def __aenter__(self) -> "DataClient":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def retrieve(
        self,
        *,
        resource: str,
        query: str,
        top_k: int = 5,
        pipeline: str | None = None,
        filter: FilterExpression | dict[str, Any] | None = None,
        return_fields: list[str] | None = None,
        trace_id: str = "",
    ) -> RetrievalResponse:
        """执行完整召回；参数在发起网络请求前按 v1 契约校验。"""
        request = RetrievalRequest.model_validate(
            {
                "resource": resource,
                "query": query,
                "top_k": top_k,
                "pipeline": pipeline,
                "filter": filter,
                "return_fields": return_fields,
                "trace_id": trace_id,
            }
        )
        payload = await self._request(
            "POST",
            "/api/v1/retrieve",
            json=request.model_dump(exclude_none=True),
            trace_id=request.trace_id,
        )
        return self._parse_data(payload, RetrievalResponse, trace_id=request.trace_id)

    async def capabilities(self, resource: str, *, trace_id: str = "") -> ResourceCapabilities:
        """读取已知资源能力；resource 仍按逻辑别名校验。"""
        validated = RetrievalRequest(resource=resource, query="capabilities", trace_id=trace_id)
        payload = await self._request(
            "GET",
            f"/api/v1/resources/{validated.resource}/capabilities",
            trace_id=validated.trace_id,
        )
        return self._parse_data(payload, ResourceCapabilities, trace_id=trace_id)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> dict[str, Any]:
        client = self._http_client()
        last_error: DataClientError | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                headers = {"Accept": "application/json"}
                if trace_id:
                    headers["X-Trace-Id"] = trace_id
                response = await client.request(
                    method,
                    path,
                    json=json,
                    headers=headers,
                )
                payload = self._json_object(response, trace_id=trace_id)
                if response.is_success:
                    return payload
                error = self._service_error(
                    response,
                    payload,
                    request_trace_id=trace_id,
                )
                if (
                    response.status_code not in self._RETRYABLE_STATUS_CODES
                    or not error.recoverable
                ):
                    raise error
                last_error = error
            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException as exc:
                last_error = DataClientError(
                    "DATA_TIMEOUT",
                    "muye-data 调用超时",
                    recoverable=True,
                    status_code=504,
                    trace_id=trace_id,
                )
                last_error.__cause__ = exc
            except httpx.HTTPError as exc:
                last_error = DataClientError(
                    "DATA_UNAVAILABLE",
                    "muye-data 不可用",
                    recoverable=True,
                    status_code=503,
                    trace_id=trace_id,
                )
                last_error.__cause__ = exc
            if attempt < self.config.max_retries:
                await asyncio.sleep(min(0.1 * (2**attempt), 1.0))

        assert last_error is not None
        logger.warning(
            "muye-data request failed trace_id=%s path=%s code=%s recoverable=%s",
            trace_id,
            path,
            last_error.code,
            last_error.recoverable,
        )
        raise last_error

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                trust_env=False,
            )
        return self._client

    @staticmethod
    def _json_object(response: httpx.Response, *, trace_id: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise DataClientError(
                "DATA_PROTOCOL_ERROR",
                "muye-data 返回非 JSON 响应",
                recoverable=False,
                status_code=response.status_code,
                trace_id=trace_id,
            ) from exc
        if not isinstance(payload, dict):
            raise DataClientError(
                "DATA_PROTOCOL_ERROR",
                "muye-data 返回协议无效",
                recoverable=False,
                status_code=response.status_code,
                trace_id=trace_id,
            )
        return payload

    @staticmethod
    def _service_error(
        response: httpx.Response,
        payload: dict[str, Any],
        *,
        request_trace_id: str,
    ) -> DataClientError:
        try:
            error = ErrorResponse.model_validate(payload)
        except ValidationError:
            return DataClientError(
                "DATA_PROTOCOL_ERROR",
                "muye-data 错误响应协议无效",
                recoverable=False,
                status_code=response.status_code,
                trace_id=request_trace_id,
            )
        return DataClientError(
            error.error_code,
            error.message,
            recoverable=error.recoverable,
            status_code=response.status_code,
            trace_id=error.trace_id or request_trace_id,
        )

    @staticmethod
    def _parse_data(
        payload: dict[str, Any],
        model_type: type[ResponseModel],
        *,
        trace_id: str,
    ) -> ResponseModel:
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise DataClientError(
                "DATA_PROTOCOL_ERROR",
                "muye-data 返回协议无效",
                recoverable=False,
                trace_id=trace_id,
            ) from exc

    async def aclose(self) -> None:
        """仅关闭由本客户端创建的 HTTP 连接池，允许幂等调用。"""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
