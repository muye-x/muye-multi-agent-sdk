"""muye-data v1 的数据库无关请求与响应契约。"""
from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RESOURCE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
TRACE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
FIELD_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$"
MAX_FILTER_DEPTH = 8
MAX_FILTER_CONDITIONS = 50
MAX_FILTER_SET_VALUES = 100
FilterScalar = str | int | float | bool


class StrictModel(BaseModel):
    """拒绝未知字段与边界处的隐式类型转换。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class FilterExpression(StrictModel):
    """跨数据库结构化过滤 AST，禁止原生查询字符串。"""

    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "and", "or", "not"]
    field: str | None = Field(default=None, pattern=FIELD_NAME_PATTERN)
    value: FilterScalar | None = None
    values: list[FilterScalar] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_FILTER_SET_VALUES,
    )
    conditions: list["FilterExpression"] | None = Field(default=None, min_length=1)
    condition: "FilterExpression | None" = None

    @model_validator(mode="after")
    def validate_shape(self) -> "FilterExpression":
        comparison = self.op in {"eq", "ne", "gt", "gte", "lt", "lte"}
        membership = self.op in {"in", "not_in"}
        logical = self.op in {"and", "or"}
        if comparison:
            valid = self.field is not None and self.value is not None
            valid = valid and self.values is None and self.conditions is None and self.condition is None
        elif membership:
            valid = self.field is not None and self.values is not None
            valid = valid and self.value is None and self.conditions is None and self.condition is None
        elif logical:
            valid = self.conditions is not None
            valid = (
                valid
                and self.field is None
                and self.value is None
                and self.values is None
                and self.condition is None
            )
        else:
            valid = self.condition is not None
            valid = (
                valid
                and self.field is None
                and self.value is None
                and self.values is None
                and self.conditions is None
            )
        if not valid:
            raise ValueError(f"过滤操作符 {self.op!r} 的字段组合无效")
        scalar_values = [self.value] if self.value is not None else list(self.values or [])
        for scalar in scalar_values:
            if isinstance(scalar, float) and not math.isfinite(scalar):
                raise ValueError("filter 数值必须是有限值")
            if isinstance(scalar, str) and len(scalar) > 4096:
                raise ValueError("filter 字符串不能超过 4096 字符")
        return self


def _filter_size(expression: FilterExpression, depth: int = 1) -> tuple[int, int]:
    """返回过滤 AST 的最大深度和总节点数。"""
    children = list(expression.conditions or [])
    if expression.condition is not None:
        children.append(expression.condition)
    if not children:
        return depth, 1
    child_stats = [_filter_size(child, depth + 1) for child in children]
    return max(item[0] for item in child_stats), 1 + sum(item[1] for item in child_stats)


class RetrievalRequest(StrictModel):
    """SDK 发往 muye-data 的完整只读召回请求。"""

    resource: str = Field(pattern=RESOURCE_NAME_PATTERN)
    query: str = Field(min_length=1, max_length=8_000)
    top_k: int = Field(default=5, ge=1, le=100)
    pipeline: str | None = Field(default=None, pattern=RESOURCE_NAME_PATTERN)
    filter: FilterExpression | None = None
    return_fields: list[str] | None = Field(default=None, max_length=50)
    trace_id: str = Field(default="", max_length=128)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query 不能为空")
        return normalized

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        normalized = value.strip()
        if normalized and re.fullmatch(TRACE_ID_PATTERN, normalized) is None:
            raise ValueError("trace_id 格式无效")
        return normalized

    @field_validator("return_fields")
    @classmethod
    def normalize_return_fields(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if len(value) != len(set(value)):
            raise ValueError("return_fields 不能重复")
        if any(re.fullmatch(FIELD_NAME_PATTERN, field) is None for field in value):
            raise ValueError("return_fields 包含非法字段名")
        return value

    @model_validator(mode="after")
    def validate_filter_budget(self) -> "RetrievalRequest":
        """与 muye-data 同步限制递归深度和总节点数。"""
        if self.filter is None:
            return self
        depth, count = _filter_size(self.filter)
        if depth > MAX_FILTER_DEPTH:
            raise ValueError(f"filter 最大嵌套深度为 {MAX_FILTER_DEPTH}")
        if count > MAX_FILTER_CONDITIONS:
            raise ValueError(f"filter 最多包含 {MAX_FILTER_CONDITIONS} 个节点")
        return self


class RetrievalHit(StrictModel):
    """数据库无关命中；score 只保证当前响应内越高越相关。"""

    id: str
    content: str
    score: float = Field(allow_inf_nan=False)
    fields: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(StrictModel):
    """一次召回的标准结果，不包含 HTTP 信封。"""

    resource: str
    pipeline: str
    trace_id: str
    took_ms: int = Field(ge=0)
    partial: bool
    warnings: list[str]
    hits: list[RetrievalHit]


class PipelineCapability(StrictModel):
    """单个命名 pipeline 的公开模式与重排能力。"""

    name: str
    type: Literal["dense", "keyword", "hybrid"]
    rerank: bool


class ResourceCapabilities(StrictModel):
    """已知资源的安全公开能力，不暴露物理库表与数据库连接。"""

    resource: str
    default_pipeline: str
    pipelines: list[PipelineCapability]
    returnable_fields: list[str]
    filterable_fields: list[str]
    filter_operators: list[str]
    max_top_k: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_default_pipeline(self) -> "ResourceCapabilities":
        if self.default_pipeline not in {pipeline.name for pipeline in self.pipelines}:
            raise ValueError("default_pipeline 必须存在于 pipelines")
        return self


class ErrorResponse(StrictModel):
    """muye-data 的稳定错误响应。"""

    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    recoverable: bool
    trace_id: str


FilterExpression.model_rebuild()
