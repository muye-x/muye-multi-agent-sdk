"""三种 Agent 模式共享的生命周期实现。"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ..config import AgentConfig
from ..contracts import AgentCapabilities, AgentEvent, AgentMetadata, AgentRequest, AgentResult, CancelResponse, ToolCapability
from ..runtime import CheckpointerManager, ExecutionManager, ExecutionOptions, SessionBusyError
from ..runtime.execution import build_checkpoint_thread_id, build_session_identity
from ..safety import GuardContext, IntentCategory, IntentGuard
from ..version import INTERNAL_PROTOCOL_VERSION, PUBLIC_PROTOCOL_VERSION

logger = logging.getLogger(__name__)

# 按需加载
if TYPE_CHECKING:
    from ..integrations.muye_data import DataClient


class BaseAgent(ABC):
    """统一处理守卫、锁、取消、超时、上下文和终态结果的基础类。"""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        data_client: DataClient | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self._executions = ExecutionManager()
        self._checkpointers = CheckpointerManager(self.config.context)
        self._data_client = data_client
        self._data_client_owned_by_sdk = False

    @property
    @abstractmethod
    def metadata(self) -> AgentMetadata:
        """返回稳定的 Agent 身份与意图声明。"""

    @property
    def tools(self) -> list[ToolCapability]:
        """返回实际可调用的工具能力，Graph 内部节点不在此列。"""
        return []

    @property
    def intent_guard_business_rules(self) -> str:
        """返回追加到固定守卫提示词后的可信业务规则，不能覆盖 SDK 安全约束。"""
        return ""

    def capabilities(self) -> AgentCapabilities:
        """根据公共 metadata 和配置生成唯一的能力声明。"""
        metadata = self.metadata
        profiles = sorted(self.config.api.profiles)
        return AgentCapabilities(
            agent_name=metadata.name,
            version=metadata.version,
            description=metadata.description,
            supported_intents=metadata.supported_intents,
            tools=self.tools,
            api_profiles=profiles,
            internal_protocol_version=INTERNAL_PROTOCOL_VERSION,
            public_protocol_version=PUBLIC_PROTOCOL_VERSION if "public" in profiles else None,
            identity=metadata.identity,
            features=self.features,
        )

    @property
    def features(self) -> list[str]:
        """返回可由 trusted caller 用于协议协商的稳定运行时能力。"""
        return ["cancel", "citation_blocks", "sse", "trusted_deadline"]

    async def invoke(
        self,
        request: AgentRequest,
        *,
        options: ExecutionOptions | None = None,
    ) -> AgentResult:
        """执行非流式请求；可以安全直接调用，无需 ContextVar 包装。"""
        options = options or ExecutionOptions()
        identity_error = self._context_identity_error(request, options)
        if identity_error is not None:
            return identity_error
        started_at = time.monotonic()
        timeout_seconds = self._execution_timeout_seconds(options, started_at)
        if timeout_seconds is None:
            return self._deadline_exceeded_result(request)
        try:
            async with asyncio.timeout(timeout_seconds):
                async with self._executions.acquire(self.metadata.name, request, options):
                    guarded = await self._guard(request, options)
                    if guarded is not None:
                        return guarded
                    return await self.execute(request, options=options)
        except SessionBusyError as exc:
            return AgentResult.interrupted("SESSION_BUSY", str(exc), recoverable=True, trace_id=request.context.trace_id)
        except asyncio.CancelledError:
            return AgentResult.interrupted("USER_CANCELLED", "当前任务已终止。", recoverable=True, trace_id=request.context.trace_id)
        except TimeoutError:
            return self._timeout_result(request, options)
        except Exception as exc:
            logger.exception(
                "Agent invoke failed agent=%s profile=%s trace_id=%s error_type=%s",
                self.metadata.name,
                options.profile,
                request.context.trace_id,
                type(exc).__name__,
            )
            return AgentResult.failure("UNKNOWN_ERROR", "Agent 执行失败。", recoverable=True, trace_id=request.context.trace_id)

    async def stream(
        self,
        request: AgentRequest,
        *,
        options: ExecutionOptions | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """执行流式请求，只产生无传输格式的 typed AgentEvent。"""
        options = options or ExecutionOptions()
        identity_error = self._context_identity_error(request, options)
        if identity_error is not None:
            yield AgentEvent.completed(identity_error)
            return
        started_at = time.monotonic()
        timeout_seconds = self._execution_timeout_seconds(options, started_at)
        if timeout_seconds is None:
            yield AgentEvent.completed(self._deadline_exceeded_result(request))
            return
        try:
            async with asyncio.timeout(timeout_seconds):
                async with self._executions.acquire(self.metadata.name, request, options):
                    guarded = await self._guard(request, options)
                    if guarded is not None:
                        yield AgentEvent.completed(guarded)
                        return
                    emitted_terminal = False
                    async for event in self.stream_events(request, options=options):
                        emitted_terminal = emitted_terminal or event.kind == "result"
                        yield event
                    if not emitted_terminal:
                        yield AgentEvent.completed(await self.execute(request, options=options))
        except SessionBusyError as exc:
            yield AgentEvent.completed(AgentResult.interrupted("SESSION_BUSY", str(exc), recoverable=True, trace_id=request.context.trace_id))
        except asyncio.CancelledError:
            yield AgentEvent.completed(AgentResult.interrupted("USER_CANCELLED", "当前任务已终止。", recoverable=True, trace_id=request.context.trace_id))
        except TimeoutError:
            yield AgentEvent.completed(self._timeout_result(request, options))
        except Exception as exc:
            logger.exception(
                "Agent stream failed agent=%s profile=%s trace_id=%s error_type=%s",
                self.metadata.name,
                options.profile,
                request.context.trace_id,
                type(exc).__name__,
            )
            yield AgentEvent.completed(AgentResult.failure("UNKNOWN_ERROR", "Agent 执行失败。", recoverable=True, trace_id=request.context.trace_id))

    @abstractmethod
    async def execute(self, request: AgentRequest, *, options: ExecutionOptions) -> AgentResult:
        """由模式实现具体执行，返回统一结果。"""

    async def stream_events(self, request: AgentRequest, *, options: ExecutionOptions) -> AsyncIterator[AgentEvent]:
        """默认把非流式结果转换为终态事件。"""
        yield AgentEvent.completed(await self.execute(request, options=options))

    async def cancel(
        self,
        *,
        user_id: str,
        session_id: str,
        profile: str,
        run_id: str | None = None,
    ) -> CancelResponse:
        """取消匹配 profile 的当前会话任务。"""
        return await self._executions.cancel(
            self.metadata.name,
            user_id=user_id,
            session_id=session_id,
            profile=profile,
            run_id=run_id,
        )

    async def checkpointer(self, options: ExecutionOptions) -> Any | None:
        """返回 profile 明确启用时的 LangGraph checkpointer。"""
        return await self._checkpointers.get(enabled=self._context_enabled(options))

    @property
    def data_client(self) -> "DataClient":
        """按需返回共享数据客户端；首次访问时才创建连接资源。"""
        if self._data_client is None:
            from ..integrations.muye_data import DataClient

            self._data_client = DataClient(self.config.data)
            self._data_client_owned_by_sdk = True
        return self._data_client

    async def aclose(self) -> None:
        """释放 SDK 托管的上下文与按需数据客户端。"""
        resources = [self._checkpointers.close()]
        if self._data_client_owned_by_sdk and self._data_client is not None:
            resources.append(self._data_client.aclose())
        await asyncio.gather(*resources)
        self._data_client = None
        self._data_client_owned_by_sdk = False

    async def _guard(self, request: AgentRequest, options: ExecutionOptions) -> AgentResult | None:
        if not self.config.intent_guard.enabled:
            return None
        model = await self.guard_model()
        if model is None:
            return None
        guard_context = GuardContext()
        if self.config.intent_guard.context_aware and self._context_enabled(options):
            guard_context = await self._intent_guard_context(request, options)
        decision = await IntentGuard(model, self.config.intent_guard.timeout_seconds).classify(
            request.task,
            self.metadata.supported_intents,
            history=guard_context.history,
            meaningful_definition=self.config.intent_guard.meaningful_definition,
            meaningless_definition=self.config.intent_guard.meaningless_definition,
            business_rules=self.intent_guard_business_rules,
            trace_id=request.context.trace_id,
        )
        if decision.category is IntentCategory.MEANINGLESS:
            if guard_context.read_failed:
                logger.warning(
                    "Intent guard history unavailable; allowing meaningless decision agent=%s profile=%s trace_id=%s",
                    self.metadata.name,
                    options.profile,
                    request.context.trace_id,
                )
                return None
            question = decision.user_message or "请补充更具体的任务信息。"
            if self.config.intent_guard.context_aware and self._context_enabled(options):
                await self._record_guard_clarification(request, options, question)
            return AgentResult.clarification(question, trace_id=request.context.trace_id)
        if decision.category is IntentCategory.VIOLATING:
            return AgentResult.interrupted("VIOLATING_INPUT", decision.user_message, trace_id=request.context.trace_id)
        return None

    async def _intent_guard_context(self, request: AgentRequest, options: ExecutionOptions) -> GuardContext:
        """返回守卫参考历史；默认模式没有统一消息状态，因此不读取 checkpoint。"""
        return GuardContext()

    async def _record_guard_clarification(
        self,
        request: AgentRequest,
        options: ExecutionOptions,
        question: str,
    ) -> None:
        """持久化守卫澄清轮次；默认模式无统一状态 schema，保持无操作。"""

    def _context_enabled(self, options: ExecutionOptions) -> bool:
        return options.context_enabled and options.profile in self.config.context.enabled_profiles

    def _execution_timeout_seconds(self, options: ExecutionOptions, started_at: float) -> float | None:
        """取整个 SDK 请求预算与 trusted caller deadline 的较小值。"""
        request_deadline = started_at + self.config.request_timeout_seconds
        effective_deadline = min(
            request_deadline,
            options.deadline_monotonic if options.deadline_monotonic is not None else request_deadline,
        )
        remaining = effective_deadline - time.monotonic()
        if remaining <= 0:
            return None
        return remaining

    @staticmethod
    def _deadline_exceeded_result(request: AgentRequest) -> AgentResult:
        """返回 deadline 已耗尽的可分类错误，避免误报为模型或工具故障。"""
        return AgentResult.failure(
            "DEADLINE_EXCEEDED",
            "请求已超过调用方设置的 deadline。",
            recoverable=True,
            trace_id=request.context.trace_id,
        )

    def _timeout_result(self, request: AgentRequest, options: ExecutionOptions) -> AgentResult:
        """区分调用方 deadline 与 SDK 自身请求预算，便于上游决定是否重试。"""
        if options.deadline_monotonic is not None and options.deadline_monotonic <= time.monotonic():
            return self._deadline_exceeded_result(request)
        return AgentResult.failure(
            "REQUEST_TIMEOUT",
            "Agent 请求超时。",
            recoverable=True,
            trace_id=request.context.trace_id,
        )

    def _context_identity_error(self, request: AgentRequest, options: ExecutionOptions) -> AgentResult | None:
        if not self._context_enabled(options):
            return None
        context = request.context
        user_id = context.user_id.strip()
        session_id = context.session_id.strip()
        if user_id and session_id and user_id != "default_user" and session_id != "default_session":
            return None
        return AgentResult.failure(
            "CONTEXT_IDENTITY_REQUIRED",
            "启用短期上下文时必须提供非默认 user_id 和 session_id。",
            recoverable=True,
            trace_id=context.trace_id,
        )

    def _runtime_config(self, request: AgentRequest, options: ExecutionOptions) -> dict[str, Any]:
        """构造 LangGraph 使用的稳定会话标识，不混入不可信扩展字段。"""
        context = request.context
        identity = build_session_identity(
            self.metadata.name,
            options.profile,
            context.user_id,
            context.session_id,
        )
        thread_id = build_checkpoint_thread_id(identity)
        return {"configurable": {"thread_id": thread_id}}

    async def guard_model(self) -> Any | None:
        """守卫默认复用模式模型；无模型模式可返回 None。"""
        return None
