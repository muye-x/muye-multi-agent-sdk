"""SDK 运行时并发、超时和关闭生命周期回归测试。"""
from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from muye_multi_agent_sdk import (
    AgentConfig,
    AgentMetadata,
    AgentRequest,
    AgentResult,
    ContextConfig,
    CustomAgent,
    IntentGuardConfig,
    ModelConfig,
    ReActAgent,
)
from muye_multi_agent_sdk.config.models import DEFAULT_MUYE_LLM_BASE_URL
from muye_multi_agent_sdk.runtime import CheckpointerManager, ExecutionManager, ExecutionOptions, SessionAcquireTimeoutError, SessionBusyError
from muye_multi_agent_sdk.safety import IntentCategory, IntentGuard


class SlowAgent(CustomAgent):
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="slow-agent", version="1", description="超时测试")

    async def execute(self, request: AgentRequest, **_kwargs: object) -> AgentResult:
        await asyncio.sleep(0.05)
        return AgentResult.success({"markdown": "done"}, trace_id=request.context.trace_id)


class ExplodingAgent(CustomAgent):
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="exploding-agent", version="1", description="日志测试")

    async def execute(self, _request: AgentRequest, **_kwargs: object) -> AgentResult:
        raise RuntimeError("test failure")


class FailingModel:
    async def ainvoke(self, _messages: object) -> object:
        raise RuntimeError("upstream unavailable")


class CloseableModel:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class MinimalReActAgent(ReActAgent):
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="react-agent", version="1", description="关闭测试")

    @property
    def instructions(self) -> str:
        return "test"

    @property
    def langchain_tools(self) -> list[object]:
        return []


class DecisionModel:
    """按预设分类结果响应的守卫模型替身。"""

    def __init__(self, *decisions: tuple[str, str]) -> None:
        self._decisions = list(decisions)
        self.messages: list[object] = []

    async def ainvoke(self, messages: object) -> object:
        self.messages.append(messages)
        category, user_message = self._decisions.pop(0)
        return SimpleNamespace(content=json.dumps({"category": category, "user_message": user_message}))


class FakeReActRuntime:
    """仅实现守卫读取和写回 checkpoint 所需的 LangGraph 公共接口。"""

    def __init__(self, messages: list[object] | None = None, *, fail_read: bool = False) -> None:
        self.messages = messages or []
        self.fail_read = fail_read
        self.updates: list[dict[str, object]] = []

    async def aget_state(self, _config: object) -> object:
        if self.fail_read:
            raise RuntimeError("checkpoint unavailable")
        return SimpleNamespace(values={"messages": list(self.messages)})

    async def aupdate_state(self, _config: object, values: dict[str, object], *, as_node: object) -> None:
        self.updates.append({"values": values, "as_node": as_node})
        new_messages = values.get("messages", [])
        if isinstance(new_messages, list):
            self.messages.extend(new_messages)


class GuardedReActAgent(MinimalReActAgent):
    """隔离真实模型与图运行时的 ReAct 守卫测试 Agent。"""

    def __init__(self, runtime: FakeReActRuntime, guard: DecisionModel, config: AgentConfig) -> None:
        super().__init__(config, guard_model=guard)  # type: ignore[arg-type]
        self._runtime = runtime
        self.executed_tasks: list[str] = []

    async def _agent(self, _options: ExecutionOptions) -> FakeReActRuntime:
        return self._runtime

    async def execute(self, request: AgentRequest, *, options: ExecutionOptions) -> AgentResult:
        self.executed_tasks.append(request.task)
        return AgentResult.success({"markdown": "已执行"}, trace_id=request.context.trace_id)


class CheckpointedGuardedReActAgent(MinimalReActAgent):
    """使用真实 LangGraph MemorySaver 验证守卫与短期上下文的协作。"""

    def __init__(self, guard: DecisionModel, config: AgentConfig) -> None:
        super().__init__(
            config,
            model=FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
            guard_model=guard,  # type: ignore[arg-type]
        )
        self.executed_tasks: list[str] = []

    async def execute(self, request: AgentRequest, *, options: ExecutionOptions) -> AgentResult:
        self.executed_tasks.append(request.task)
        return AgentResult.success({"markdown": "已执行"}, trace_id=request.context.trace_id)


def _guarded_config() -> AgentConfig:
    return AgentConfig(
        context=ContextConfig(enabled_profiles={"internal"}),
        intent_guard=IntentGuardConfig(enabled=True),
    )


async def _complete_run(manager: ExecutionManager, session_id: str) -> None:
    request = AgentRequest(task="test", context={"user_id": "user", "session_id": session_id})
    async with manager.acquire("agent", request, ExecutionOptions()):
        pass


def test_execution_manager_releases_idle_session_lock() -> None:
    manager = ExecutionManager()

    async def run_all() -> None:
        await asyncio.gather(*(_complete_run(manager, f"s{index}") for index in range(3)))

    asyncio.run(run_all())

    assert manager._active == {}
    assert manager._locks == {}


def test_execution_manager_keeps_default_fail_fast_behavior() -> None:
    manager = ExecutionManager()
    request = AgentRequest(task="test", context={"user_id": "user", "session_id": "session"})

    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder() -> None:
            async with manager.acquire("agent", request, ExecutionOptions()):
                entered.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        with pytest.raises(SessionBusyError):
            async with manager.acquire("agent", request, ExecutionOptions()):
                pass
        release.set()
        await task

    asyncio.run(run())


def test_execution_manager_identity_is_not_ambiguous_when_ids_contain_colons() -> None:
    """不同身份不得因分隔符出现在字段中而共享同一执行锁。"""
    manager = ExecutionManager()
    first = AgentRequest(
        task="test",
        context={"user_id": "alice:mobile", "session_id": "trip"},
    )
    second = AgentRequest(
        task="test",
        context={"user_id": "alice", "session_id": "mobile:trip"},
    )

    async def run() -> None:
        async with manager.acquire("agent", first, ExecutionOptions()):
            async with manager.acquire("agent", second, ExecutionOptions()):
                pass

    asyncio.run(run())


def test_checkpoint_thread_id_is_stable_and_collision_resistant() -> None:
    """checkpoint key 应对同一身份稳定，并区分旧拼接格式会碰撞的身份。"""
    agent = SlowAgent()
    options = ExecutionOptions(profile="internal", context_enabled=True)
    first = AgentRequest(
        task="test",
        context={"user_id": "alice:mobile", "session_id": "trip"},
    )
    second = AgentRequest(
        task="test",
        context={"user_id": "alice", "session_id": "mobile:trip"},
    )

    first_thread_id = agent._runtime_config(first, options)["configurable"]["thread_id"]
    repeated_thread_id = agent._runtime_config(first, options)["configurable"]["thread_id"]
    second_thread_id = agent._runtime_config(second, options)["configurable"]["thread_id"]

    assert first_thread_id == repeated_thread_id
    assert first_thread_id != second_thread_id
    assert first_thread_id.startswith("muye-session-v1:")
    assert "alice" not in first_thread_id


def test_execution_manager_waits_for_previous_run_when_configured() -> None:
    manager = ExecutionManager()
    request = AgentRequest(task="test", context={"user_id": "user", "session_id": "session"})

    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        acquired_after_wait = asyncio.Event()

        async def holder() -> None:
            async with manager.acquire("agent", request, ExecutionOptions()):
                entered.set()
                await release.wait()

        async def waiter() -> None:
            options = ExecutionOptions(wait_timeout_seconds=1)
            async with manager.acquire("agent", request, options):
                acquired_after_wait.set()

        holder_task = asyncio.create_task(holder())
        await entered.wait()
        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert not acquired_after_wait.is_set()
        release.set()
        await asyncio.gather(holder_task, waiter_task)
        assert acquired_after_wait.is_set()

    asyncio.run(run())
    assert manager._locks == {}


def test_execution_manager_reports_queue_timeout() -> None:
    manager = ExecutionManager()
    request = AgentRequest(task="test", context={"user_id": "user", "session_id": "session"})

    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder() -> None:
            async with manager.acquire("agent", request, ExecutionOptions()):
                entered.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        with pytest.raises(SessionAcquireTimeoutError):
            async with manager.acquire("agent", request, ExecutionOptions(wait_timeout_seconds=0.01)):
                pass
        release.set()
        await task

    asyncio.run(run())


def test_execution_manager_watchdog_cancels_idle_run() -> None:
    manager = ExecutionManager()
    request = AgentRequest(task="test", context={"user_id": "user", "session_id": "session"})

    async def run() -> str | None:
        async with manager.acquire("agent", request, ExecutionOptions(idle_timeout_seconds=0.02)) as active:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return active.watchdog_reason
        return None

    assert asyncio.run(run()) == "idle_timeout"
    assert manager._locks == {}


def test_execution_manager_watchdog_respects_activity_and_max_hold() -> None:
    manager = ExecutionManager()
    request = AgentRequest(task="test", context={"user_id": "user", "session_id": "session"})

    async def run() -> str | None:
        async with manager.acquire(
            "agent",
            request,
            ExecutionOptions(idle_timeout_seconds=0.04, max_hold_timeout_seconds=0.07),
        ) as active:
            try:
                while True:
                    await asyncio.sleep(0.01)
                    active.touch()
            except asyncio.CancelledError:
                return active.watchdog_reason
        return None

    assert asyncio.run(run()) == "max_hold_timeout"


def test_execution_manager_cancels_run_with_custom_execution_key() -> None:
    manager = ExecutionManager()
    request = AgentRequest(task="test", context={"user_id": "user", "session_id": "session"})

    async def run() -> tuple[str, bool]:
        entered = asyncio.Event()
        cancelled = False

        async def holder() -> None:
            nonlocal cancelled
            try:
                async with manager.acquire(
                    "agent",
                    request,
                    ExecutionOptions(execution_key="custom:session"),
                ):
                    entered.set()
                    await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled = True

        task = asyncio.create_task(holder())
        await entered.wait()
        response = await manager.cancel(
            "agent",
            user_id="user",
            session_id="session",
            profile="internal",
        )
        await task
        return response.status, cancelled

    assert asyncio.run(run()) == ("cancelled", True)
    assert manager._active == {}
    assert manager._active_keys_by_identity == {}


def test_checkpointer_manager_initializes_once_for_concurrent_requests() -> None:
    manager = CheckpointerManager(AgentConfig().context)
    created = 0
    checkpointer = object()

    async def create() -> tuple[object, AsyncExitStack]:
        nonlocal created
        created += 1
        await asyncio.sleep(0)
        return checkpointer, AsyncExitStack()

    manager._create = create  # type: ignore[method-assign]

    async def get_all() -> list[object | None]:
        return await asyncio.gather(*(manager.get(enabled=True) for _ in range(3)))

    assert asyncio.run(get_all()) == [checkpointer, checkpointer, checkpointer]
    assert created == 1
    asyncio.run(manager.close())


def test_checkpointer_manager_uses_configured_postgres_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    import psycopg_pool
    from langgraph.checkpoint.postgres import aio as postgres_aio

    pools: list[FakePool] = []

    class FakePool:
        def __init__(self, conninfo: str, **kwargs: object) -> None:
            self.conninfo = conninfo
            self.kwargs = kwargs
            self.closed = False
            pools.append(self)

        async def __aenter__(self) -> "FakePool":
            return self

        async def __aexit__(self, *_args: object) -> None:
            self.closed = True

    class FakeSaver:
        def __init__(self, pool: FakePool) -> None:
            self.pool = pool
            self.setup_called = False

        async def setup(self) -> None:
            self.setup_called = True

    monkeypatch.setattr(psycopg_pool, "AsyncConnectionPool", FakePool)
    monkeypatch.setattr(postgres_aio, "AsyncPostgresSaver", FakeSaver)
    manager = CheckpointerManager(
        ContextConfig(
            backend="postgres",
            postgres_uri="postgresql://user:password@db.test/database",
            postgres_pool_min_size=2,
            postgres_pool_max_size=7,
            postgres_pool_timeout_seconds=11,
            postgres_pool_max_idle_seconds=22,
            postgres_pool_max_lifetime_seconds=33,
        )
    )

    async def run() -> FakeSaver:
        saver = await manager.get(enabled=True)
        assert isinstance(saver, FakeSaver)
        await manager.close()
        return saver

    saver = asyncio.run(run())
    assert saver.setup_called
    assert saver.pool is pools[0]
    assert pools[0].kwargs["min_size"] == 2
    assert pools[0].kwargs["max_size"] == 7
    assert pools[0].kwargs["timeout"] == 11
    assert pools[0].kwargs["max_idle"] == 22
    assert pools[0].kwargs["max_lifetime"] == 33
    assert pools[0].closed


def test_context_config_rejects_inverted_postgres_pool_size() -> None:
    with pytest.raises(ValueError, match="min_size"):
        ContextConfig(postgres_pool_min_size=3, postgres_pool_max_size=2)


def test_intent_guard_fails_open_when_model_is_unavailable() -> None:
    decision = asyncio.run(IntentGuard(FailingModel(), 1).classify("test", [], trace_id="trace-1"))

    assert decision.category is IntentCategory.ANALYSIS


def test_react_guard_allows_destination_that_completes_recent_question() -> None:
    runtime = FakeReActRuntime(
        [
            HumanMessage(content="帮我规划三天旅行"),
            AIMessage(content="请告诉我目的地。"),
        ]
    )
    guard = DecisionModel(("CONTEXT_CONTINUATION", ""))
    agent = GuardedReActAgent(runtime, guard, _guarded_config())
    request = AgentRequest(task="西安", context={"user_id": "u1", "session_id": "s1"})

    result = asyncio.run(agent.invoke(request, options=ExecutionOptions(profile="internal", context_enabled=True)))

    assert result.status == "success"
    assert agent.executed_tasks == ["西安"]
    prompt = guard.messages[0][1].content  # type: ignore[index,union-attr]
    assert "请告诉我目的地" in prompt
    assert "西安" in prompt


def test_guard_clarification_is_saved_for_next_contextual_answer() -> None:
    runtime = FakeReActRuntime()
    guard = DecisionModel(("MEANINGLESS", "请告诉我旅行目的地。"), ("CONTEXT_CONTINUATION", ""))
    agent = GuardedReActAgent(runtime, guard, _guarded_config())
    options = ExecutionOptions(profile="internal", context_enabled=True)
    context = {"user_id": "u1", "session_id": "s1"}

    first = asyncio.run(agent.invoke(AgentRequest(task="帮我安排旅行", context=context), options=options))
    second = asyncio.run(agent.invoke(AgentRequest(task="西安", context=context), options=options))

    assert first.status == "clarification_needed"
    assert len(runtime.updates) == 1
    saved_messages = runtime.updates[0]["values"]["messages"]  # type: ignore[index]
    assert isinstance(saved_messages, list)
    assert isinstance(saved_messages[0], HumanMessage)
    assert isinstance(saved_messages[1], AIMessage)
    assert second.status == "success"
    assert agent.executed_tasks == ["西安"]


def test_guard_clarification_round_trip_uses_real_memory_checkpointer() -> None:
    """守卫澄清写入真实 checkpoint 后，只让同会话的字段补充继续执行。"""

    guard = DecisionModel(
        ("MEANINGLESS", "请告诉我旅行目的地。"),
        ("CONTEXT_CONTINUATION", ""),
        ("MEANINGLESS", "请补充旅行需求。"),
    )
    agent = CheckpointedGuardedReActAgent(guard, _guarded_config())
    options = ExecutionOptions(profile="internal", context_enabled=True)
    primary_context = {"user_id": "u1", "session_id": "s1"}
    other_context = {"user_id": "u1", "session_id": "s2"}

    async def run_turns() -> tuple[AgentResult, AgentResult, AgentResult]:
        first = await agent.invoke(AgentRequest(task="帮我安排旅行", context=primary_context), options=options)
        second = await agent.invoke(AgentRequest(task="西安", context=primary_context), options=options)
        other_session = await agent.invoke(AgentRequest(task="西安", context=other_context), options=options)
        await agent.aclose()
        return first, second, other_session

    first, second, other_session = asyncio.run(run_turns())

    assert first.status == "clarification_needed"
    assert second.status == "success"
    assert other_session.status == "clarification_needed"
    assert agent.executed_tasks == ["西安"]
    second_prompt = guard.messages[1][1].content  # type: ignore[index,union-attr]
    assert "帮我安排旅行" in second_prompt
    assert "请告诉我旅行目的地" in second_prompt
    assert "西安" in second_prompt


def test_guard_history_excludes_system_and_tool_content() -> None:
    runtime = FakeReActRuntime(
        [
            SystemMessage(content="system secret"),
            ToolMessage(content="tool secret", tool_call_id="tool-1"),
            HumanMessage(content="去旅行"),
            AIMessage(content="请提供目的地"),
        ]
    )
    guard = DecisionModel(("CONTEXT_CONTINUATION", ""))
    agent = GuardedReActAgent(runtime, guard, _guarded_config())
    request = AgentRequest(task="西安", context={"user_id": "u1", "session_id": "s1"})

    asyncio.run(agent.invoke(request, options=ExecutionOptions(profile="internal", context_enabled=True)))

    prompt = guard.messages[0][1].content  # type: ignore[index,union-attr]
    assert "system secret" not in prompt
    assert "tool secret" not in prompt
    assert "请提供目的地" in prompt


def test_guard_history_read_failure_fails_open_for_meaningless_result() -> None:
    runtime = FakeReActRuntime(fail_read=True)
    guard = DecisionModel(("MEANINGLESS", "请补充任务"))
    agent = GuardedReActAgent(runtime, guard, _guarded_config())
    request = AgentRequest(task="西安", context={"user_id": "u1", "session_id": "s1"})

    result = asyncio.run(agent.invoke(request, options=ExecutionOptions(profile="internal", context_enabled=True)))

    assert result.status == "success"
    assert agent.executed_tasks == ["西安"]


def test_violating_input_is_interrupted_and_not_saved_to_history() -> None:
    runtime = FakeReActRuntime([HumanMessage(content="请提供目的地")])
    guard = DecisionModel(("VIOLATING", ""))
    agent = GuardedReActAgent(runtime, guard, _guarded_config())
    request = AgentRequest(task="违规输入", context={"user_id": "u1", "session_id": "s1"})

    result = asyncio.run(agent.invoke(request, options=ExecutionOptions(profile="internal", context_enabled=True)))

    assert result.status == "interrupted"
    assert runtime.updates == []
    assert agent.executed_tasks == []


def test_context_enabled_rejects_default_identity() -> None:
    agent = SlowAgent(AgentConfig(context=ContextConfig(enabled_profiles={"internal"})))

    result = asyncio.run(agent.invoke(AgentRequest(task="test"), options=ExecutionOptions(profile="internal", context_enabled=True)))

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "CONTEXT_IDENTITY_REQUIRED"


def test_invoke_timeout_uses_request_timeout_error() -> None:
    agent = SlowAgent(AgentConfig(request_timeout_seconds=0.01))

    result = asyncio.run(agent.invoke(AgentRequest(task="test")))

    assert result.error is not None
    assert result.error.code == "REQUEST_TIMEOUT"


def test_invoke_logs_trace_id_without_task_body(caplog: pytest.LogCaptureFixture) -> None:
    request = AgentRequest(task="secret task body", context={"trace_id": "trace-1"})

    result = asyncio.run(ExplodingAgent().invoke(request))

    assert result.error is not None
    assert result.error.code == "UNKNOWN_ERROR"
    assert "trace-1" in caplog.text
    assert "secret task body" not in caplog.text


def test_sdk_model_environment_overrides_legacy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUYE_LLM_MODEL", "legacy-model")
    monkeypatch.setenv("MUYE_LLM_BASE_URL", "http://legacy.test")
    monkeypatch.setenv("MUYE_SDK_MODEL", "sdk-model")
    monkeypatch.setenv("MUYE_SDK_MODEL_BASE_URL", "http://sdk.test")
    monkeypatch.setenv("MUYE_SDK_MODEL_ENABLE_THINKING", "false")

    config = AgentConfig.from_env(env_file=None)

    assert config.model.model == "sdk-model"
    assert config.model.base_url == "http://sdk.test"
    assert config.model.enable_thinking is False


def test_sdk_model_environment_omits_unconfigured_gateway_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MUYE_SDK_MODEL", "MUYE_LLM_MODEL", "MUYE_SDK_MODEL_ENABLE_THINKING"):
        monkeypatch.delenv(name, raising=False)

    config = AgentConfig.from_env(env_file=None)

    assert config.model.model is None
    assert config.model.enable_thinking is None
    assert config.model.base_url == DEFAULT_MUYE_LLM_BASE_URL


def test_openai_compatible_model_requires_name_and_rejects_thinking() -> None:
    with pytest.raises(ValueError, match="必须配置模型名"):
        ModelConfig(provider="openai_compatible")
    with pytest.raises(ValueError, match="仅支持 muye provider"):
        ModelConfig(provider="openai_compatible", model="gpt-test", enable_thinking=True)


def test_react_close_preserves_injected_model_and_clears_cached_agents() -> None:
    model = CloseableModel()
    agent = MinimalReActAgent(model=model)  # type: ignore[arg-type]
    agent._agents[True] = object()

    asyncio.run(agent.aclose())

    assert agent._agents == {}
    assert not model.closed


def test_react_close_closes_sdk_owned_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = CloseableModel()
    monkeypatch.setattr("muye_multi_agent_sdk.modes.react.build_chat_model", lambda _config: model)
    agent = MinimalReActAgent()
    agent._get_model()

    asyncio.run(agent.aclose())

    assert model.closed
