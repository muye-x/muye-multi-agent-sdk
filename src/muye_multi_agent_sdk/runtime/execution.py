"""显式执行上下文、会话互斥和取消控制。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ..contracts import AgentRequest, CancelResponse


SessionIdentity = tuple[str, str, str, str]
ExecutionKey = SessionIdentity | str


def build_session_identity(
    agent_name: str,
    profile: str,
    user_id: str,
    session_id: str,
) -> SessionIdentity:
    """构造无分隔符碰撞的进程内会话标识。"""
    return agent_name, profile, user_id, session_id


def build_checkpoint_thread_id(identity: SessionIdentity) -> str:
    """将结构化会话标识编码为稳定且不暴露原始身份的 checkpoint key。"""
    canonical_identity = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()
    return f"muye-session-v1:{digest}"


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """每次调用的可信 runtime 与并发控制信息，不存入业务 `extra`。

    默认保持 SDK Agent 的 fail-fast single-flight 语义。需要排队或活动监控的
    编排服务可显式传入对应字段，且可用 ``execution_key`` 覆盖默认的会话锁域。
    """

    profile: str = "internal"
    context_enabled: bool = False
    public_route: bool = False
    execution_key: str | None = None
    wait_for_session: bool = False
    wait_timeout_seconds: float | None = None
    idle_timeout_seconds: float | None = None
    max_hold_timeout_seconds: float | None = None
    deadline_monotonic: float | None = None


@dataclass(slots=True)
class ActiveRun:
    """当前持有会话执行权的运行信息。"""

    run_id: str
    task: asyncio.Task[object] | None
    acquired_at: float
    last_activity_at: float
    watchdog_reason: str | None = None
    _watchdog_task: asyncio.Task[None] | None = None

    def touch(self) -> None:
        """记录一次可观测活动，供 idle watchdog 判定。"""
        self.last_activity_at = time.monotonic()

    @property
    def watchdog_triggered(self) -> bool:
        """是否由 idle 或 max-hold watchdog 触发取消。"""
        return self.watchdog_reason is not None


@dataclass(slots=True)
class _LockState:
    lock: asyncio.Lock
    users: int = 0


class SessionBusyError(RuntimeError):
    """同一会话已有任务时抛出。"""


class SessionAcquireTimeoutError(TimeoutError):
    """在配置的排队等待时间内没有获得会话执行权。"""


class ExecutionManager:
    """进程内 single-flight、可选排队与取消管理。"""

    def __init__(self) -> None:
        self._locks: dict[ExecutionKey, _LockState] = {}
        self._active: dict[ExecutionKey, ActiveRun] = {}
        self._active_keys_by_identity: dict[SessionIdentity, ExecutionKey] = {}

    @staticmethod
    def identity_key(
        agent_name: str,
        request: AgentRequest,
        options: ExecutionOptions,
    ) -> SessionIdentity:
        context = request.context
        return build_session_identity(
            agent_name,
            options.profile,
            context.user_id,
            context.session_id,
        )

    @staticmethod
    def key(
        agent_name: str,
        request: AgentRequest,
        options: ExecutionOptions,
    ) -> ExecutionKey:
        if options.execution_key:
            return options.execution_key
        return ExecutionManager.identity_key(agent_name, request, options)

    @asynccontextmanager
    async def acquire(self, agent_name: str, request: AgentRequest, options: ExecutionOptions):
        """获取执行权，并在退出时停止 watchdog 与释放会话锁。

        默认维持 fail-fast 行为；``wait_for_session`` 表示无限期排队，设置
        ``wait_timeout_seconds`` 则在限定时间内排队。idle/max-hold watchdog 仅在
        成功获取锁后启动。
        """
        key = self.key(agent_name, request, options)
        identity_key = self.identity_key(agent_name, request, options)
        for name, value in (
            ("会话等待", options.wait_timeout_seconds),
            ("idle watchdog", options.idle_timeout_seconds),
            ("max-hold watchdog", options.max_hold_timeout_seconds),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} 超时必须为非负数。")
        state = self._locks.setdefault(key, _LockState(lock=asyncio.Lock()))
        state.users += 1
        lock = state.lock
        acquired = False
        should_wait = options.wait_for_session or options.wait_timeout_seconds is not None
        if not should_wait and lock.locked():
            self._release_state(key, state)
            raise SessionBusyError("当前会话已有任务正在执行，请先终止当前任务后再发送新问题。")
        try:
            if options.wait_timeout_seconds is None:
                await lock.acquire()
            else:
                try:
                    await asyncio.wait_for(lock.acquire(), timeout=options.wait_timeout_seconds)
                except asyncio.TimeoutError as exc:
                    raise SessionAcquireTimeoutError("等待当前会话的前一个任务完成超时。") from exc
            acquired = True
            now = time.monotonic()
            run = ActiveRun(
                run_id=f"run_{uuid.uuid4().hex}",
                task=asyncio.current_task(),
                acquired_at=now,
                last_activity_at=now,
            )
            self._active[key] = run
            self._active_keys_by_identity[identity_key] = key
            run._watchdog_task = self._start_watchdog(run, options)
            yield run
        finally:
            if acquired:
                active = self._active.get(key)
                if active is not None:
                    self._active.pop(key, None)
                    if self._active_keys_by_identity.get(identity_key) == key:
                        self._active_keys_by_identity.pop(identity_key, None)
                    if active._watchdog_task is not None:
                        active._watchdog_task.cancel()
                lock.release()
            self._release_state(key, state)

    def _release_state(self, key: ExecutionKey, state: _LockState) -> None:
        state.users -= 1
        if state.users == 0 and self._locks.get(key) is state:
            self._locks.pop(key, None)

    @staticmethod
    def _start_watchdog(run: ActiveRun, options: ExecutionOptions) -> asyncio.Task[None] | None:
        timeouts = [value for value in (options.idle_timeout_seconds, options.max_hold_timeout_seconds) if value is not None]
        if not timeouts:
            return None
        return asyncio.create_task(ExecutionManager._watchdog(run, options))

    @staticmethod
    async def _watchdog(run: ActiveRun, options: ExecutionOptions) -> None:
        timeouts = [value for value in (options.idle_timeout_seconds, options.max_hold_timeout_seconds) if value is not None]
        poll_interval = min(5.0, max(0.01, min(timeouts) / 2))
        try:
            while True:
                await asyncio.sleep(poll_interval)
                now = time.monotonic()
                if options.idle_timeout_seconds is not None and now - run.last_activity_at > options.idle_timeout_seconds:
                    run.watchdog_reason = "idle_timeout"
                elif options.max_hold_timeout_seconds is not None and now - run.acquired_at > options.max_hold_timeout_seconds:
                    run.watchdog_reason = "max_hold_timeout"
                else:
                    continue
                if run.task is not None and not run.task.done():
                    run.task.cancel()
                return
        except asyncio.CancelledError:
            return

    async def cancel(
        self,
        agent_name: str,
        *,
        user_id: str,
        session_id: str,
        profile: str,
        run_id: str | None = None,
        execution_key: str | None = None,
    ) -> CancelResponse:
        identity_key = build_session_identity(agent_name, profile, user_id, session_id)
        key = execution_key or self._active_keys_by_identity.get(identity_key, identity_key)
        active = self._active.get(key)
        if active is None or (run_id and active.run_id != run_id):
            return CancelResponse(status="not_found", message="当前会话没有可取消的任务")
        if active.task is not None:
            active.task.cancel()
        return CancelResponse(status="cancelled", message="已请求终止当前任务", run_id=active.run_id)
