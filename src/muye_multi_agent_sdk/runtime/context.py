"""LangGraph 短期上下文的可选资源管理。"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from ..config import ContextConfig


class CheckpointerManager:
    """按显式配置延迟创建并在应用关闭时释放 checkpointer。"""

    def __init__(self, config: ContextConfig) -> None:
        self._config = config
        self._stack: AsyncExitStack | None = None
        self._checkpointer: Any | None = None
        self._lock = asyncio.Lock()

    async def get(self, *, enabled: bool) -> Any | None:
        if not enabled:
            return None
        if self._checkpointer is not None:
            return self._checkpointer
        async with self._lock:
            if self._checkpointer is not None:
                return self._checkpointer
            checkpointer, stack = await self._create()
            self._checkpointer = checkpointer
            self._stack = stack
            return checkpointer

    async def _create(self) -> tuple[Any, AsyncExitStack | None]:
        if self._config.backend == "memory":
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver(), None
        if self._config.backend == "sqlite":
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            path = self._config.sqlite_path.strip() or "agent_checkpoints.db"
            if path != ":memory:":
                path = str(Path(path).expanduser())
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            return await self._create_managed(AsyncSqliteSaver.from_conn_string(path))
        if self._config.backend == "postgres":
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            if self._config.postgres_uri is None:
                raise ValueError("postgres 短期上下文需要 MUYE_SDK_CONTEXT_POSTGRES_URI")
            pool = AsyncConnectionPool(
                self._config.postgres_uri.get_secret_value(),
                min_size=self._config.postgres_pool_min_size,
                max_size=self._config.postgres_pool_max_size,
                timeout=self._config.postgres_pool_timeout_seconds,
                max_idle=self._config.postgres_pool_max_idle_seconds,
                max_lifetime=self._config.postgres_pool_max_lifetime_seconds,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                open=False,
            )
            return await self._create_postgres(pool, AsyncPostgresSaver)
        raise ValueError(f"不支持的上下文后端: {self._config.backend}")

    @staticmethod
    async def _create_postgres(pool: Any, saver_type: type[Any]) -> tuple[Any, AsyncExitStack]:
        """在同一生命周期内打开连接池、初始化 saver 并确保失败时关闭。"""
        stack = AsyncExitStack()
        try:
            managed_pool = await stack.enter_async_context(pool)
            checkpointer = saver_type(managed_pool)
            await checkpointer.setup()
        except BaseException:
            await stack.aclose()
            raise
        return checkpointer, stack

    async def _create_managed(self, resource: Any) -> tuple[Any, AsyncExitStack]:
        stack = AsyncExitStack()
        try:
            checkpointer = await stack.enter_async_context(resource)
            await checkpointer.setup()
        except BaseException:
            await stack.aclose()
            raise
        return checkpointer, stack

    async def close(self) -> None:
        async with self._lock:
            stack = self._stack
            self._stack = None
            self._checkpointer = None
            if stack is not None:
                await stack.aclose()
