"""muye-llm 网关的 SDK 标准客户端。

本模块提供 ``MuyeLlmChatModel`` LangChain 适配器，供 ReAct、IntentGuard 和
PromptTool 统一调用 muye-llm。适配器只负责协议转换与工具调用透传，不注入计费字段。
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, PrivateAttr

from ...config.models import normalize_model_base_url
from .constants import MuyeLlmStreamEventName
from .errors import MuyeLlmError
from .events import MuyeLlmStreamEvent
from .http_client import MuyeLlmHttpClient
from .messages import (
    extract_json_payload,
    parse_tool_calls,
    to_message_payload,
    tool_calls_to_langchain,
    tool_to_openai,
)


class MuyeLlmChatModel(BaseChatModel):
    """LangChain 兼容的 muye-llm 聊天模型适配器。

    ``model_name`` 是 muye-llm 注册的模型 alias，``None`` 时由网关使用默认模型。
    ``enable_thinking`` 同样仅在显式设置时透传，避免覆盖网关默认能力策略。SDK 仅
    负责协议转换与工具调用透传，不接受调用级模型切换。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str = "http://127.0.0.1:9850"
    timeout: float = 30.0
    temperature: float = 0.1
    max_tokens: int = 4096
    model_name: str | None = None
    enable_thinking: bool | None = None
    trust_env: bool = False

    _bound_tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _bound_tool_choice: str | dict[str, Any] | None = PrivateAttr(default=None)
    _client: httpx.AsyncClient | None = PrivateAttr(default=None)
    _owns_client: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs: Any) -> None:
        if "base_url" in kwargs and isinstance(kwargs["base_url"], str):
            kwargs["base_url"] = normalize_model_base_url(kwargs["base_url"])
        if "model_name" in kwargs and kwargs["model_name"] is not None:
            model_name = str(kwargs["model_name"]).strip()
            if not model_name:
                raise ValueError("model_name 不能为空")
            kwargs["model_name"] = model_name
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "muye-llm"

    def bind_tools(
        self,
        tools: list[BaseTool | dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "MuyeLlmChatModel":
        """绑定工具定义，返回新模型实例以匹配 LangChain 的不可变调用习惯。"""

        converted_tools = [
            converted
            for tool in tools
            if (converted := tool_to_openai(tool)) is not None
        ]
        instance = self.__class__(
            base_url=self.base_url,
            model_name=self.model_name,
            enable_thinking=self.enable_thinking,
            timeout=self.timeout,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            trust_env=self.trust_env,
        )
        instance._bound_tools = converted_tools
        instance._bound_tool_choice = tool_choice
        # bind_tools 会生成新模型实例；复用原实例托管的连接池以避免子实例泄漏。
        instance._client = self._http_client()
        instance._owns_client = False
        return instance

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        """使用普通 JSON 文本兼容 LangChain 结构化输出。

        muye-llm 网关不一定支持 OpenAI function calling/tool-call 风格的
        structured output。这里显式追加 JSON 输出约束并在 SDK 侧解析，供
        LLMToolSelectorMiddleware 等中间件稳定获得 dict/Pydantic 结果。
        """

        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"MuyeLlmChatModel.with_structured_output 不支持参数: {unsupported}")

        schema_dict = (
            schema.model_json_schema()
            if isinstance(schema, type) and issubclass(schema, BaseModel)
            else schema
        )
        schema_text = json.dumps(schema_dict, ensure_ascii=False)

        async def _ainvoke(input_: Any) -> Any:
            messages = self._normalize_structured_input(input_)
            messages.append(
                SystemMessage(
                    content=(
                        "你必须只返回一个符合以下 JSON Schema 的 JSON 值，不要包含解释、Markdown 或代码块。\n"
                        f"JSON Schema:\n{schema_text}"
                    )
                )
            )
            raw = await self.chat_text(messages)
            parsing_error: Exception | None = None
            parsed: Any = None
            try:
                parsed = extract_json_payload(raw)
                if isinstance(schema, type) and issubclass(schema, BaseModel):
                    parsed = schema.model_validate(parsed)
            except Exception as exc:  # noqa: BLE001 - 返回结构需携带原始解析错误
                parsing_error = exc
                if not include_raw:
                    raise
            if include_raw:
                return {
                    "raw": AIMessage(content=raw),
                    "parsed": parsed,
                    "parsing_error": parsing_error,
                }
            return parsed

        return RunnableLambda(_ainvoke)

    def _normalize_structured_input(self, input_: Any) -> list[BaseMessage | dict[str, Any]]:
        if isinstance(input_, str):
            return [HumanMessage(content=input_)]
        if hasattr(input_, "to_messages"):
            return list(input_.to_messages())
        if isinstance(input_, list):
            normalized: list[BaseMessage] = []
            for item in input_:
                if isinstance(item, BaseMessage):
                    normalized.append(item)
                    continue
                if isinstance(item, dict):
                    role = str(item.get("role") or "user")
                    content = item.get("content") or ""
                    if role == "system":
                        normalized.append(SystemMessage(content=content))
                    elif role == "assistant":
                        normalized.append(AIMessage(content=content))
                    else:
                        normalized.append(HumanMessage(content=content))
                    continue
                normalized.append(HumanMessage(content=str(item)))
            return normalized
        return [HumanMessage(content=str(input_))]

    def build_request_body(
        self,
        messages: list[BaseMessage] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """构造 muye-llm 请求体，不接受或注入计费上下文。"""

        raw_messages: list[dict[str, Any]]
        if messages and isinstance(messages[0], dict):
            raw_messages = [dict(message) for message in messages]  # type: ignore[index]
        else:
            raw_messages = [
                to_message_payload(message)
                for message in messages  # type: ignore[assignment]
            ]

        body: dict[str, Any] = {
            "messages": raw_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "trace_id": kwargs.get("trace_id", ""),
        }
        if self._bound_tools:
            body["tools"] = self._bound_tools
        if self._bound_tool_choice is not None:
            body["tool_choice"] = self._bound_tool_choice
        if self.model_name is not None:
            body["model"] = self.model_name
        if self.enable_thinking is not None:
            body["enable_thinking"] = self.enable_thinking

        return body

    def _message_from_content(self, content: str) -> AIMessage:
        tool_calls = parse_tool_calls(content)
        if tool_calls:
            return AIMessage(content="", tool_calls=tool_calls_to_langchain(tool_calls))
        return AIMessage(content=content)

    def set_http_client(self, client: httpx.AsyncClient | None) -> None:
        """注入外部 AsyncClient，主要用于进程级复用连接池或单元测试 MockTransport。"""

        if self._owns_client and self._client is not None:
            raise RuntimeError("不能在 SDK 托管客户端已创建后替换 HTTP client；请先调用 aclose()")
        self._client = client
        self._owns_client = False

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, trust_env=self.trust_env)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """关闭 SDK 自行创建的连接池，不接管调用方注入的客户端。"""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None
        self._owns_client = False

    async def _post_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return await MuyeLlmHttpClient(
            base_url=self.base_url,
            timeout=self.timeout,
            trust_env=self.trust_env,
            client=self._http_client(),
        ).post_chat(body)

    async def chat_text(
        self,
        messages: list[BaseMessage] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """调用 muye-llm 非流式接口并返回文本内容。"""

        body = self.build_request_body(messages, **kwargs)
        try:
            payload = await self._post_chat(body)
        except (httpx.HTTPError, ValueError) as exc:
            raise MuyeLlmError("muye-llm调用失败") from exc
        if not payload.get("success"):
            raise MuyeLlmError(str(payload.get("message") or "muye-llm返回失败"))
        content = str((payload.get("data") or {}).get("content") or "").strip()
        if not content:
            raise MuyeLlmError("muye-llm返回空内容")
        return content

    async def stream_events(
        self,
        messages: list[BaseMessage] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[MuyeLlmStreamEvent]:
        """调用 muye-llm 流式接口并产出结构化事件。"""

        body = self.build_request_body(messages, **kwargs)
        try:
            client = MuyeLlmHttpClient(
                base_url=self.base_url,
                timeout=self.timeout,
                trust_env=self.trust_env,
                client=self._http_client(),
            )
            async for event in client.stream_events(body):
                yield event
        except MuyeLlmError:
            raise
        except httpx.HTTPError as exc:
            raise MuyeLlmError("muye-llm流式调用失败") from exc

    async def stream_text(
        self,
        messages: list[BaseMessage] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """调用 muye-llm 流式接口并只产出文本 token。"""

        async for event in self.stream_events(messages, **kwargs):
            if event.event == MuyeLlmStreamEventName.TOKEN and event.content:
                yield event.content
            if event.event == MuyeLlmStreamEventName.ERROR:
                raise MuyeLlmError(event.message or "muye-llm流式调用失败")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = await self.chat_text(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=self._message_from_content(content))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("MuyeLlmChatModel._generate() 不支持在已有事件循环中调用")
        return asyncio.run(
            self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for event in self.stream_events(messages, **kwargs):
            if event.event == MuyeLlmStreamEventName.ERROR:
                raise MuyeLlmError(event.message or "muye-llm流式调用失败")
            if event.event == MuyeLlmStreamEventName.TOOL_CALLS:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", tool_calls=tool_calls_to_langchain(event.tool_calls))
                )
                continue
            if event.event != MuyeLlmStreamEventName.TOKEN or not event.content:
                continue
            token = event.content
            # 兼容旧网关将完整 tool_calls JSON 作为单个 token 返回的行为。
            legacy_tool_calls = parse_tool_calls(token.strip())
            if legacy_tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", tool_calls=tool_calls_to_langchain(legacy_tool_calls))
                )
                continue
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                await run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """将异步网关流桥接为同步迭代器。

        网络读取在专用线程的事件循环中进行，避免等待完整响应后才返回首个 chunk。
        LangChain 的同步回调必须在调用线程执行；消费者提前关闭迭代器时，会取消生产
        task 并关闭底层异步迭代器，避免连接和后台线程遗留。
        """
        chunks: queue.Queue[ChatGenerationChunk | BaseException | object] = queue.Queue(maxsize=1)
        finished = object()
        stopped = threading.Event()

        def put(item: ChatGenerationChunk | BaseException | object) -> bool:
            """在消费者停止后放弃排队，防止生产线程被满队列永久阻塞。"""
            while not stopped.is_set():
                try:
                    chunks.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        async def produce() -> None:
            iterator = self._astream(messages, stop=stop, run_manager=None, **kwargs)
            next_chunk = asyncio.create_task(anext(iterator))
            try:
                while True:
                    completed, _ = await asyncio.wait(
                        {next_chunk},
                        timeout=0.05,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if stopped.is_set():
                        next_chunk.cancel()
                        with suppress(asyncio.CancelledError):
                            await next_chunk
                        break
                    if not completed:
                        continue
                    try:
                        chunk = next_chunk.result()
                    except StopAsyncIteration:
                        break
                    if not put(chunk):
                        break
                    next_chunk = asyncio.create_task(anext(iterator))
            except BaseException as exc:
                put(exc)
            finally:
                next_chunk.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await next_chunk
                await iterator.aclose()
                put(finished)

        thread = threading.Thread(target=lambda: asyncio.run(produce()), daemon=True)
        thread.start()
        try:
            while True:
                item = chunks.get()
                if item is finished:
                    return
                if isinstance(item, BaseException):
                    raise item
                if run_manager and item.message.content:
                    run_manager.on_llm_new_token(item.message.content, chunk=item)
                yield item
        finally:
            stopped.set()
            thread.join(timeout=1)
