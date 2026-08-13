# Muye Multi-Agent SDK

[English](README.en.md) · 简体中文

`muye-multi-agent-sdk` 是独立的 Python Agent SDK。
发布名为 `muye-multi-agent-sdk`，Python 导入名为 `muye_multi_agent_sdk`。

## 功能

- 三种统一模式：`CustomAgent`、`GraphAgent`、`ReActAgent`。
- 标准 HTTP API：`/health`、`/capabilities`、`/invoke`、`/invoke/stream`、`/cancel`。
- internal/public profile，public 输出只投影可展示的 Markdown、图表和 JSON。
- typed AgentEvent 与单一 SSE transport，固定生命周期：
  `session_start -> 中间事件 -> done -> session_end`。
- 进程内同会话互斥、精确取消和请求超时。
- 可选短期上下文：memory、SQLite、Postgres checkpointer，默认关闭。
- 可选意图守卫：结构化分类无意义和违规输入，默认关闭且 fail-open。
- 模型注入优先；内置 `muye-llm` 与 OpenAI-compatible 工厂。
- 可选 `muye-data` 只读客户端，三种 Agent 模式均可按需召回数据。
- ReAct 模式支持 LangChain 工具；Graph 模式支持 LangGraph 节点进度事件。
- v2.1 提供可选 Channel endpoint 与 `ChannelAgentClient`，供微信等第三方通道以受认证的标准文本契约调用 Agent。


## 安装

安装 SDK 即可使用三种 Agent 模式、FastAPI/Uvicorn transport、Muye 与
OpenAI-compatible 模型、SQLite 上下文和 internal Agent client：

```bash
python -m pip install 'muye-multi-agent-sdk>=2.1.0'
```

Postgres 上下文后端需要额外安装对应驱动：

```bash
python -m pip install 'muye-multi-agent-sdk[postgres]>=2.1.0'
```

`[all]` 面向源码开发和 CI，会安装所有模型与上下文后端，不建议作为生产环境的默认选择。

源码开发：

```bash
python -m pip install -e '.[all,dev]'
```

## 三种模式

| 模式 | 实现者职责 | 场景 |
| --- | --- | --- |
| `CustomAgent` | `metadata`、`execute()`，可选 `stream_events()` | 确定性流程、外部系统适配 |
| `GraphAgent` | `metadata`、`build_graph()`、`result_from_state()` | LangGraph 状态图 |
| `ReActAgent` | `metadata`、`instructions`、`langchain_tools` | LLM 自主工具调用 |

三种模式均使用 `AgentRequest`、`AgentResult`、`AgentEvent`。模式层不直接生成 SSE
字符串；仅 `create_app()` 的 transport 层编码 HTTP/SSE。

中间 `AgentEvent` 可表达 `thinking`、`tool`、`block` 和 `result`。SDK transport 会为这些
事件生成标准 SSE 信封；调用 SDK 的上层服务可以在自身 API 中继续投影事件，但不应把特定服务
的 block id、UI 展示状态或前端持久化规则视作 SDK 契约。

## 最小 Custom Agent

```python
from muye_multi_agent_sdk import (
    AgentMetadata,
    AgentRequest,
    AgentResult,
    CustomAgent,
    create_app,
)


class EchoAgent(CustomAgent):
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="echo-agent", version="1.0.0", description="示例 Agent")

    async def execute(self, request: AgentRequest, **_) -> AgentResult:
        return AgentResult.success({"markdown": f"已处理：{request.task}"})


app = create_app(EchoAgent())
```

`create_app()` 默认不允许浏览器跨域访问。需要开放前端来源时应显式传入
`cors_origins=["https://app.example.com"]`；只有确认任意来源均可信时才使用
`cors_origins=["*"]`。CORS 不是身份认证，internal 端点仍应部署在可信网络或认证代理之后。

## 模型配置

模型可直接注入 `ReActAgent(..., model=chat_model)`。未注入时，SDK 按
`ModelConfig` 构建内置 provider：

- `provider="muye"`：调用 `muye-llm` 的 `/api/v2/chat` 与 `/api/v2/chat/stream`。
  `MUYE_SDK_MODEL` 是网关注册的模型 alias；未设置时由网关选择
  `MUYE_LLM_DEFAULT_MODEL`。`MUYE_SDK_MODEL_ENABLE_THINKING` 未设置时同样保留网关
  默认值，显式 `true` 或 `false` 才会覆盖。
- `provider="openai_compatible"`：通过 OpenAI-compatible base URL、model 和 API key
  构造 `ChatOpenAI`。

调用方注入的 `model` 和 `guard_model` 由调用方负责关闭，便于跨 Agent 共享连接池；
仅 SDK 通过内置工厂创建的模型会在 `create_app()` 的应用关闭阶段自动关闭。

环境变量优先使用 `MUYE_SDK_*`。为兼容已有服务，未设置 SDK 键时，
`MUYE_LLM_MODEL` 和 `MUYE_LLM_BASE_URL` 分别兼容回退为模型名和地址；新部署应只使用
`MUYE_SDK_*`。例如：

```dotenv
MUYE_SDK_MODEL_PROVIDER=muye
MUYE_SDK_MODEL_BASE_URL=http://127.0.0.1:9850
# 可选：必须是 muye-llm 模型注册表中的 alias，不是上游 provider_model
MUYE_SDK_MODEL=deepseek-v4-flash
# 可选：true/false；留空时由 muye-llm 默认配置决定
MUYE_SDK_MODEL_ENABLE_THINKING=
MUYE_SDK_API_PROFILES=internal,public
MUYE_SDK_PUBLIC_PATH=/api/v1/example
MUYE_SDK_INTENT_GUARD=false
MUYE_SDK_CONTEXT_PROFILES=
```

仓储不分发 `.env` 或环境配置模板。密钥、连接串和其他敏感配置必须由部署环境注入。

Postgres checkpointer 使用异步连接池。启用时应由部署环境注入连接串，并按实际并发配置池容量：

```dotenv
MUYE_SDK_CONTEXT_PROFILES=internal
MUYE_SDK_CONTEXT_BACKEND=postgres
MUYE_SDK_CONTEXT_POSTGRES_URI=
MUYE_SDK_CONTEXT_POSTGRES_POOL_MIN_SIZE=1
MUYE_SDK_CONTEXT_POSTGRES_POOL_MAX_SIZE=10
MUYE_SDK_CONTEXT_POSTGRES_POOL_TIMEOUT_SECONDS=30
MUYE_SDK_CONTEXT_POSTGRES_POOL_MAX_IDLE_SECONDS=300
MUYE_SDK_CONTEXT_POSTGRES_POOL_MAX_LIFETIME_SECONDS=3600
```

当 `MUYE_SDK_INTENT_GUARD=true` 与 `MUYE_SDK_CONTEXT_PROFILES` 同时启用时，守卫会读取
同一 `agent/profile/user_id/session_id` 的最近人机文本，以识别“西安”这类对上一轮追问的
字段补充。可通过 `MUYE_SDK_INTENT_GUARD_CONTEXT_AWARE`、
`MUYE_SDK_INTENT_GUARD_HISTORY_MAX_MESSAGES`、`MUYE_SDK_INTENT_GUARD_HISTORY_MAX_CHARS` 和
`MUYE_SDK_INTENT_GUARD_HISTORY_IO_TIMEOUT_SECONDS` 调整预算。启用短期上下文的请求必须提供
非默认 `user_id` 和 `session_id`，避免共享 checkpoint。

## 按需数据召回

SDK 通过 `DataClient` 调用可信内网的 `muye-data`，不直接连接 Milvus、OpenSearch 或其他
数据库，也不提供建表、写入、更新和删除接口。数据资源、物理库表、字段映射与检索 pipeline
均由 `muye-data` 部署配置管理；Agent 只使用逻辑 resource alias。

```dotenv
MUYE_SDK_DATA_BASE_URL=http://127.0.0.1:9840
MUYE_SDK_DATA_TIMEOUT_SECONDS=15
MUYE_SDK_DATA_MAX_RETRIES=0
```

Custom 与 Graph 模式可以直接调用同一个异步客户端：

```python
result = await self.data_client.retrieve(
    resource="product_knowledge",
    query=request.task,
    pipeline="hybrid",
    top_k=5,
)
```

ReAct 模式使用固定作用域工具，resource、pipeline、过滤条件和返回字段不能由模型覆盖：

```python
from muye_multi_agent_sdk.tools import create_data_retrieval_tool

tool = create_data_retrieval_tool(
    self.data_client,
    resource="product_knowledge",
    pipeline="hybrid",
    fixed_filter={"op": "eq", "field": "tenant_id", "value": "tenant-1"},
    return_fields=["title", "source_url"],
)
```


## 验证

```bash
python -m pytest -q tests
python -m compileall -q src examples
python -m build --wheel .
```

许可证：[MIT](LICENSE)
