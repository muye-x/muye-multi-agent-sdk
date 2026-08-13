# Muye Multi-Agent SDK

[简体中文](README.md) · English

`muye-multi-agent-sdk` is an independent Python SDK for building and serving
Custom, Graph, and ReAct Agents. The distribution name is
`muye-multi-agent-sdk`; the import package is `muye_multi_agent_sdk`.

## Features

- Three unified modes: `CustomAgent`, `GraphAgent`, and `ReActAgent`.
- Standard HTTP endpoints: `/health`, `/capabilities`, `/invoke`,
  `/invoke/stream`, and `/cancel`.
- Internal and public profiles; public responses project only displayable
  Markdown, charts, and JSON.
- Typed `AgentEvent` objects and one SSE transport with the fixed lifecycle:
  `session_start -> intermediate events -> done -> session_end`.
- Per-session in-process exclusion, precise cancellation, and request timeouts.
- Optional short-term context using memory, SQLite, or Postgres checkpointers;
  disabled by default.
- Optional intent guard for structured classification of irrelevant or prohibited
  input; disabled by default and fail-open.
- Dependency injection for models, plus built-in Muye and OpenAI-compatible
  factories.
- Optional read-only `muye-data` client for on-demand retrieval in all Agent modes.
- LangChain tools for ReAct and LangGraph node-progress events for Graph agents.
- Optional authenticated Channel endpoint and `ChannelAgentClient` for WeChat and
  other third-party channels using a standard text-only contract.

## Installation

Install the SDK for the three Agent modes, FastAPI/Uvicorn transport, Muye and
OpenAI-compatible models, SQLite context, and the internal Agent client:

```bash
python -m pip install 'muye-multi-agent-sdk>=2.1.0'
```

Install the Postgres context backend separately when needed:

```bash
python -m pip install 'muye-multi-agent-sdk[postgres]>=2.1.0'
```

The `[all]` extra is intended for source development and CI. It installs all
model and context backends and is not recommended as the default production
installation.

For source development:

```bash
python -m pip install -e '.[all,dev]'
```

## Agent modes

| Mode | Implementer responsibilities | Typical use |
| --- | --- | --- |
| `CustomAgent` | `metadata`, `execute()`, optional `stream_events()` | Deterministic workflows and external-system adapters |
| `GraphAgent` | `metadata`, `build_graph()`, `result_from_state()` | LangGraph state graphs |
| `ReActAgent` | `metadata`, `instructions`, `langchain_tools` | LLM-driven tool use |

All modes use `AgentRequest`, `AgentResult`, and `AgentEvent`. Mode implementations
do not generate SSE strings; the `create_app()` transport layer owns HTTP/SSE
encoding. Intermediate events can represent `thinking`, `tool`, `block`, and
`result` activity.

## Minimal Custom Agent

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
        return AgentMetadata(name="echo-agent", version="1.0.0", description="Example Agent")

    async def execute(self, request: AgentRequest, **_) -> AgentResult:
        return AgentResult.success({"markdown": f"Processed: {request.task}"})


app = create_app(EchoAgent())
```

Browser CORS is disabled by default. Pass explicit trusted origins through
`cors_origins` when a browser client is required. CORS is not authentication;
internal endpoints should still be protected by a trusted network or an
authentication proxy.

## Model configuration

Inject a chat model directly with `ReActAgent(..., model=chat_model)`. When no
model is injected, the SDK builds one from `ModelConfig`:

- `provider="muye"` calls the Muye LLM gateway at `/api/v2/chat` and
  `/api/v2/chat/stream`.
- `provider="openai_compatible"` builds `ChatOpenAI` from the configured base URL,
  model, and API key.

SDK-specific environment variables use the `MUYE_SDK_*` prefix. For compatibility,
`MUYE_LLM_MODEL` and `MUYE_LLM_BASE_URL` remain fallback names; new deployments
should use the SDK-prefixed names.

```dotenv
MUYE_SDK_MODEL_PROVIDER=muye
MUYE_SDK_MODEL_BASE_URL=http://127.0.0.1:9850
MUYE_SDK_MODEL=deepseek-v4-flash
MUYE_SDK_MODEL_ENABLE_THINKING=
MUYE_SDK_API_PROFILES=internal,public
MUYE_SDK_PUBLIC_PATH=/api/v1/example
MUYE_SDK_INTENT_GUARD=false
MUYE_SDK_CONTEXT_PROFILES=
```

The repository does not distribute `.env` files or configuration templates.
Inject secrets, connection strings, and other sensitive settings through the
deployment environment.

## On-demand data retrieval

The SDK uses `DataClient` to call a trusted internal `muye-data` service. It does
not connect directly to Milvus, OpenSearch, or other databases, and exposes no
database write APIs. Agents use logical resource aliases configured by
`muye-data`:

```python
result = await self.data_client.retrieve(
    resource="product_knowledge",
    query=request.task,
    pipeline="hybrid",
    top_k=5,
)
```

ReAct agents can use a fixed-scope retrieval tool. The model cannot override its
resource, pipeline, filters, or return fields:

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

## Channel endpoint

Pass `channel_request_verifier` to `create_app()` to opt in to the authenticated
`POST /internal/v1/channels/invoke` endpoint. Requests contain only a normalized
text message and trusted channel/session identifiers; provider credentials and
raw provider identifiers are not forwarded into Agent context.

## Verification

```bash
python -m pytest -q tests
python -m compileall -q src examples
python -m pip check
python -m build --wheel .
```

License: [MIT](LICENSE)
