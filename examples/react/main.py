"""ReActAgent 最小可运行示例。"""
from __future__ import annotations

from langchain_core.tools import tool

from muye_multi_agent_sdk import AgentMetadata, ReActAgent, create_app


@tool
def echo(text: str) -> str:
    """原样返回输入文本。"""
    return text


class ExampleReActAgent(ReActAgent):
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="react-example", version="1.0.0", description="ReAct 模式示例")

    @property
    def instructions(self) -> str:
        return "必要时使用 echo 工具，并使用中文回答。"

    @property
    def langchain_tools(self) -> list:
        return [echo]


app = create_app(ExampleReActAgent())
