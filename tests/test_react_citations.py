"""ReAct 工具结果到结构化 citation block 的回归测试。"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from muye_multi_agent_sdk import AgentMetadata, AgentRequest, ReActAgent


class _CitationAgent(ReActAgent):
    @property
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(name="citation-agent", version="1.0.0", description="citation test")

    @property
    def instructions(self) -> str:
        return "test"

    @property
    def langchain_tools(self) -> list:
        return []


def test_react_result_extracts_only_valid_unique_tool_citations() -> None:
    result = _CitationAgent()._result_from_messages(
        [
            AIMessage(content="请看 citation-fake"),
            ToolMessage(
                tool_call_id="call-1",
                name="retrieve_knowledge",
                content=json.dumps(
                    {
                        "citations": [
                            {
                                "citation_id": "citation-refund",
                                "title": "退款政策",
                                "source": "handbook.md",
                                "locator": "#refund",
                            },
                            {"citation_id": "invalid"},
                        ]
                    }
                ),
            ),
            ToolMessage(
                tool_call_id="call-2",
                name="retrieve_knowledge",
                content=json.dumps(
                    {
                        "citations": [
                            {
                                "citation_id": "citation-refund",
                                "title": "退款政策",
                                "source": "handbook.md",
                            },
                            {
                                "citation_id": "citation-security",
                                "title": "安全说明",
                                "source": "handbook.md",
                            },
                        ]
                    }
                ),
            ),
            AIMessage(content="已找到答案。"),
        ],
        AgentRequest(task="退款"),
    )

    assert result.result_data == {"markdown": "已找到答案。"}
    assert [citation.citation_id for citation in result.citations] == [
        "citation-refund",
        "citation-security",
    ]
    assert result.citations[0].locator == "#refund"
