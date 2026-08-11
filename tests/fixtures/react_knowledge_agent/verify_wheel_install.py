"""由 CI 在干净 wheel 环境执行，验证 fixture 没有回退到 SDK 源码路径。"""
from agent import FixtureKnowledgeAgent
from main import app
from muye_multi_agent_sdk import AgentIdentity, assert_agent_contract


agent = FixtureKnowledgeAgent()
assert app is not None
assert_agent_contract(
    agent,
    expected_identity=AgentIdentity(
        agent_id="agent_fixture_knowledge",
        agent_version="1.0.0",
        descriptor_checksum="a" * 64,
        source_tree_checksum="b" * 64,
    ),
    required_features={"cancel", "citation_blocks", "sse", "trusted_deadline"},
)
assert agent.langchain_tools[0].name == "fixture_knowledge_retrieve"
