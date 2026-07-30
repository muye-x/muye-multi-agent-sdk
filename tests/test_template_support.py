"""标准模板配置加载和 capabilities 断言的离线测试。"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from muye_multi_agent_sdk import AgentCapabilities, AgentIdentity, TemplateRuntimeError, assert_agent_contract, load_yaml_config
from muye_multi_agent_sdk.version import INTERNAL_PROTOCOL_VERSION


class _TemplateSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str


class _CapabilityProvider:
    def __init__(self, capabilities: AgentCapabilities) -> None:
        self._capabilities = capabilities

    def capabilities(self) -> AgentCapabilities:
        return self._capabilities


def _identity() -> AgentIdentity:
    return AgentIdentity(
        agent_id="agent_fixture_knowledge",
        agent_version="1.0.0",
        descriptor_checksum="a" * 64,
        source_tree_checksum="b" * 64,
    )


def test_load_yaml_config_validates_shape_and_rejects_symlink(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("name: fixture\n", encoding="utf-8")

    assert load_yaml_config(config_path, _TemplateSettings).name == "fixture"

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("name: fixture\nunexpected: value\n", encoding="utf-8")
    with pytest.raises(TemplateRuntimeError, match="不符合运行时契约") as error:
        load_yaml_config(invalid_path, _TemplateSettings)
    assert error.value.code == "TEMPLATE_CONFIG_INVALID"

    link_path = tmp_path / "linked.yaml"
    link_path.symlink_to(config_path)
    with pytest.raises(TemplateRuntimeError, match="不可用"):
        load_yaml_config(link_path, _TemplateSettings)


def test_assert_agent_contract_requires_identity_protocol_and_features() -> None:
    identity = _identity()
    provider = _CapabilityProvider(
        AgentCapabilities(
            agent_name="fixture-knowledge",
            version="1.0.0",
            description="fixture",
            internal_protocol_version=INTERNAL_PROTOCOL_VERSION,
            identity=identity,
            features=["cancel", "sse", "trusted_deadline"],
        )
    )

    assert assert_agent_contract(
        provider,
        expected_identity=identity,
        required_features={"cancel", "sse"},
    ).identity == identity

    with pytest.raises(TemplateRuntimeError, match="缺少必要能力"):
        assert_agent_contract(
            provider,
            expected_identity=identity,
            required_features={"citation"},
        )
