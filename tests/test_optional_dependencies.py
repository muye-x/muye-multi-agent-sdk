"""核心与可选依赖分组的发布契约测试。"""
from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_default_install_includes_primary_modes_and_integrations() -> None:
    """默认安装必须覆盖主要模式、transport、模型、SQLite 和 internal client。"""
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    optional_dependency_groups = metadata["project"]["optional-dependencies"]

    assert {
        "fastapi>=0.115.0",
        "httpx>=0.27",
        "langchain-core>=1.2.0",
        "langchain>=1.2.15",
        "langchain-openai>=0.3.0",
        "langgraph>=1.1.6",
        "langgraph-checkpoint-sqlite>=3.0.3",
        "uvicorn[standard]>=0.30.0",
    }.issubset(dependencies)
    assert {
        "standard",
        "server",
        "graph",
        "react",
        "muye-llm",
        "internal-client",
        "openai",
        "sqlite",
    }.isdisjoint(optional_dependency_groups)


def test_model_factory_import_does_not_load_internal_http_client() -> None:
    """模型工厂与 internal HTTP client 不应共享无关的导入副作用。"""
    result = _run_isolated(
        """
import sys

from muye_multi_agent_sdk.integrations import build_chat_model

assert callable(build_chat_model)
assert "muye_multi_agent_sdk.integrations.internal_agent" not in sys.modules
"""
    )

    assert result.returncode == 0, result.stderr
