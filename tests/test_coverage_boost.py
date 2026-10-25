from __future__ import annotations

import os

import pytest

import agent
import llm
from tools import RepoWorkspace


def test_agent_environment_integer_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_INT_VAR", "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        agent._environment_integer("TEST_INT_VAR", 10)

    monkeypatch.setenv("TEST_INT_VAR", "0")
    with pytest.raises(ValueError, match="must be at least 1"):
        agent._environment_integer("TEST_INT_VAR", 10, minimum=1)

    monkeypatch.delenv("TEST_INT_VAR", raising=False)
    assert agent._environment_integer("TEST_INT_VAR", 42) == 42


def test_llm_helpers_and_model_router_methods() -> None:
    # _positive_int
    assert llm._positive_int("NONEXISTENT_VAR", 5) == 5
    os.environ["TEMP_POS_INT"] = "-5"
    with pytest.raises(llm.LLMError, match="must be at least 1"):
        llm._positive_int("TEMP_POS_INT", 5, minimum=1)
    os.environ["TEMP_POS_INT"] = "abc"
    with pytest.raises(llm.LLMError, match="must be an integer"):
        llm._positive_int("TEMP_POS_INT", 5)

    # _model_chain
    chain = llm._model_chain("primary-model")
    assert chain[0] == "primary-model"

    # ModelRouter
    router = llm.ModelRouter(["model-a", "model-b", "model-c"])
    assert "model-a" in router.candidate_models(now=0.0)
    assert router.earliest_cooldown_delay(now=0.0) is None

    router.mark_unavailable("model-b", "404 Not Found")
    assert "model-b" not in router.candidate_models(now=0.0)
    assert "unavailable" in router.summary()

    router.mark_cooldown("model-c", 10.0, "Rate limited", now=100.0)
    assert "model-c" not in router.candidate_models(now=105.0)
    delay = router.earliest_cooldown_delay(now=105.0)
    assert delay is not None and delay <= 5.0
    assert "model-c" in router.candidate_models(now=115.0)

    router.mark_success("model-c")
    assert router.health["model-c"].consecutive_429 == 0

    with pytest.raises(llm.LLMError, match="At least one model"):
        llm.ModelRouter([])


def test_llm_openai_conversion_and_helpers() -> None:
    schemas = [
        {
            "name": "test_func",
            "description": "test function",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
    ]
    converted_tools = llm._convert_tools_openai(schemas)
    assert len(converted_tools) == 1
    assert converted_tools[0]["type"] == "function"
    assert converted_tools[0]["function"]["name"] == "test_func"

    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [{"name": "test_func", "args": {"x": "val"}, "id": "call-123"}],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call-123"},
    ]
    converted_msgs = llm._convert_messages_openai("system prompt", messages)
    assert converted_msgs[0]["role"] == "system"
    assert converted_msgs[1]["role"] == "user"
    assert converted_msgs[2]["role"] == "assistant"
    assert converted_msgs[3]["role"] == "tool"
    assert converted_msgs[3]["tool_call_id"] == "call-123"


def test_tools_branches_coverage(workspace: RepoWorkspace) -> None:
    # 1. Suspicious files branches: .env, .pyc, .log, .sqlite, .pem
    (workspace.root / ".env.dev").write_text("DEBUG=True\n", encoding="utf-8")
    (workspace.root / "test.pyc").write_bytes(b"compiled")
    (workspace.root / "app.log").write_text("log line\n", encoding="utf-8")
    (workspace.root / "data.sqlite").write_bytes(b"sqlite format 3\x00")
    (workspace.root / "server.pem").write_text("cert\n", encoding="utf-8")

    suspicious = workspace.list_suspicious_files()
    assert ".env.dev" in suspicious
    assert "test.pyc" in suspicious
    assert "app.log" in suspicious
    assert "data.sqlite" in suspicious
    assert "server.pem" in suspicious

    # 2. Workflows with invalid YAML or non-dict
    wf_dir = workspace.root / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "bad.yml").write_text("invalid: [yaml: unclosed", encoding="utf-8")
    (wf_dir / "list.yml").write_text("- item1\n- item2\n", encoding="utf-8")

    wf_summary = workspace.list_workflows()
    assert "could not parse YAML" in wf_summary
    assert "workflow root is not a mapping" in wf_summary

    # 3. parse_manifest error cases & various manifest types
    assert "does not exist" in workspace.parse_manifest("nonexistent.txt")
    (workspace.root / "package.json").write_text("{bad json", encoding="utf-8")
    assert "invalid JSON" in workspace.parse_manifest("package.json")

    # pyproject without dependencies
    (workspace.root / "pyproject.toml").write_text("[project]\nname='empty'\n", encoding="utf-8")
    out = workspace.parse_manifest("pyproject.toml")
    assert "Total dependencies: 0" in out

    # 4. git_history_search with dummy repo
    res_search = workspace.git_history_search("test")
    assert isinstance(res_search, str)

    # 5. git_show validation
    git_show_bad = workspace.git_show("not-a-hash")
    assert "valid hex commit hash" in git_show_bad
