from __future__ import annotations

import json

import agent_loop
from agent_loop import _completion_gaps
from llm import LLMReply, ToolCall
from tools import RepoWorkspace


class FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _reply(name: str | None = None, args=None) -> LLMReply:
    calls = [ToolCall(id=f"call-{name}", name=name, args=args or {})] if name else []
    return LLMReply(text="", tool_calls=calls, raw=None, raw_content=None)


def test_tool_sequence_is_dynamic_and_finding_is_validated(
    workspace: RepoWorkspace, monkeypatch
) -> None:
    client = FakeClient()
    monkeypatch.setattr(agent_loop, "create_client", lambda: client)
    call_count = 0

    def fake_chat(_client, _system_prompt, messages, _schemas, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _reply("list_files")
        tool_outputs = [m.get("content", "") for m in messages if m.get("role") == "tool"]
        if call_count == 2:
            assert "app.py" in tool_outputs[-1]
            return _reply("read_file", {"path": "app.py"})
        if call_count == 3:
            assert 'password = "[REDACTED]"' in tool_outputs[-1]
            return _reply(
                "report_violation",
                {
                    "rule_id": "SEC-001",
                    "category": "security",
                    "severity": "high",
                    "file": "app.py",
                    "line": 1,
                    "description": "A password is hard-coded in source.",
                    "evidence": "The assignment contains a password value.",
                    "recommendation": "Read the credential from a secret store.",
                    "snippet": 'password = "[REDACTED]"',
                },
            )
        if call_count == 4:
            return _reply("finish_audit")
        if call_count == 5:
            assert "Before finishing" in tool_outputs[-1]
            return _reply("read_file", {"path": "README.md"})
        assert call_count == 6
        return _reply("finish_audit")

    monkeypatch.setattr(agent_loop, "chat", fake_chat)
    result = agent_loop.run_agent_loop(
        workspace,
        "https://github.com/example/repo.git",
        max_steps=7,
        timeout_seconds=30,
    )
    assert result.completed
    assert len(result.findings) == 1
    assert [entry.get("tool") for entry in result.trace] == [
        "list_files",
        "read_file",
        "report_violation",
        "finish_audit",
        "read_file",
        "finish_audit",
    ]
    assert "long-secret-value-123" not in json.dumps(result.trace)
    assert client.closed


def test_tool_errors_are_returned_to_the_model(workspace: RepoWorkspace, monkeypatch) -> None:
    client = FakeClient()
    monkeypatch.setattr(agent_loop, "create_client", lambda: client)
    workspace.list_files()
    workspace.read_file("README.md")
    workspace.read_file("app.py")
    saw_error = False

    def fake_chat(_client, _system_prompt, messages, _schemas, **_kwargs):
        nonlocal saw_error
        if not saw_error:
            return _reply("read_file", {"path": "../../outside"})
        tool_outputs = [m.get("content", "") for m in messages if m.get("role") == "tool"]
        assert "Error:" in tool_outputs[-1]
        return _reply("finish_audit")

    def chat_then_mark(*args, **kwargs):
        nonlocal saw_error
        result = fake_chat(*args, **kwargs)
        if result.tool_calls and result.tool_calls[0].name == "read_file":
            saw_error = True
        return result

    monkeypatch.setattr(agent_loop, "chat", chat_then_mark)
    result = agent_loop.run_agent_loop(
        workspace,
        "https://github.com/example/repo.git",
        max_steps=3,
        timeout_seconds=30,
    )
    assert result.completed
    assert result.trace[0]["output"].startswith("Error:")


def test_max_steps_returns_partial_result(workspace: RepoWorkspace, monkeypatch) -> None:
    client = FakeClient()
    monkeypatch.setattr(agent_loop, "create_client", lambda: client)
    monkeypatch.setattr(agent_loop, "chat", lambda *args, **kwargs: _reply("list_files"))
    result = agent_loop.run_agent_loop(
        workspace,
        "https://github.com/example/repo.git",
        max_steps=1,
        timeout_seconds=30,
    )
    assert not result.completed
    assert "Maximum audit step count" in result.error
    assert len(result.trace) == 1
    assert client.closed


def test_no_tool_reply_gets_one_continue_prompt(workspace: RepoWorkspace, monkeypatch) -> None:
    client = FakeClient()
    monkeypatch.setattr(agent_loop, "create_client", lambda: client)
    workspace.list_files()
    workspace.read_file("README.md")
    workspace.read_file("app.py")
    calls = 0

    def fake_chat(_client, _system_prompt, messages, _schemas, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _reply()
        assert any(
            m.get("role") == "user" and "Continue the audit" in m.get("content", "")
            for m in messages
        )
        return _reply("finish_audit")

    monkeypatch.setattr(agent_loop, "chat", fake_chat)
    result = agent_loop.run_agent_loop(
        workspace,
        "https://github.com/example/repo.git",
        max_steps=3,
        timeout_seconds=30,
    )
    assert result.completed
    assert calls == 2


def test_completion_gate_requires_real_source_file(workspace: RepoWorkspace) -> None:
    (workspace.root / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
    workspace.list_files()
    workspace.read_file("README.md")
    workspace.read_file("setup.py")
    gaps = _completion_gaps(workspace)
    assert any("production source" in gap for gap in gaps)
    workspace.read_file("app.py")
    assert not any("production source" in gap for gap in _completion_gaps(workspace))
