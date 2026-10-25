from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from agent_loop import run_agent_loop
from llm import LLMReply, ToolCall
from tools import RepoWorkspace


@pytest.fixture
def defect_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "defect-repo"
    repo.mkdir()
    (repo / "app.py").write_text(
        'AWS_ACCESS_KEY = "AKIA1234567890123456"\nprint("running app")\n',
        encoding="utf-8",
    )
    workflow_dir = repo / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ci.yml").write_text(
        """name: CI
on: [push]
permissions: write-all
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - run: echo "deploying"
""",
        encoding="utf-8",
    )
    (repo / ".env.production").write_text(
        'PASSWORD = "secret_database_password"\n', encoding="utf-8"
    )
    return repo


@pytest.fixture
def healthy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "healthy-repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Healthy Project\nFull documentation.\n", encoding="utf-8")
    (repo / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "test_app.py").write_text("def test_add():\n    assert True\n", encoding="utf-8")
    workflow_dir = repo / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ci.yml").write_text(
        """name: CI
on: [push]
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest
""",
        encoding="utf-8",
    )
    return repo


def test_end_to_end_defect_audit_detects_violations_and_redacts(
    defect_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = RepoWorkspace(defect_repo)
    planted_secret = "AKIA1234567890123456"

    class FakeClient:
        def close(self):
            pass

    observed_tool_inputs: list[str] = []

    step_index = 0

    def scripted_chat(_client, prompt, messages, _schemas, **_kwargs):
        nonlocal step_index
        if "strict DevOps audit critic" in prompt:
            # Critic pass approves all findings
            return LLMReply('{"supported": true, "reason": "verified finding"}', [], None, None)

        step_index += 1
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        latest_output = tool_messages[-1]["content"] if tool_messages else ""

        if step_index == 1:
            return LLMReply("", [ToolCall("1", "list_files", {})], None, None)

        if step_index == 2:
            assert "app.py" in latest_output
            observed_tool_inputs.append("scan_secrets")
            return LLMReply("", [ToolCall("2", "scan_secrets", {"directory": ""})], None, None)

        if step_index == 3:
            assert "app.py:1" in latest_output
            observed_tool_inputs.append("read_app")
            return LLMReply("", [ToolCall("3", "read_file", {"path": "app.py"})], None, None)

        if step_index == 4:
            assert 'AWS_ACCESS_KEY = "[REDACTED_AWS_KEY]"' in latest_output
            observed_tool_inputs.append("report_secret")
            return LLMReply(
                "",
                [
                    ToolCall(
                        "4",
                        "report_violation",
                        {
                            "rule_id": "SEC-001",
                            "category": "security",
                            "severity": "critical",
                            "file": "app.py",
                            "line": 1,
                            "description": "Hardcoded AWS Access Key in source file.",
                            "evidence": "AWS credential found on line 1.",
                            "recommendation": "Rotate key and load from AWS Secrets Manager.",
                            "snippet": 'AWS_ACCESS_KEY = "[REDACTED_AWS_KEY]"',
                        },
                    )
                ],
                None,
                None,
            )

        if step_index == 5:
            observed_tool_inputs.append("list_workflows")
            return LLMReply("", [ToolCall("5", "list_workflows", {})], None, None)

        if step_index == 6:
            assert "write-all" in latest_output
            observed_tool_inputs.append("read_workflow")
            return LLMReply(
                "", [ToolCall("6", "read_file", {"path": ".github/workflows/ci.yml"})], None, None
            )

        if step_index == 7:
            assert "permissions: write-all" in latest_output
            observed_tool_inputs.append("report_ci")
            return LLMReply(
                "",
                [
                    ToolCall(
                        "7",
                        "report_violation",
                        {
                            "rule_id": "CI-001",
                            "category": "ci",
                            "severity": "high",
                            "file": ".github/workflows/ci.yml",
                            "line": 3,
                            "description": "Workflow grants overly permissive write-all permissions.",
                            "evidence": "permissions: write-all configured at root level.",
                            "recommendation": "Use least-privilege permissions per job.",
                            "snippet": "permissions: write-all",
                        },
                    )
                ],
                None,
                None,
            )

        if step_index == 8:
            observed_tool_inputs.append("suspicious")
            return LLMReply("", [ToolCall("8", "list_suspicious_files", {})], None, None)

        if step_index == 9:
            assert ".env.production" in latest_output
            observed_tool_inputs.append("read_env")
            return LLMReply(
                "", [ToolCall("9", "read_file", {"path": ".env.production"})], None, None
            )

        if step_index == 10:
            assert "PASSWORD" in latest_output
            observed_tool_inputs.append("report_hygiene")
            return LLMReply(
                "",
                [
                    ToolCall(
                        "10",
                        "report_violation",
                        {
                            "rule_id": "HYG-001",
                            "category": "repository_hygiene",
                            "severity": "medium",
                            "file": ".env.production",
                            "line": 1,
                            "description": "Committed production environment configuration file.",
                            "evidence": "Found committed .env.production with database configuration.",
                            "recommendation": "Add .env* to .gitignore and remove from VCS.",
                            "snippet": 'PASSWORD = "[REDACTED]"',
                        },
                    )
                ],
                None,
                None,
            )

        return LLMReply("", [ToolCall("11", "finish_audit", {})], None, None)

    monkeypatch.setattr("agent_loop.create_client", lambda: FakeClient())
    monkeypatch.setattr("agent_loop.chat", scripted_chat)

    result = run_agent_loop(
        workspace,
        "https://github.com/example/defect-repo",
        max_steps=20,
        timeout_seconds=30,
        enable_critic=True,
    )

    assert result.completed
    assert len(result.findings) >= 3

    # Validate output schema
    schema_path = Path(__file__).parent.parent / "schemas" / "report.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report_data = {
        "repository": "https://github.com/example/defect-repo",
        "violations": result.findings,
    }
    jsonschema.validate(instance=report_data, schema=schema)

    # Assert dynamic tool selection sequence occurred
    assert "scan_secrets" in observed_tool_inputs
    assert "read_app" in observed_tool_inputs
    assert "list_workflows" in observed_tool_inputs
    assert "read_workflow" in observed_tool_inputs

    # Assert secret value NEVER appears in findings, evidence, trace, or JSON dump
    dumped_findings = json.dumps(result.findings)
    dumped_trace = json.dumps(result.trace)
    assert planted_secret not in dumped_findings
    assert planted_secret not in dumped_trace
    assert "secret_database_password" not in dumped_findings
    assert "secret_database_password" not in dumped_trace


def test_end_to_end_healthy_repo_produces_zero_findings(
    healthy_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = RepoWorkspace(healthy_repo)

    class FakeClient:
        def close(self):
            pass

    turn = 0

    def fake_chat(_client, _prompt, _messages, _schemas, **_kwargs):
        nonlocal turn
        turn += 1
        if turn == 1:
            return LLMReply("", [ToolCall("1", "list_files", {})], None, None)
        if turn == 2:
            return LLMReply("", [ToolCall("2", "read_file", {"path": "README.md"})], None, None)
        if turn == 3:
            return LLMReply("", [ToolCall("3", "read_file", {"path": "app.py"})], None, None)
        return LLMReply("", [ToolCall("4", "finish_audit", {})], None, None)

    monkeypatch.setattr("agent_loop.create_client", lambda: FakeClient())
    monkeypatch.setattr("agent_loop.chat", fake_chat)

    result = run_agent_loop(
        workspace,
        "https://github.com/example/healthy-repo",
        max_steps=10,
        timeout_seconds=30,
        enable_critic=False,
    )
    assert result.completed
    assert result.findings == []
