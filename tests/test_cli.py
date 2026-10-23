from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent
from agent_loop import AuditResult
from llm import LLMError


def test_cli_requires_repo_and_output() -> None:
    with pytest.raises(SystemExit) as error:
        agent.main([])
    assert error.value.code == 2


def test_cli_rejects_invalid_github_url(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        agent.main(["--repo", "file:///tmp/repo", "--output", str(tmp_path / "report.json")])
    assert error.value.code == 2


def test_cli_success_writes_json_and_optional_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MAX_STEPS", raising=False)

    def fake_clone(url, destination, timeout_seconds):
        destination.mkdir(parents=True)
        (destination / "README.md").write_text("# safe fixture\n", encoding="utf-8")
        return "https://github.com/example/repo.git"

    def fake_audit(workspace, repository_url, **kwargs):
        assert (workspace.root / "README.md").is_file()
        return AuditResult(
            findings=[],
            trace=[{"step": 1, "tool": "list_files", "output": "README.md"}],
            completed=True,
            steps=2,
        )

    monkeypatch.setattr(agent, "clone_public_repo", fake_clone)
    monkeypatch.setattr(agent, "run_agent_loop", fake_audit)
    output = tmp_path / "report.json"
    trace = tmp_path / "trace.jsonl"
    result = agent.main(
        [
            "--repo",
            "https://github.com/example/repo",
            "--output",
            str(output),
            "--trace-output",
            str(trace),
        ]
    )
    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "repository": "https://github.com/example/repo.git",
        "violations": [],
    }
    assert json.loads(trace.read_text(encoding="utf-8").strip())["tool"] == "list_files"


def test_cli_writes_partial_report_and_returns_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        agent,
        "clone_public_repo",
        lambda url, destination, timeout_seconds: destination.mkdir(parents=True),
    )
    monkeypatch.setattr(
        agent,
        "run_agent_loop",
        lambda *args, **kwargs: AuditResult([], [], False, 1, "LLM timed out"),
    )
    output = tmp_path / "partial.json"
    result = agent.main(
        ["--repo", "https://github.com/example/repo", "--output", str(output)]
    )
    assert result == 1
    assert json.loads(output.read_text(encoding="utf-8"))["violations"] == []


def test_cli_handles_missing_api_key_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        agent,
        "clone_public_repo",
        lambda url, destination, timeout_seconds: destination.mkdir(parents=True),
    )

    def fail_audit(*args, **kwargs):
        raise LLMError("GEMINI_API_KEY is missing")

    monkeypatch.setattr(agent, "run_agent_loop", fail_audit)
    result = agent.main(
        ["--repo", "https://github.com/example/repo", "--output", str(tmp_path / "report.json")]
    )
    assert result == 1


def test_cli_rejects_trace_report_path_collision(tmp_path: Path) -> None:
    same_path = str(tmp_path / "report.json")
    with pytest.raises(SystemExit) as error:
        agent.main(
            [
                "--repo",
                "https://github.com/example/repo",
                "--output",
                same_path,
                "--trace-output",
                same_path,
            ]
        )
    assert error.value.code == 2
