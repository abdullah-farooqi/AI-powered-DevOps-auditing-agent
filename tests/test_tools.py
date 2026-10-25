from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import tools
from tools import RepoWorkspace, clone_public_repo, normalize_github_url, redact


def test_list_files_skips_git_and_lists_sizes(workspace: RepoWorkspace) -> None:
    result = workspace.list_files()
    assert "app.py" in result
    assert "README.md" in result
    assert ".git/config" not in result
    assert "bytes" in result
    assert workspace.list_files_called


def test_list_files_root_aliases_and_rejections(workspace: RepoWorkspace) -> None:
    # "" treats as root
    workspace.root_listing_called = False
    assert "app.py" in workspace.list_files("")
    assert workspace.root_listing_called is True

    # "." treats as root
    workspace.root_listing_called = False
    assert "app.py" in workspace.list_files(".")
    assert workspace.root_listing_called is True

    # "./" treats as root
    workspace.root_listing_called = False
    assert "app.py" in workspace.list_files("./")
    assert workspace.root_listing_called is True

    # Traversals, git, and absolute paths are rejected
    assert "Path escapes" in workspace.list_files("..")
    assert "Reading .git" in workspace.list_files(".git")
    assert "Absolute paths" in workspace.list_files("/root")


def test_list_files_can_narrow_to_directory(workspace: RepoWorkspace) -> None:
    (workspace.root / "src").mkdir()
    (workspace.root / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    result = workspace.list_files("src")
    assert "src/main.py" in result
    assert "README.md" not in result


def test_list_files_caps_results(repo_root: Path) -> None:
    for index in range(505):
        (repo_root / f"item-{index:03}.txt").write_text("x", encoding="utf-8")
    output = RepoWorkspace(repo_root).list_files()
    assert "truncated: showing 500" in output


def test_read_file_numbers_lines_and_redacts(workspace: RepoWorkspace) -> None:
    result = workspace.read_file("app.py")
    assert '1: password = "[REDACTED]"' in result
    assert '2: print("hello")' in result
    assert "app.py" in workspace.files_read


def test_read_file_range_and_missing_file(workspace: RepoWorkspace) -> None:
    assert "2: print" in workspace.read_file("app.py", start_line=2, end_line=2)
    assert workspace.read_file("missing.txt").startswith("Error: File not found")


def test_read_file_rejects_binary_and_invalid_range(workspace: RepoWorkspace) -> None:
    (workspace.root / "blob.bin").write_bytes(b"a\x00b")
    assert "Binary file" in workspace.read_file("blob.bin")
    assert "positive integer" in workspace.read_file("app.py", start_line=0)
    assert "past the end" in workspace.read_file("app.py", start_line=99)


def test_safe_path_blocks_traversal_git_and_external_symlinks(
    workspace: RepoWorkspace, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="escapes"):
        workspace.safe_path("../../outside")
    with pytest.raises(ValueError, match=".git"):
        workspace.safe_path(".git/config")
    external = tmp_path / "outside.txt"
    external.write_text("no", encoding="utf-8")
    (workspace.root / "escape.txt").symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        workspace.safe_path("escape.txt")


def test_search_code_returns_redacted_snippets(workspace: RepoWorkspace) -> None:
    result = workspace.search_code(r"password", file_glob="*.py")
    assert "app.py:1:" in result
    assert "long-secret-value-123" not in result
    assert "[REDACTED]" in result


def test_search_code_reports_invalid_regex_and_rejects_nested_repetition(
    workspace: RepoWorkspace,
) -> None:
    assert "Invalid regex" in workspace.search_code("(")
    (workspace.root / "redos.txt").write_text("a" * 28 + "!", encoding="utf-8")
    result = workspace.search_code(r"(a+)+$", timeout_seconds=0.5)
    assert "timed out" in result


def test_redact_masks_common_secret_formats() -> None:
    secrets = [
        "AKIA1234567890123456",
        "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "AIza12345678901234567890123456789012",
        "xoxb-12345678901234567890",
        "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----",
        'password="some-private-value"',
    ]
    output = redact("\n".join(secrets))
    for secret in secrets:
        assert secret not in output
    assert "REDACTED" in output


def test_normalize_github_url() -> None:
    assert normalize_github_url("https://github.com/example/repo") == (
        "https://github.com/example/repo"
    )
    assert normalize_github_url("https://github.com/example/repo.git/") == (
        "https://github.com/example/repo"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/example/repo",
        "git@github.com:example/repo.git",
        "https://github.com.evil.test/example/repo",
        "https://github.com/example/repo/extra",
        "https://user:pass@github.com/example/repo",
        "https://github.com/example/../repo",
        "https://github.com/example/repo?token=secret",
    ],
)
def test_normalize_github_url_rejects_unsafe_values(url: str) -> None:
    with pytest.raises(ValueError):
        normalize_github_url(url)


def test_clone_uses_argument_list_and_noninteractive_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    destination = tmp_path / "clone"
    url = clone_public_repo("https://github.com/example/repo", destination)
    assert url.endswith("repo.git")
    assert isinstance(captured["command"], list)
    assert "--depth" in captured["command"]
    assert captured["kwargs"]["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_rejects_non_github_before_running_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tools.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("git must not run for an invalid URL"),
    )
    with pytest.raises(ValueError):
        clone_public_repo("file:///tmp/repo", tmp_path / "clone")


def test_git_log_reads_only_commit_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test Author"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True
    )
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-m", "Initial audit fixture"],
        check=True,
        capture_output=True,
    )
    assert "Initial audit fixture" in RepoWorkspace(root).git_log()


def test_list_workflows_handles_yaml_on_key_and_flags(workspace: RepoWorkspace) -> None:
    workflow = workspace.root / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        """name: CI
on: [pull_request_target]
permissions: write-all
jobs:
  test:
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - name: tests
        continue-on-error: true
        run: pytest || true
""",
        encoding="utf-8",
    )
    summary = workspace.list_workflows()
    assert "pull_request_target" in summary
    assert "permissions" in summary
    assert "checks out untrusted pull request head" in summary
    assert "continue-on-error=true" in summary
    assert "run contains '|| true'" in summary


def test_insecure_repo_fixture_combines_a_fake_secret_and_bad_workflow(
    insecure_repo: Path,
) -> None:
    workspace = RepoWorkspace(insecure_repo)
    listing = workspace.list_files()
    secret = workspace.read_file("app.py")
    workflow = workspace.list_workflows()

    assert "README.md" not in listing
    assert "fake-assignment-key-12345" not in secret
    assert "[REDACTED]" in secret
    assert "pull_request_target" in workflow
    assert "write-all" in workflow


def test_scan_secrets_detects_credentials_without_values(workspace: RepoWorkspace) -> None:
    (workspace.root / "creds.py").write_text(
        'AWS_KEY = "AKIA1234567890123456"\nGH = "ghp_abcdefghijklmnopqrstuvwxyz123456"\n',
        encoding="utf-8",
    )
    result = workspace.scan_secrets()
    assert "creds.py:1: aws_access_key" in result
    assert "creds.py:2: github_token" in result
    # Never leak secret values
    assert "AKIA1234567890123456" not in result
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in result


def test_list_suspicious_files_detects_untracked_artifacts(workspace: RepoWorkspace) -> None:
    (workspace.root / ".env").write_text("SECRET=123\n", encoding="utf-8")
    (workspace.root / ".env.example").write_text("SECRET=\n", encoding="utf-8")
    (workspace.root / "app.pyc").write_bytes(b"\x00\x01")
    (workspace.root / "debug.log").write_text("log data\n", encoding="utf-8")
    (workspace.root / "data.sqlite").write_bytes(b"SQLite format 3\x00")
    (workspace.root / "id_rsa").write_text("private key", encoding="utf-8")

    result = workspace.list_suspicious_files()
    assert ".env" in result
    assert ".env.example" not in result  # .env.example should be ignored
    assert "app.pyc" in result
    assert "debug.log" in result
    assert "data.sqlite" in result
    assert "id_rsa" in result


def test_parse_manifest_classifies_dependencies(workspace: RepoWorkspace) -> None:
    (workspace.root / "requirements.txt").write_text(
        "requests==2.31.0\npytest>=8.0.0\nflask\nblack*\n",
        encoding="utf-8",
    )
    result = workspace.parse_manifest("requirements.txt")
    assert "requests: ==2.31.0 (pinned)" in result
    assert "pytest: >=8.0.0 (range)" in result
    assert "flask: [none] (unpinned)" in result
    assert "black: * (wildcard)" in result
    assert "No lockfile found" in result

    # With lockfile present
    (workspace.root / "requirements.lock").write_text("# lockfile\n", encoding="utf-8")
    result_with_lock = workspace.parse_manifest("requirements.txt")
    assert "Lockfile present: requirements.lock" in result_with_lock


def test_git_show_validates_commit_and_runs_read_only(workspace: RepoWorkspace) -> None:
    # Reject non-hex commit argument
    assert "Error: commit must be a valid hex commit hash" in workspace.git_show("../escape")
    assert "Error: commit must be a valid hex commit hash" in workspace.git_show("invalid;rm")

    # Valid hex commit format (mocked subprocess result or no commit)
    result = workspace.git_show("a1b2c3d4")
    assert isinstance(result, str)


def test_check_equivalents_detects_compensating_configs(workspace: RepoWorkspace) -> None:
    (workspace.root / "pyproject.toml").write_text(
        "[tool.ruff]\nline-length = 100\n", encoding="utf-8"
    )
    (workspace.root / "poetry.lock").write_text("# poetry lock\n", encoding="utf-8")
    dep_dir = workspace.root / ".github"
    dep_dir.mkdir(parents=True, exist_ok=True)
    (dep_dir / "dependabot.yml").write_text("version: 2\n", encoding="utf-8")

    result = workspace.check_equivalents()
    assert "pyproject.toml" in result
    assert "poetry.lock" in result
    assert ".github/dependabot.yml" in result


def test_list_workflows_richer_analysis(workspace: RepoWorkspace) -> None:
    wf = workspace.root / ".github" / "workflows" / "deploy.yml"
    wf.parent.mkdir(parents=True, exist_ok=True)
    wf.write_text(
        """name: Deploy
on:
  workflow_run:
    workflows: [CI]
    types: [completed]
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo test
  deploy:
    runs-on: self-hosted
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - name: install
        run: pip install git+https://github.com/org/repo
      - name: run script
        run: curl -sSL https://example.com/install.sh | bash
      - name: echo secret
        run: echo ${{ secrets.MY_SECRET }}
""",
        encoding="utf-8",
    )
    summary = workspace.list_workflows()
    assert "workflow_run" in summary
    assert "branches=['main']" in summary
    assert "deploy or release job has no dependency on test jobs" in summary
    assert "self-hosted runner on pull_request" not in summary  # on push, not pull_request
    assert "GITHUB_TOKEN write scopes: contents" in summary
    assert "curl | sh execution" in summary
    assert "pip install of unpinned git URL" in summary
    assert "secrets used directly inside run command" in summary


def test_parse_manifest_various_ecosystems(workspace: RepoWorkspace) -> None:
    # package.json
    (workspace.root / "package.json").write_text(
        json.dumps(
            {
                "dependencies": {"express": "^4.18.2", "lodash": "*"},
                "devDependencies": {"jest": "29.5.0"},
            }
        ),
        encoding="utf-8",
    )
    res_pkg = workspace.parse_manifest("package.json")
    assert "express (dependencies): ^4.18.2 (range)" in res_pkg
    assert "lodash (dependencies): * (wildcard)" in res_pkg
    assert "jest (devDependencies): 29.5.0 (pinned)" in res_pkg

    # pyproject.toml
    (workspace.root / "pyproject.toml").write_text(
        '[project]\ndependencies = ["requests>=2.0", "fastapi==0.100.0"]\n', encoding="utf-8"
    )
    res_pyproject = workspace.parse_manifest("pyproject.toml")
    assert "requests: >=2.0 (range)" in res_pyproject
    assert "fastapi: ==0.100.0 (pinned)" in res_pyproject

    # Pipfile
    (workspace.root / "Pipfile").write_text(
        '[packages]\nrequests = "*"\nflask = "==2.0.1"\n', encoding="utf-8"
    )
    res_pipfile = workspace.parse_manifest("Pipfile")
    assert "requests: * (wildcard)" in res_pipfile
    assert "flask: ==2.0.1 (pinned)" in res_pipfile

    # Cargo.toml
    (workspace.root / "Cargo.toml").write_text(
        '[dependencies]\nserde = "1.0"\ntokio = { version = "1.0" }\n', encoding="utf-8"
    )
    res_cargo = workspace.parse_manifest("Cargo.toml")
    assert "serde: 1.0 (pinned)" in res_cargo

    # go.mod
    (workspace.root / "go.mod").write_text(
        "module mymodule\n\ngo 1.20\n\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.1\n)\n",
        encoding="utf-8",
    )
    res_go = workspace.parse_manifest("go.mod")
    assert "github.com/gin-gonic/gin: v1.9.1 (pinned)" in res_go

    # composer.json
    (workspace.root / "composer.json").write_text(
        json.dumps({"require": {"monolog/monolog": "^2.0"}}), encoding="utf-8"
    )
    res_composer = workspace.parse_manifest("composer.json")
    assert "monolog/monolog (require): ^2.0 (range)" in res_composer


def test_tools_dispatch_handles_unknown_and_safe_tools(workspace: RepoWorkspace) -> None:
    from tools import RepositoryTools

    repo_tools = RepositoryTools(workspace)

    assert "unknown inspection tool" in repo_tools.dispatch("invalid_tool", {})
    assert "README.md" in repo_tools.dispatch("list_files", {})
    assert "No suspicious files" in repo_tools.dispatch("list_suspicious_files", {})
    assert "Repository Equivalents" in repo_tools.dispatch("check_equivalents", {})
    assert "Error:" in repo_tools.dispatch("git_show", {"commit": "not-a-hash"})
    assert "Error:" in repo_tools.dispatch("git_history_search", {"pattern": ""})
