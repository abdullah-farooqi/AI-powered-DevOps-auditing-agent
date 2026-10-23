from __future__ import annotations

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
    assert "nested repetition" in workspace.search_code(r"(a+)+")


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
        "https://github.com/example/repo.git"
    )
    assert normalize_github_url("https://github.com/example/repo.git/") == (
        "https://github.com/example/repo.git"
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
