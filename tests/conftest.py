from __future__ import annotations

from pathlib import Path

import pytest

from tools import RepoWorkspace


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text(
        'password = "long-secret-value-123"\nprint("hello")\n', encoding="utf-8"
    )
    (root / "README.md").write_text("# Demo repository\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("private internals\n", encoding="utf-8")
    return root


@pytest.fixture
def workspace(repo_root: Path) -> RepoWorkspace:
    return RepoWorkspace(repo_root)


@pytest.fixture
def insecure_repo(tmp_path: Path) -> Path:
    """A compact scenario with a fake credential, risky CI, and no README."""
    root = tmp_path / "insecure-repo"
    workflow = root / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    (root / "app.py").write_text('API_KEY = "fake-assignment-key-12345"\n', encoding="utf-8")
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
    return root
