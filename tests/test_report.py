from __future__ import annotations

import json
from pathlib import Path

from report import FindingStore, write_report
from tools import RepoWorkspace


def _finding(**overrides):
    finding = {
        "rule_id": "SEC-001",
        "category": "security",
        "severity": "high",
        "file": "app.py",
        "line": 1,
        "description": "A credential is assigned directly in source.",
        "evidence": "The line assigns a password value.",
        "recommendation": "Load the credential from a secret store or environment variable.",
        "snippet": 'password = "[REDACTED]"',
    }
    finding.update(overrides)
    return finding


def test_file_finding_requires_file_to_be_read(workspace: RepoWorkspace) -> None:
    store = FindingStore(workspace)
    result = store.add(_finding())
    assert "read app.py first" in result
    assert store.findings == []


def test_valid_finding_is_recorded_without_snippet_or_secret(workspace: RepoWorkspace) -> None:
    workspace.read_file("app.py")
    store = FindingStore(workspace)
    candidate = _finding(
        description="Hard-coded credential AKIA1234567890123456 in source.",
        evidence="The password value appears on this assignment.",
    )
    assert store.add(candidate) == "Finding recorded."
    assert len(store.findings) == 1
    encoded = json.dumps(store.findings)
    assert "snippet" not in encoded
    assert "AKIA1234567890123456" not in encoded


def test_bad_severity_and_mismatched_rule_category_are_rejected(
    workspace: RepoWorkspace,
) -> None:
    workspace.read_file("app.py")
    store = FindingStore(workspace)
    assert "severity must be" in store.add(_finding(severity="urgent"))
    assert "does not match category" in store.add(
        _finding(rule_id="DEP-001", category="security")
    )


def test_snippet_and_line_are_verified(workspace: RepoWorkspace) -> None:
    workspace.read_file("app.py")
    store = FindingStore(workspace)
    assert "snippet was not found" in store.add(_finding(snippet="totally invented evidence"))
    assert "past the end" in store.add(_finding(line=50))


def test_report_path_must_be_read_before_finding(workspace: RepoWorkspace) -> None:
    workspace.list_files()
    store = FindingStore(workspace)
    assert "read app.py first" in store.add(_finding())


def test_repository_wide_absence_requires_list_files(workspace: RepoWorkspace) -> None:
    store = FindingStore(workspace)
    candidate = _finding(
        rule_id="TEST-001",
        category="testing",
        severity="medium",
        file=None,
        line=None,
        snippet=None,
        description="No test files are present.",
        evidence="The repository file listing contained no tests or test configuration.",
    )
    assert "call list_files" in store.add(candidate)
    workspace.list_files()
    assert store.add(candidate) == "Finding recorded."


def test_findings_are_deduplicated(workspace: RepoWorkspace) -> None:
    workspace.read_file("app.py")
    store = FindingStore(workspace)
    assert store.add(_finding()) == "Finding recorded."
    assert store.add(_finding(line=2)) == "Duplicate finding ignored."
    assert len(store.findings) == 1


def test_empty_report_is_valid_json_and_secret_safe(
    tmp_path: Path, workspace: RepoWorkspace
) -> None:
    workspace.read_file("app.py")
    store = FindingStore(workspace)
    store.add(_finding(description="Credential ghp_abcdefghijklmnopqrstuvwxyz123456 is exposed."))
    output = tmp_path / "nested" / "report.json"
    write_report("https://github.com/example/repo", store.findings, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["repository"] == "https://github.com/example/repo"
    assert len(payload["violations"]) == 1
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in output.read_text(encoding="utf-8")

    empty = tmp_path / "empty.json"
    write_report("https://github.com/example/healthy", [], empty)
    assert json.loads(empty.read_text(encoding="utf-8")) == {
        "repository": "https://github.com/example/healthy",
        "violations": [],
    }


def test_snippet_with_line_prefixes_and_multiline_is_accepted(
    workspace: RepoWorkspace,
) -> None:
    workspace.read_file("app.py")
    store = FindingStore(workspace)
    candidate = _finding(
        line=1,
        snippet='1: password = "[REDACTED]"\n2: print("hello")',
    )
    assert store.add(candidate) == "Finding recorded."
    assert len(store.findings) == 1


def test_lenient_rule_id_and_category_normalization(workspace: RepoWorkspace) -> None:
    workspace.read_file("app.py")
    store = FindingStore(workspace)
    candidate = _finding(
        rule_id="sec-1",
        category="CI/CD",
        snippet='password = "[REDACTED]"',
    )
    # SEC rule prefix does not match CI/CD category
    assert "does not match" in store.add(candidate)

    # SEC rule prefix with uppercase category
    candidate2 = _finding(
        rule_id="sec-1",
        category="Security",
        snippet='password = "[REDACTED]"',
    )
    assert store.add(candidate2) == "Finding recorded."
    assert store.findings[0]["rule_id"] == "SEC-001"
    assert store.findings[0]["category"] == "security"
