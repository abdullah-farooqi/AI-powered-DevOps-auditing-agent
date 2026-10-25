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
    rejection = store.add(_finding(snippet="totally invented evidence"))
    assert "snippet was not found" in rejection
    assert "WITHOUT line-number prefixes" in rejection
    assert "past the end" in store.add(_finding(line=50))


def test_numbered_multiline_snippet_and_clean_snippet_accepted(
    workspace: RepoWorkspace,
) -> None:
    workspace.read_file("app.py")
    store = FindingStore(workspace)
    # Numbered multi-line snippet accepted
    numbered_candidate = _finding(
        rule_id="SEC-001",
        category="security",
        snippet='1: password = "[REDACTED]"\n2: print("hello")',
        line=1,
    )
    assert store.add(numbered_candidate) == "Finding recorded."

    # Clean snippet accepted on fresh store
    store2 = FindingStore(workspace)
    clean_candidate = _finding(
        rule_id="SEC-001",
        category="security",
        snippet='password = "[REDACTED]"',
        line=1,
    )
    assert store2.add(clean_candidate) == "Finding recorded."


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


def test_severity_sanity_rules_adjust_severity(workspace: RepoWorkspace) -> None:
    workspace.read_file("app.py")
    store = FindingStore(workspace)

    # 1. Exposed credential adjusted upward from low to high
    cred_finding = _finding(
        rule_id="SEC-001",
        category="security",
        severity="low",
        description="A password token is exposed in code",
        line=1,
    )
    assert store.add(cred_finding) == "Finding recorded."
    assert store.findings[0]["severity"] == "high"
    assert len(store.adjustments) == 1
    assert store.adjustments[0]["from_severity"] == "low"
    assert store.adjustments[0]["to_severity"] == "high"

    # 2. Hygiene finding adjusted downward from critical to medium
    workspace.list_files()
    hyg_store = FindingStore(workspace)
    hyg_finding = _finding(
        rule_id="HYG-001",
        category="repository_hygiene",
        severity="critical",
        file=None,
        line=None,
        snippet=None,
        description="Junk build directory committed",
        evidence="Found build output files in repository",
    )
    assert hyg_store.add(hyg_finding) == "Finding recorded."
    assert hyg_store.findings[0]["severity"] == "medium"
    assert hyg_store.adjustments[0]["to_severity"] == "medium"

    # 3. Absence finding without impact adjusted downward to low
    abs_store = FindingStore(workspace)
    abs_finding = _finding(
        rule_id="COLLAB-001",
        category="collaboration",
        severity="high",
        file=None,
        line=None,
        snippet=None,
        description="Missing PR template",
        evidence="No template file found in .github directory",
    )
    assert abs_store.add(abs_finding) == "Finding recorded."
    assert abs_store.findings[0]["severity"] == "low"


def test_deduplication_overlapping_lines_and_description_similarity(
    workspace: RepoWorkspace,
) -> None:
    (workspace.root / "app.py").write_text(
        'password = "long-secret-value-123"\nx = 1\nprint("hello")\ny = 2\nz = 3\n',
        encoding="utf-8",
    )
    workspace.read_file("app.py")
    store = FindingStore(workspace)

    # Record first finding on line 1
    assert store.add(_finding(line=1)) == "Finding recorded."

    # Overlapping line within +-3 in same category ignored
    dup_overlap = _finding(
        rule_id="SEC-002",
        line=3,
        description="Another secret in the same code block",
        snippet='print("hello")',
    )
    assert store.add(dup_overlap) == "Duplicate finding ignored (overlapping line in same file)."

    # Similar description in same category (Jaccard similarity >= 0.7) ignored
    dup_desc = _finding(
        rule_id="SEC-003",
        file=None,
        line=None,
        snippet=None,
        description="A credential is assigned directly in source.",
        evidence="Another instance of direct credential assignment",
    )
    workspace.list_files()
    assert store.add(dup_desc) == "Duplicate finding ignored (similar description in same category)."

    # Dissimilar description or different category allowed
    distinct_finding = _finding(
        rule_id="QUAL-001",
        category="code_quality",
        severity="medium",
        line=1,
        description="Bare except clause used",
        snippet='print("hello")',
    )
    assert store.add(distinct_finding) == "Finding recorded."
