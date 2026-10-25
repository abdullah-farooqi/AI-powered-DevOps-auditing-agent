"""Finding validation, evidence checks, de-duplication, and JSON output."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from tools import RepoWorkspace, redact

CATEGORIES = {
    "repository_hygiene",
    "dependency_management",
    "code_quality",
    "testing",
    "ci",
    "security",
    "configuration",
    "collaboration",
    "build_release",
}
SEVERITIES = {"critical", "high", "medium", "low"}
RULE_PREFIX_CATEGORY = {
    "HYG": "repository_hygiene",
    "DEP": "dependency_management",
    "QUAL": "code_quality",
    "TEST": "testing",
    "CI": "ci",
    "SEC": "security",
    "CFG": "configuration",
    "COLLAB": "collaboration",
    "BUILD": "build_release",
}
_RULE_ID = re.compile(r"^([A-Z]+)-\d{3}$")
_FINDING_FIELDS = (
    "rule_id",
    "category",
    "severity",
    "file",
    "line",
    "description",
    "evidence",
    "recommendation",
)

REPORT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "report_violation",
    "description": (
        "Record one evidence-backed repository finding. Read the cited file first. "
        "For a file finding provide an exact short snippet from the redacted read_file output; "
        "for a repository-wide absence finding use file=null and line=null after list_files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "rule_id": {"type": "string", "description": "Stable identifier such as SEC-001."},
            "category": {"type": "string", "enum": sorted(CATEGORIES)},
            "severity": {"type": "string", "enum": sorted(SEVERITIES)},
            "file": {"type": "string", "nullable": True, "description": "Repository-relative path, or null for repository-wide absence."},
            "line": {"type": "integer", "nullable": True, "description": "1-based line number, or null."},
            "description": {"type": "string"},
            "evidence": {"type": "string"},
            "recommendation": {"type": "string"},
            "snippet": {"type": "string", "description": "Exact short quote from redacted file content; required when file is not null."},
        },
        "required": list(_FINDING_FIELDS),
    },
}


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _clean_snippet(text: str) -> str:
    cleaned = [re.sub(r"^\s*\d+[:\|\-]\s*", "", line) for line in text.splitlines()]
    return "\n".join(cleaned)


class FindingStore:
    """Validate findings against repository content already shown to the model."""

    def __init__(self, workspace: RepoWorkspace):
        self.workspace = workspace
        self.findings: list[dict[str, Any]] = []
        self._finding_keys: set[tuple[str, str | None, int | None]] = set()
        self._description_keys: set[tuple[str | None, str]] = set()

    def add(self, candidate: dict[str, Any]) -> str:
        if not isinstance(candidate, dict):
            return "Error: finding arguments must be an object."
        missing = [field for field in _FINDING_FIELDS if field not in candidate]
        if missing:
            return f"Error: missing required fields: {', '.join(missing)}."

        rule_id = candidate["rule_id"]
        if isinstance(rule_id, str):
            rule_id_clean = rule_id.strip().upper()
            m = re.match(r"^([A-Z]+)[-_](\d+)$", rule_id_clean)
            if m:
                p, n = m.groups()
                rule_id = f"{p}-{int(n):03d}"
            else:
                rule_id = rule_id_clean

        category = candidate["category"]
        if isinstance(category, str):
            category = category.strip().lower()
            if category == "ci/cd":
                category = "ci"

        severity = candidate["severity"]
        if isinstance(severity, str):
            severity = severity.strip().lower()

        file_name = candidate["file"]
        if file_name in ("null", "None", ""):
            file_name = None

        line = candidate["line"]
        if line in ("null", "None", ""):
            line = None
        elif isinstance(line, str) and line.isdigit():
            line = int(line)

        if not isinstance(rule_id, str) or not _RULE_ID.fullmatch(rule_id):
            return "Error: rule_id must use an uppercase prefix and three digits, such as SEC-001."
        prefix = _RULE_ID.fullmatch(rule_id).group(1)
        if prefix not in RULE_PREFIX_CATEGORY:
            return f"Error: unsupported rule_id prefix {prefix}."
        if not isinstance(category, str) or category not in CATEGORIES:
            return f"Error: category must be one of {', '.join(sorted(CATEGORIES))}."
        if RULE_PREFIX_CATEGORY[prefix] != category:
            return f"Error: rule_id prefix {prefix} does not match category {category}."
        if not isinstance(severity, str) or severity not in SEVERITIES:
            return f"Error: severity must be one of {', '.join(sorted(SEVERITIES))}."
        if file_name is not None and not isinstance(file_name, str):
            return "Error: file must be a repository-relative path or null."
        if line is not None and (type(line) is not int or line < 1):
            return "Error: line must be a positive 1-based integer or null."

        for field in ("description", "evidence", "recommendation"):
            value = candidate[field]
            if not isinstance(value, str) or not value.strip():
                return f"Error: {field} must be a non-empty string."
            if len(value) > 2000:
                return f"Error: {field} must be at most 2000 characters."

        snippet = candidate.get("snippet")
        normalized_file: str | None = None
        if file_name is None:
            if not self.workspace.root_listing_called:
                return "Error: call list_files without a directory before reporting a repository-wide absence."
            if snippet not in (None, ""):
                return "Error: snippet must be omitted when file is null."
        else:
            try:
                full_path = self.workspace.safe_path(file_name)
                normalized_file = self.workspace.normalize_relative_path(file_name)
            except ValueError as exc:
                return f"Error: invalid finding path: {exc}"
            if normalized_file not in self.workspace.files_read:
                return f"Error: read {normalized_file} first before reporting a finding in it."
            if not full_path.is_file():
                return f"Error: finding file does not exist: {normalized_file}"
            if not isinstance(snippet, str) or not snippet.strip():
                return "Error: provide a short exact snippet for file-based findings."
            try:
                raw_text = full_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return "Error: finding file could not be read as UTF-8 text."
            file_lines = raw_text.splitlines()
            if line is not None and line > len(file_lines):
                return f"Error: line {line} is past the end of {normalized_file}."
            safe_text = redact(raw_text)

            cleaned_snippet = _clean_snippet(snippet)
            norm_clean = _normalize_whitespace(cleaned_snippet)
            norm_orig = _normalize_whitespace(snippet)
            safe_norm = _normalize_whitespace(safe_text)
            raw_norm = _normalize_whitespace(raw_text)

            matches_file = (
                norm_clean in safe_norm
                or norm_orig in safe_norm
                or norm_clean in raw_norm
                or norm_orig in raw_norm
                or norm_clean.strip(" ,;.") in safe_norm
                or norm_orig.strip(" ,;.") in safe_norm
            )
            if not matches_file:
                return "Error: snippet was not found in the cited file near the reported line."

            if line is not None:
                snippet_lines = [l for l in cleaned_snippet.splitlines() if l.strip()]
                snip_len = max(1, len(snippet_lines))
                first = max(0, line - 15)
                last = min(len(file_lines), line + snip_len + 15)
                window_safe = _normalize_whitespace(redact("\n".join(file_lines[first:last])))
                window_raw = _normalize_whitespace("\n".join(file_lines[first:last]))
                matches_window = (
                    norm_clean in window_safe
                    or norm_orig in window_safe
                    or norm_clean in window_raw
                    or norm_orig in window_raw
                    or norm_clean.strip(" ,;.") in window_safe
                    or norm_orig.strip(" ,;.") in window_safe
                )
                if not matches_window:
                    for idx, f_line in enumerate(file_lines, 1):
                        f_sub = _normalize_whitespace(redact("\n".join(file_lines[idx - 1 : idx + snip_len + 2])))
                        if norm_clean in f_sub or (snippet_lines and _clean_snippet(snippet_lines[0]).strip() in f_line):
                            line = idx
                            break
            elif file_lines:
                snippet_lines = [l for l in cleaned_snippet.splitlines() if l.strip()]
                snip_len = max(1, len(snippet_lines))
                for idx, f_line in enumerate(file_lines, 1):
                    f_sub = _normalize_whitespace(redact("\n".join(file_lines[idx - 1 : idx + snip_len + 2])))
                    if norm_clean in f_sub or (snippet_lines and _clean_snippet(snippet_lines[0]).strip() in f_line):
                        line = idx
                        break

        clean_finding = {
            "rule_id": rule_id,
            "category": category,
            "severity": severity,
            "file": redact(normalized_file) if normalized_file is not None else None,
            "line": line,
            "description": redact(candidate["description"].strip()),
            "evidence": redact(candidate["evidence"].strip()),
            "recommendation": redact(candidate["recommendation"].strip()),
        }

        finding_key = (rule_id, clean_finding["file"], line)
        description_key = (
            clean_finding["file"],
            _normalize_whitespace(clean_finding["description"]).casefold(),
        )
        if finding_key in self._finding_keys or description_key in self._description_keys:
            return "Duplicate finding ignored."

        self._finding_keys.add(finding_key)
        self._description_keys.add(description_key)
        self.findings.append(clean_finding)
        return "Finding recorded."


def report_tool_schema() -> dict[str, Any]:
    return REPORT_TOOL_SCHEMA


def write_report(repository: str, findings: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write a sanitized, valid report atomically to the requested path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_findings = []
    for finding in findings:
        clean_findings.append(
            {
                field: redact(value) if isinstance(value, str) else value
                for field, value in finding.items()
            }
        )
    payload = {"repository": redact(repository), "violations": clean_findings}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
