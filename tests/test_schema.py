from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from report import write_report


@pytest.fixture
def report_schema() -> dict:
    schema_path = Path(__file__).parent.parent / "schemas" / "report.schema.json"
    assert schema_path.is_file(), "Schema file schemas/report.schema.json not found."
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_schema_validates_empty_violations(report_schema: dict) -> None:
    data = {
        "repository": "https://github.com/example/repo",
        "violations": [],
    }
    jsonschema.validate(instance=data, schema=report_schema)


def test_schema_validates_complete_findings(report_schema: dict) -> None:
    data = {
        "repository": "https://github.com/example/repo",
        "violations": [
            {
                "rule_id": "SEC-001",
                "category": "security",
                "severity": "high",
                "file": "app.py",
                "line": 15,
                "description": "Hardcoded secret key in configuration.",
                "evidence": 'SECRET_KEY = "[REDACTED]"',
                "recommendation": "Load secrets from environment variables or a vault.",
            },
            {
                "rule_id": "HYG-002",
                "category": "repository_hygiene",
                "severity": "low",
                "file": None,
                "line": None,
                "description": "Repository lacks a top-level README documentation file.",
                "evidence": "No README.md, README.rst, or README found in repository root.",
                "recommendation": "Add a comprehensive README.md file.",
            },
        ],
    }
    jsonschema.validate(instance=data, schema=report_schema)


def test_schema_validates_write_report_output(
    tmp_path: Path, report_schema: dict
) -> None:
    out = tmp_path / "test_report.json"
    findings = [
        {
            "rule_id": "CI-001",
            "category": "ci_cd",
            "severity": "critical",
            "file": ".github/workflows/deploy.yml",
            "line": 42,
            "description": "Deploy job triggers without dependency on tests.",
            "evidence": "needs: [] on production deploy job.",
            "recommendation": "Add test job to needs list.",
        }
    ]
    write_report("https://github.com/example/repo", findings, out)
    generated = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(instance=generated, schema=report_schema)


def test_schema_validates_sample_report_file(report_schema: dict) -> None:
    sample_path = Path(__file__).parent.parent / "sample_report.json"
    if sample_path.is_file():
        data = json.loads(sample_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=report_schema)


def test_schema_rejects_invalid_severity(report_schema: dict) -> None:
    bad_data = {
        "repository": "https://github.com/example/repo",
        "violations": [
            {
                "rule_id": "SEC-001",
                "category": "security",
                "severity": "urgent",  # Invalid severity
                "file": "app.py",
                "line": 1,
                "description": "desc",
                "evidence": "evid",
                "recommendation": "recom",
            }
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_data, schema=report_schema)


def test_schema_rejects_missing_required_fields(report_schema: dict) -> None:
    bad_data = {
        "repository": "https://github.com/example/repo",
        "violations": [
            {
                "rule_id": "SEC-001",
                # missing category, severity, etc.
                "description": "desc",
            }
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_data, schema=report_schema)


def test_schema_rejects_extra_top_level_properties(report_schema: dict) -> None:
    bad_data = {
        "repository": "https://github.com/example/repo",
        "violations": [],
        "extra_field": "disallowed",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_data, schema=report_schema)
