# DevOps Repository Auditing Agent

An AI agent that autonomously audits a public GitHub repository for DevOps best-practice violations and writes an evidence-based JSON report.

The LLM decides which inspection tools to call and in what order. It follows up on what it observes, and every tool output is interpreted by the model before any finding is reported. There is no fixed checklist and no agent framework. The control loop, tool dispatch, and validation are plain Python.

## Features

- Accepts any public GitHub repository URL and runs unattended
- Explicit **Observe → Reason → Select Tool → Execute Tool → Interpret → Report** loop
- Read-only inspection: the target repo is never modified and its code is never executed
- Evidence verification before a finding is accepted (file exists, line matches), plus de-duplication
- Secret redaction in tool outputs, logs, and the report
- Full tool-call trace for transparency
- Retries with backoff on API errors, timeouts, and a hard step limit

## Requirements

- Python 3.10+
- `git` installed and on your PATH
- An API key for a supported LLM provider (Gemini free tier works)

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in your key:

```bash
cp .env.example .env
```

| Variable | Description | Example |
|---|---|---|
| `LLM_PROVIDER` | Provider to use | `gemini` |
| `LLM_MODEL` | Model name | `gemini-flash-latest` |
| `GEMINI_API_KEY` | API key for the selected provider | *(your key)* |
| `MAX_STEPS` | Max agent loop iterations | `25` |
| `TOOL_TIMEOUT_SECONDS` | Timeout per tool call | `30` |

Never commit `.env`. Only `.env.example` (variable names, no real values) belongs in the repo.

To use a different provider, set `LLM_PROVIDER`, `LLM_MODEL`, and the matching key variable listed in `.env.example`. Provider-specific code lives only in `llm.py`.

## Usage

```bash
python agent.py --repo https://github.com/example/demo-project --output report.json
```

| Argument | Required | Description |
|---|---|---|
| `--repo` | yes | Public GitHub repository URL |
| `--output` | yes | Path of the JSON report to write |

Exit codes: `0` on a completed inspection (report written, even if `violations` is empty), nonzero on fatal errors such as invalid URL, clone failure, or missing API key, with a clear message on stderr.

## Output format

```json
{
  "repository": "https://github.com/example/demo-project",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "critical",
      "file": "config.py",
      "line": 17,
      "description": "Hardcoded API credential",
      "evidence": "Credential assigned directly in source code",
      "recommendation": "Use a secrets manager or environment variable"
    }
  ]
}
```

- Severity is one of `critical`, `high`, `medium`, `low`
- `file` and `line` are `null` when not applicable
- Secret values are never included in any field

See `sample_report.json` for real output from the agent.

## Architecture

```
agent.py           CLI entry point (argparse), error handling, exit codes
agent_loop.py      Agent loop: LLM call -> tool dispatch -> feed results back
llm.py             Provider wrapper, retries, backoff
prompts.py         System prompt: inspection areas, evidence rules, severity guide
tools.py           Inspection tools and tool schemas
report.py          Violation schema, validation, de-duplication, JSON writer
tests/             Unit tests for tools, loop, and report validation
```

### How the loop works

1. The repository is shallow-cloned into a temporary directory.
2. The LLM receives the system prompt and the tool schemas.
3. The LLM chooses a tool (for example `list_files`, then `read_file` on a workflow it found).
4. The agent executes the tool, redacts secrets from the output, and returns the result to the LLM.
5. The LLM interprets the result and either calls more tools or records a finding with `report_violation`.
6. The loop ends when the LLM calls `finish_audit` or `MAX_STEPS` is reached.
7. Findings are validated, de-duplicated, and written to the JSON report. The temp clone is deleted.

### Tools

| Tool | Purpose |
|---|---|
| `list_files()` | Discover repository structure |
| `read_file(path)` | Read a file (size-capped, path-restricted) |
| `search_code(pattern)` | Search for suspicious patterns |
| `git_log()` | Inspect commit history |
| `report_violation(...)` | Record a finding with evidence |
| `finish_audit()` | Signal that the investigation is complete |

### Inspection areas

Repository hygiene, dependency management, code quality, testing, GitHub Actions/CI, security, configuration, collaboration, and build/release.

### Evidence rules

- A missing optional file or tool is not reported as a defect when an equivalent exists (for example, `pyproject.toml` instead of `requirements.txt`).
- Remote settings such as branch protection are never inferred from repository files.
- Every finding must be supported by content the agent actually read.
- If a check cannot be performed, nothing is invented.

## Safety

- The inspected repository is only read, never modified
- Repository code is never executed (no pytest/pip/scripts from the target)
- Path traversal outside the clone is blocked
- Secrets are redacted before reaching logs, traces, or the report
- Timeouts on cloning and tool calls, with a bounded number of loop steps
- No findings or answers are hardcoded for any specific repository

## Testing

```bash
pytest tests/ -v
```

Tests use a small fixture repository and a mocked LLM, so no API key or network is needed. They cover each tool, path-safety and redaction, the agent loop's dispatch behavior, report validation, and CLI error handling.

## Limitations

- Findings depend on the chosen model; smaller free-tier models may miss subtle issues
- Free-tier API rate limits can slow down large audits (the agent retries with backoff)
- Only public GitHub repositories are supported
- Remote-only configuration (branch protection, repository secrets) cannot be inspected

## License

MIT (or your chosen license)