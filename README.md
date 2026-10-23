# DevOps Repository Auditing Agent

A plain-Python agent that inspects a public GitHub repository for defensible DevOps and security issues and writes a JSON report. Gemini chooses which read-only inspection tool to call next; later calls can depend on earlier tool results. The project does not use LangChain, LangGraph, or another agent framework.

## What it does

- Audits repository hygiene, dependencies, code quality, tests, CI, security, configuration, collaboration, and build/release practices.
- Clones only public HTTPS repositories hosted at `github.com` into a temporary directory.
- Uses dynamic model-selected tool calls: `list_files`, `read_file`, `search_code`, `git_log`, and `list_workflows`.
- Verifies file findings against a file already read and a matching short snippet before adding them to the report.
- Redacts common credentials in tool output, traces, and final findings.
- Never executes code from the cloned repository or modifies the clone.
- Writes partial findings if the model fails or reaches its step/time limit; incomplete runs exit nonzero.

## Requirements

- Python 3.10 or newer
- Git on `PATH`
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env                      # Windows: copy .env.example .env
```

Edit `.env` locally and set `GEMINI_API_KEY`. Never paste the key into source code or commit `.env`; `.gitignore` excludes it. The checked-in `.env.example` contains variable names and defaults only.

| Variable | Default | Purpose |
|---|---:|---|
| `LLM_PROVIDER` | `gemini` | Provider; Gemini is currently implemented |
| `LLM_MODEL` | `gemini-3.6-flash` | Model used for audit turns |
| `GEMINI_API_KEY` | empty | Gemini credential |
| `MAX_STEPS` | `25` | Maximum model/tool-selection turns |
| `TOOL_TIMEOUT_SECONDS` | `30` | Limit for local Git inspection |
| `CLONE_TIMEOUT_SECONDS` | `120` | Maximum clone duration |
| `LLM_TIMEOUT_MS` | `60000` | Per-request model timeout |
| `LLM_MAX_RETRIES` | `4` | Retries after transient 429, 5xx, timeout, or connection errors |

## Run an audit

```bash
python agent.py \
  --repo https://github.com/OWNER/REPOSITORY \
  --output report.json \
  --trace-output trace.jsonl
```

`--repo` and `--output` are required. `--trace-output` is optional and writes redacted JSON Lines records of model-selected tool calls. `--max-steps` overrides `MAX_STEPS`; `--timeout-seconds` sets the total clone-plus-audit limit and is capped at 600 seconds.

Exit status is `0` when the audit finishes and the report is written. A partial report is still written after an LLM failure, timeout, or step limit, and the process exits nonzero with a warning. Clone, URL, configuration, and output-file failures also exit nonzero.

## Report format

The report is UTF-8 JSON:

```json
{
  "repository": "https://github.com/OWNER/REPOSITORY.git",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "high",
      "file": "config.py",
      "line": 17,
      "description": "A credential is assigned directly in source.",
      "evidence": "The assignment contains a password value.",
      "recommendation": "Load credentials from an environment variable or secret store."
    }
  ]
}
```

Allowed categories are `repository_hygiene`, `dependency_management`, `code_quality`, `testing`, `ci`, `security`, `configuration`, `collaboration`, and `build_release`. Severity is `critical`, `high`, `medium`, or `low`. `file` and `line` may be `null` for repository-wide absence findings. The evidence snippet used for validation is not included in the final report. A healthy repository can produce an empty `violations` list.

[`sample_report.json`](sample_report.json) is an authentic completed audit report for the public `psf/requests` repository.

## Architecture

| File | Responsibility |
|---|---|
| `agent.py` | CLI, validated clone lifecycle, timeouts, JSON/trace output |
| `agent_loop.py` | Dynamic LLM/tool loop, step limit, trace, partial results |
| `llm.py` | Gemini SDK setup, manual function calling, transient retries |
| `prompts.py` | Audit scope, evidence rules, precision and severity guidance |
| `tools.py` | GitHub URL validation, safe clone, redaction, read-only inspection tools |
| `report.py` | Finding schema validation, evidence verification, de-duplication, JSON writing |
| `main.py` | Small reusable `generate_text(prompt)` helper for other Python code |
| `tests/` | Local tests; they mock provider calls and do not require a key or network |

The agent does not pre-run an inspection checklist. It gives the model a repository-ready message and tool schemas, then executes the selected tool calls itself. Automatic SDK function execution is disabled. Tool output is capped and redacted before it returns to the model.

## Safety and evidence

- The URL validator accepts only `https://github.com/OWNER/REPOSITORY` URLs.
- Git is invoked with an argument list, a timeout, prompts disabled, and credential helpers disabled.
- Repository paths are resolved and checked for traversal and symlinks outside the clone; `.git` internals are not readable through file tools.
- `read_file` numbers lines, caps output, and rejects binary or oversized files. `search_code` limits pattern size, files, bytes, and matches.
- File findings require a prior `read_file`, a valid line if supplied, and a snippet found in the redacted file contents. Missing-file findings require a prior `list_files` call.
- Remote properties such as branch protection are not inferred from local files. Optional files are not defects when an equivalent is present.
- The cloned repository is never executed, installed, or modified.

## Tests and lint

```bash
python -m pytest -q tests
ruff check .
```

Tests cover path traversal, redaction, tool output, workflow parsing, report evidence and de-duplication, dynamic tool dispatch, retries, and CLI behavior. They use local temporary fixtures and mocked Gemini responses.

## Limitations

- Findings depend on the model and the evidence available in a public clone; review results before acting on them.
- Rate limits or temporary model errors can leave a partial report.
- The agent cannot inspect branch protection, repository secrets, or other remote-only settings.
- Private repositories, SSH URLs, and local filesystem paths are intentionally unsupported.
