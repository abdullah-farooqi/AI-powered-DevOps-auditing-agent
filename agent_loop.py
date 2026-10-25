"""Dynamic tool-selection loop for repository audits."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from llm import LLMError, chat, create_client
from prompts import AUDIT_SYSTEM_PROMPT
from report import REPORT_TOOL_SCHEMA, FindingStore
from tools import INSPECTION_TOOL_SCHEMAS, RepositoryTools, RepoWorkspace, redact

FINISH_TOOL_SCHEMA: dict[str, Any] = {
    "name": "finish_audit",
    "description": "Finish only after considering relevant areas and reading representative README, project, source, test, and workflow files when they exist.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
MAX_TOOL_RESULT_CHARS = 8_000
MAX_CONTEXT_TOOL_CHARS = 48_000
KEEP_RECENT_TOOL_RESULTS = 8
_SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
}
_DEPENDENCY_FILES = {
    "cargo.toml",
    "composer.json",
    "go.mod",
    "package.json",
    "pipfile",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "uv.lock",
}


@dataclass
class AuditResult:
    findings: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    completed: bool
    steps: int
    error: str | None = None


def _completion_gaps(workspace: RepoWorkspace) -> list[str]:
    """Require representative file coverage when those files exist."""
    inventory = workspace._walk_files()
    readable_paths = [relative for relative, _, size, _ in inventory if size <= 5 * 1024 * 1024]
    read_paths = {path.casefold() for path in workspace.files_read}
    gaps: list[str] = []

    if not workspace.root_listing_called:
        gaps.append("call list_files without a directory to see the full repository structure")

    def require_read(label: str, candidates: list[str]) -> None:
        if candidates and not any(path.casefold() in read_paths for path in candidates):
            gaps.append(f"read a {label} file, for example {candidates[0]}")

    readmes = [
        path
        for path in readable_paths
        if path.rsplit("/", 1)[-1].casefold().startswith("readme.")
        or path.rsplit("/", 1)[-1].casefold() == "readme"
    ]
    require_read("README", readmes)

    dependency_files = [
        path
        for path in readable_paths
        if path.rsplit("/", 1)[-1].casefold() in _DEPENDENCY_FILES
        or (
            path.rsplit("/", 1)[-1].casefold().startswith("requirements")
            and path.casefold().endswith(".txt")
        )
    ]
    require_read("dependency or project manifest", dependency_files)

    source_files = []
    test_files = []
    workflow_files = []
    for path in readable_paths:
        parts = path.casefold().split("/")
        filename = parts[-1]
        is_test = (
            any(part in {"test", "tests", "__tests__", "spec", "specs"} for part in parts)
            or filename.startswith("test_")
            or filename.endswith(("_test.go", ".spec.js", ".spec.ts"))
        )
        extension = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
        if is_test:
            test_files.append(path)
        elif (
            extension in _SOURCE_EXTENSIONS
            and filename not in {"setup.py", "conftest.py"}
            and not any(part in {"docs", "examples", "vendor"} for part in parts)
        ):
            source_files.append(path)
        if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")):
            workflow_files.append(path)

    require_read("production source", source_files)
    require_read("test", test_files)
    require_read("GitHub Actions workflow", workflow_files)
    return gaps


def _tool_result_for_context(value: str) -> str:
    safe = redact(value)
    if len(safe) > MAX_TOOL_RESULT_CHARS:
        safe = safe[:MAX_TOOL_RESULT_CHARS] + "\n[tool output truncated]"
    return safe


def _compact_messages(messages: list[dict[str, Any]]) -> None:
    tool_indices = [i for i, message in enumerate(messages) if message.get("role") == "tool"]
    total = sum(len(str(messages[i].get("content", ""))) for i in tool_indices)
    for index in tool_indices[:-KEEP_RECENT_TOOL_RESULTS]:
        if total <= MAX_CONTEXT_TOOL_CHARS:
            break
        previous = str(messages[index].get("content", ""))
        messages[index]["content"] = "[Earlier tool output omitted from model context; see the redacted trace.]"
        total -= len(previous) - len(messages[index]["content"])


def run_agent_loop(
    workspace: RepoWorkspace,
    repository_url: str,
    *,
    max_steps: int = 25,
    timeout_seconds: int = 600,
    tool_timeout_seconds: int = 30,
) -> AuditResult:
    """Run a bounded audit, letting Gemini decide each next tool call."""
    repo_tools = RepositoryTools(workspace, timeout_seconds=tool_timeout_seconds)
    findings = FindingStore(workspace)
    tool_schemas = [*INSPECTION_TOOL_SCHEMAS, REPORT_TOOL_SCHEMA, FINISH_TOOL_SCHEMA]
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"The public repository {repository_url} has been cloned into a read-only "
                "workspace. Begin the audit by choosing the inspection tool you consider "
                "most useful."
            ),
        }
    ]
    trace: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    active_model = os.getenv("LLM_MODEL", "gemini-3.6-flash").strip() or "gemini-3.6-flash"
    completed = False
    error_message: str | None = None
    no_tool_nudge_sent = False
    steps = 0

    client = create_client()
    try:
        for step in range(1, max_steps + 1):
            steps = step
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error_message = f"Global audit timeout reached after {timeout_seconds} seconds."
                break
            try:
                reply = chat(
                    client,
                    AUDIT_SYSTEM_PROMPT,
                    messages,
                    tool_schemas,
                    model=active_model,
                    request_timeout_ms=max(1_000, int(remaining * 1000)),
                    deadline_monotonic=deadline,
                )
            except LLMError as exc:
                error_message = redact(str(exc))
                trace.append(
                    {
                        "step": step,
                        "type": "llm_error",
                        "output": error_message[:MAX_TOOL_RESULT_CHARS],
                    }
                )
                break

            if reply.model_used and reply.model_used != active_model:
                trace.append(
                    {
                        "step": step,
                        "type": "model_fallback",
                        "from_model": active_model,
                        "to_model": reply.model_used,
                        "context_resumed": True,
                    }
                )
                active_model = reply.model_used

            calls_for_history = [
                {"id": call.provider_id or call.id, "name": call.name, "args": call.args}
                for call in reply.tool_calls
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": redact(reply.text),
                    "tool_calls": calls_for_history,
                    "raw_content": reply.raw_content,
                }
            )

            if not reply.tool_calls:
                if no_tool_nudge_sent:
                    error_message = "The model stopped without calling finish_audit."
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue the audit with an appropriate tool call, or call "
                            "finish_audit if the relevant areas have been covered."
                        ),
                    }
                )
                no_tool_nudge_sent = True
                _compact_messages(messages)
                continue

            should_finish = False
            for call in reply.tool_calls:
                if call.name == "finish_audit":
                    gaps = _completion_gaps(workspace)
                    if gaps:
                        output = "Before finishing, " + "; ".join(gaps) + "."
                    else:
                        output = "Audit finished."
                        should_finish = True
                elif call.name == "report_violation":
                    output = findings.add(call.args)
                else:
                    output = repo_tools.dispatch(call.name, call.args)
                output = _tool_result_for_context(output)

                try:
                    safe_args = json.loads(redact(json.dumps(call.args, ensure_ascii=False, default=str)))
                except (TypeError, ValueError):
                    safe_args = redact(str(call.args))[:1000]
                trace.append(
                    {
                        "step": step,
                        "tool": redact(call.name),
                        "tool_call_id": call.id,
                        "arguments": safe_args,
                        "output": output[:MAX_TOOL_RESULT_CHARS],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.provider_id,
                        "name": call.name,
                        "content": output,
                    }
                )

            _compact_messages(messages)
            if should_finish:
                completed = True
                break
        else:
            error_message = f"Maximum audit step count ({max_steps}) reached; writing partial findings."
    finally:
        client.close()

    return AuditResult(
        findings=findings.findings,
        trace=trace,
        completed=completed,
        steps=steps,
        error=error_message,
    )
