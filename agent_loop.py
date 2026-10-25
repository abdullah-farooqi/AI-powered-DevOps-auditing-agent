"""Dynamic tool-selection loop for repository audits."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from llm import (
    DEFAULT_CRITIC_MODEL,
    DEFAULT_MODEL,
    LLMError,
    ModelRouter,
    _model_chain,
    chat,
    create_client,
)
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
        messages[index]["content"] = (
            "[Earlier tool output omitted from model context; see the redacted trace.]"
        )
        total -= len(previous) - len(messages[index]["content"])


def _run_critic_pass(
    client: Any,
    workspace: RepoWorkspace,
    findings: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    *,
    critic_model: str,
    timeout_ms: int = 30_000,
) -> list[dict[str, Any]]:
    if not findings:
        return []

    critic_system_prompt = (
        "You are a strict DevOps audit critic. Review the proposed finding against the verified snippet "
        "and surrounding file context. Determine whether the evidence genuinely supports the violation and "
        "that no valid equivalent or compensating control exists. "
        'Respond strictly with a single JSON object: {"supported": true|false, "reason": "..."}.'
    )

    verified: list[dict[str, Any]] = []
    for finding in findings:
        surrounding_context = ""
        file_path = finding.get("file")
        line = finding.get("line")
        if file_path:
            try:
                full_path = workspace.safe_path(file_path)
                if full_path.is_file():
                    lines = full_path.read_text(encoding="utf-8").splitlines()
                    start = max(0, (line or 1) - 6)
                    end = min(len(lines), (line or 1) + 5)
                    surrounding_context = redact(
                        "\n".join(
                            f"{i}: {line_text}"
                            for i, line_text in enumerate(lines[start:end], start + 1)
                        )
                    )
            except (OSError, UnicodeDecodeError, ValueError):
                pass
        else:
            surrounding_context = workspace.check_equivalents()

        critic_user_prompt = (
            f"Finding candidate:\n{json.dumps(finding, indent=2)}\n\n"
            f"Surrounding repository evidence:\n{surrounding_context}\n\n"
            "Does the evidence support this finding as a real DevOps/security violation without false assumptions?"
        )

        try:
            reply = chat(
                client,
                critic_system_prompt,
                [{"role": "user", "content": critic_user_prompt}],
                [],
                model=critic_model,
                temperature=0.1,
                request_timeout_ms=timeout_ms,
            )
            raw_text = reply.text.strip()
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            decision = json.loads(match.group(0)) if match else json.loads(raw_text)

            is_supported = decision.get("supported", True)
            reason = decision.get("reason", "No reason provided")
            if is_supported is False:
                trace.append(
                    {
                        "type": "critic_drop",
                        "rule_id": finding["rule_id"],
                        "file": finding.get("file"),
                        "line": finding.get("line"),
                        "reason": redact(str(reason)),
                    }
                )
                continue
        except Exception as exc:  # noqa: BLE001
            trace.append(
                {
                    "type": "critic_error",
                    "rule_id": finding["rule_id"],
                    "error": redact(str(exc))[:500],
                }
            )

        verified.append(finding)
    return verified


def run_agent_loop(
    workspace: RepoWorkspace,
    repository_url: str,
    *,
    max_steps: int = 40,
    timeout_seconds: int = 600,
    tool_timeout_seconds: int = 30,
    enable_critic: bool = True,
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
    active_model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    router = ModelRouter(_model_chain(active_model))
    completed = False
    error_message: str | None = None
    no_tool_nudge_sent = False
    wrap_up_nudge_sent = False
    finish_rejections = 0
    steps = 0

    client = create_client()
    try:
        for step in range(1, max_steps + 1):
            steps = step
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error_message = f"Global audit timeout reached after {timeout_seconds} seconds."
                break

            if max_steps - step == 5 and not wrap_up_nudge_sent:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Budget notice: Exactly 5 steps remain in this audit. "
                            "Please stop exploring new areas, report any remaining evidenced violations, and call finish_audit."
                        ),
                    }
                )
                wrap_up_nudge_sent = True

            try:
                reply = chat(
                    client,
                    AUDIT_SYSTEM_PROMPT,
                    messages,
                    tool_schemas,
                    model=active_model,
                    request_timeout_ms=max(1_000, int(remaining * 1000)),
                    deadline_monotonic=deadline,
                    router=router,
                    trace=trace,
                    step=step,
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
                        finish_rejections += 1
                        if finish_rejections >= 2:
                            output = "Audit finished (coverage gaps recorded in trace)."
                            should_finish = True
                            trace.append(
                                {
                                    "step": step,
                                    "type": "finish_unmet_gaps",
                                    "gaps": gaps,
                                }
                            )
                        else:
                            output = "Before finishing, " + "; ".join(gaps) + "."
                    else:
                        output = "Audit finished."
                        should_finish = True
                elif call.name == "report_violation":
                    prev_adj_count = len(findings.adjustments)
                    output = findings.add(call.args)
                    if len(findings.adjustments) > prev_adj_count:
                        for adj in findings.adjustments[prev_adj_count:]:
                            trace.append(
                                {
                                    "step": step,
                                    "type": "severity_adjusted",
                                    **adj,
                                }
                            )
                else:
                    output = repo_tools.dispatch(call.name, call.args)
                output = _tool_result_for_context(output)

                try:
                    safe_args = json.loads(
                        redact(json.dumps(call.args, ensure_ascii=False, default=str))
                    )
                except (TypeError, ValueError):
                    safe_args = redact(str(call.args))[:1000]
                trace.append(
                    {
                        "step": step,
                        "tool": redact(call.name),
                        "tool_call_id": call.id,
                        "model": reply.model_used,
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
            error_message = (
                f"Maximum audit step count ({max_steps}) reached; writing partial findings."
            )

        final_findings = findings.findings
        critic_active = enable_critic and os.getenv(
            "ENABLE_CRITIC", "true"
        ).strip().lower() not in ("false", "0", "no")
        if critic_active and final_findings:
            critic_model = (
                os.getenv("LLM_CRITIC_MODEL", DEFAULT_CRITIC_MODEL).strip() or DEFAULT_CRITIC_MODEL
            )
            critic_budget_ms = min(30_000, max(5_000, int((deadline - time.monotonic()) * 1000)))
            final_findings = _run_critic_pass(
                client,
                workspace,
                final_findings,
                trace,
                critic_model=critic_model,
                timeout_ms=critic_budget_ms,
            )
    finally:
        client.close()

    return AuditResult(
        findings=final_findings,
        trace=trace,
        completed=completed,
        steps=steps,
        error=error_message,
    )
