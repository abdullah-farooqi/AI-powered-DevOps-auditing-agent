"""Command-line entry point for the repository auditing agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from agent_loop import run_agent_loop
from llm import LLMError
from report import write_report
from tools import (
    CloneError,
    RepoWorkspace,
    clone_public_repo,
    normalize_github_url,
    redact,
)

load_dotenv()
MAX_GLOBAL_TIMEOUT_SECONDS = 600


def _environment_integer(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a public GitHub repository for evidence-backed DevOps issues."
    )
    parser.add_argument("--repo", required=True, help="Public GitHub repository HTTPS URL")
    parser.add_argument("--output", required=True, help="Path for the JSON report")
    parser.add_argument(
        "--trace-output",
        help="Optional JSONL path for the redacted tool-call trace",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=os.getenv("MAX_STEPS", "40"),
        help="Maximum LLM turns (default: MAX_STEPS or 40)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=MAX_GLOBAL_TIMEOUT_SECONDS,
        help="Global audit limit, at most 600 seconds",
    )
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Skip the LLM critic verification pass",
    )
    return parser


def _write_trace(trace: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in trace:
            safe_line = redact(json.dumps(entry, ensure_ascii=False, default=str))
            handle.write(safe_line + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be a positive integer")
    if not 1 <= args.timeout_seconds <= MAX_GLOBAL_TIMEOUT_SECONDS:
        parser.error("--timeout-seconds must be between 1 and 600")

    try:
        clone_timeout = _environment_integer("CLONE_TIMEOUT_SECONDS", 120)
        tool_timeout = _environment_integer("TOOL_TIMEOUT_SECONDS", 30)
        repository_url = normalize_github_url(args.repo)
    except ValueError as exc:
        parser.error(str(exc))

    output_path = Path(args.output)
    trace_path = Path(args.trace_output) if args.trace_output else None
    if trace_path is not None and output_path.resolve() == trace_path.resolve():
        parser.error("--output and --trace-output must be different paths")

    try:
        global_deadline = time.monotonic() + args.timeout_seconds
        with tempfile.TemporaryDirectory(prefix="devops-audit-") as temporary_root:
            clone_path = Path(temporary_root) / "repository"
            clone_budget = min(
                clone_timeout,
                max(1, int(global_deadline - time.monotonic())),
            )
            clone_public_repo(repository_url, clone_path, timeout_seconds=clone_budget)
            remaining = global_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Global audit timeout reached while cloning the repository.")
            workspace = RepoWorkspace(clone_path)
            result = run_agent_loop(
                workspace,
                repository_url,
                max_steps=args.max_steps,
                timeout_seconds=max(1, int(remaining)),
                tool_timeout_seconds=tool_timeout,
                enable_critic=not args.no_critic,
            )

        write_report(repository_url, result.findings, output_path)
        if trace_path is not None:
            _write_trace(result.trace, trace_path)
    except (ValueError, CloneError, LLMError, OSError) as exc:
        print(redact(f"Audit failed: {exc}"), file=sys.stderr)
        return 1

    print(f"Report written to {output_path} ({len(result.findings)} finding(s)).")
    if trace_path is not None:
        print(f"Redacted trace written to {trace_path}.")
    for event in result.trace:
        if event.get("type") == "model_fallback":
            print(
                "Gemini model changed from "
                f"{event['from_model']} to {event['to_model']}; continuing with the existing audit context."
            )

    inspection_steps = sum(
        1 for event in result.trace if event.get("tool") not in (None, "finish_audit")
    )

    if not result.completed:
        detail = f": {result.error}" if result.error else ""
        if inspection_steps > 0:
            print(f"Warning: Audit incomplete; partial report was written{detail}", file=sys.stderr)
            return 0
        print(f"Audit failed; no inspection steps completed{detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
