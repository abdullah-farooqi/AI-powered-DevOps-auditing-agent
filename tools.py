"""Safe, read-only inspection tools for public GitHub repositories."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import yaml

MAX_LIST_ENTRIES = 500
MAX_WALK_ENTRIES = 20_000
MAX_READ_BYTES = 5 * 1024 * 1024
MAX_READ_OUTPUT = 20_000
MAX_READ_LINES = 300
MAX_SEARCH_FILE_BYTES = 1024 * 1024
MAX_SEARCH_FILES = 2_000
MAX_SEARCH_TOTAL_BYTES = 20 * 1024 * 1024
MAX_SEARCH_MATCHES = 50

_SECRET_PATTERNS = (
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)(\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[=:]\s*)"
    r"(['\"]?)([^\s'\"]{8,})\2"
)
_RULE_PATTERN = re.compile(r"^([A-Z]+)-(\d{3})$")


class CloneError(RuntimeError):
    """Raised when a public GitHub repository cannot be cloned safely."""


def redact(text: str) -> str:
    """Mask common credential formats while preserving useful context."""
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return _ASSIGNMENT_SECRET.sub(r"\1\2[REDACTED]\2", redacted)


def normalize_github_url(url: str) -> str:
    """Validate a public GitHub HTTPS repository URL and return its canonical URL."""
    try:
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected an HTTPS GitHub repository URL.") from exc

    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or len(parts) != 2
    ):
        raise ValueError("Expected https://github.com/OWNER/REPOSITORY.")

    owner, repository = parts
    repository = repository.removesuffix(".git")
    valid_part = re.compile(r"^[A-Za-z0-9_.-]+$")
    if (
        not owner
        or not repository
        or owner in {".", ".."}
        or repository in {".", ".."}
        or not valid_part.fullmatch(owner)
        or not valid_part.fullmatch(repository)
    ):
        raise ValueError("The GitHub owner or repository name is invalid.")
    return f"https://github.com/{owner}/{repository}"


def clone_public_repo(url: str, destination: str | Path, timeout_seconds: int = 120) -> str:
    """Clone a validated public GitHub repository without invoking a shell."""
    canonical_url = normalize_github_url(url)
    clone_url = f"{canonical_url}.git"
    destination = Path(destination).resolve()
    env = os.environ.copy()
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
    )
    command = [
        "git",
        "-c",
        "core.symlinks=false",
        "-c",
        "credential.helper=",
        "clone",
        "--depth",
        "100",
        "--no-tags",
        "--single-branch",
        "--",
        clone_url,
        str(destination),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise CloneError("Git is not installed or is not available on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise CloneError(f"Git clone timed out after {timeout_seconds} seconds.") from exc

    if result.returncode:
        detail = redact((result.stderr or result.stdout or "git clone failed").strip())
        raise CloneError(f"Could not clone public repository: {detail[:1200]}")
    return clone_url


class RepoWorkspace:
    """A read-only view of a repository checkout."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Repository root must be a directory.")
        self.files_read: set[str] = set()
        self.list_files_called = False
        self.root_listing_called = False
        self.walk_truncated = False

    def safe_path(self, relative_path: str) -> Path:
        """Resolve a repository-relative path while blocking traversal and .git."""
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("A non-empty repository-relative path is required.")
        if "\x00" in relative_path or "\\" in relative_path:
            raise ValueError("Invalid repository path.")
        if relative_path.startswith("/"):
            raise ValueError("Absolute paths are not allowed.")

        parts = [part for part in relative_path.split("/") if part not in ("", ".")]
        if not parts or ".." in parts:
            raise ValueError("Path escapes repository.")
        if ".git" in parts:
            raise ValueError("Reading .git internals is not allowed.")

        candidate = self.root.joinpath(*parts).resolve(strict=False)
        if candidate == self.root or self.root not in candidate.parents:
            raise ValueError("Path escapes repository.")
        return candidate

    @staticmethod
    def normalize_relative_path(relative_path: str) -> str:
        return PurePosixPath(*[p for p in relative_path.split("/") if p not in ("", ".")]).as_posix()

    def _walk_files(self) -> list[tuple[str, Path, int, bool]]:
        entries: list[tuple[str, Path, int, bool]] = []
        self.walk_truncated = False
        for current, directories, filenames in os.walk(self.root, followlinks=False):
            current_path = Path(current)
            directories[:] = sorted(d for d in directories if d != ".git")
            for directory in list(directories):
                full_path = current_path / directory
                if full_path.is_symlink():
                    directories.remove(directory)
                    rel = full_path.relative_to(self.root).as_posix()
                    entries.append((rel, full_path, full_path.lstat().st_size, True))
                    if len(entries) >= MAX_WALK_ENTRIES:
                        self.walk_truncated = True
                        return sorted(entries, key=lambda item: item[0].casefold())

            for filename in filenames:
                full_path = current_path / filename
                rel = full_path.relative_to(self.root).as_posix()
                try:
                    stat = full_path.lstat()
                except OSError:
                    continue
                entries.append((rel, full_path, stat.st_size, full_path.is_symlink()))
                if len(entries) >= MAX_WALK_ENTRIES:
                    self.walk_truncated = True
                    return sorted(entries, key=lambda item: item[0].casefold())
        return sorted(entries, key=lambda item: item[0].casefold())

    def list_files(self, directory: str = "") -> str:
        if not isinstance(directory, str):
            return "Error: directory must be a string."
        normalized_directory = ""
        base = self.root
        trimmed_dir = directory.strip()
        if trimmed_dir in ("", ".", "./"):
            self.root_listing_called = True
        else:
            try:
                base = self.safe_path(directory)
            except ValueError as exc:
                return f"Error: {exc}"
            if not base.is_dir():
                return f"Error: not a directory: {directory}"
            normalized_directory = self.normalize_relative_path(directory).rstrip("/") + "/"

        entries = [
            entry
            for entry in self._walk_files()
            if entry[0].startswith(normalized_directory)
        ]
        total = len(entries)
        shown = entries[:MAX_LIST_ENTRIES]
        lines = [
            f"{rel}  ({size} bytes{', symlink' if symlink else ''})"
            for rel, _, size, symlink in shown
        ]
        if total > MAX_LIST_ENTRIES:
            if self.walk_truncated:
                lines.append(
                    f"[truncated: showing {MAX_LIST_ENTRIES}; file scan capped at "
                    f"{MAX_WALK_ENTRIES}+ entries]"
                )
            else:
                lines.append(f"[truncated: showing {MAX_LIST_ENTRIES} of {total} files]")
        self.list_files_called = True
        return redact("\n".join(lines) if lines else "No files found.")

    def _read_text(self, relative_path: str) -> tuple[Path, str]:
        full_path = self.safe_path(relative_path)
        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {relative_path}")
        size = full_path.stat().st_size
        if size > MAX_READ_BYTES:
            raise ValueError(f"File is too large to inspect ({size} bytes).")
        data = full_path.read_bytes()
        if b"\x00" in data:
            raise ValueError("Binary file, not shown.")
        try:
            return full_path, data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Binary or non-UTF-8 file, not shown.") from exc

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        try:
            _, text = self._read_text(path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return redact(f"Error: {exc}")

        lines = text.splitlines()
        if not isinstance(start_line, int) or start_line < 1:
            return "Error: start_line must be a positive integer."
        if end_line is not None and (not isinstance(end_line, int) or end_line < start_line):
            return "Error: end_line must be an integer no smaller than start_line."
        normalized = self.normalize_relative_path(path)
        if not lines:
            self.files_read.add(normalized)
            return "[empty file]"
        if start_line > len(lines):
            return f"Error: start_line {start_line} is past the end of the file."

        self.files_read.add(normalized)
        line_cap_end = start_line + MAX_READ_LINES - 1
        requested_end = end_line if end_line is not None else len(lines)
        final_line = min(requested_end, len(lines), line_cap_end)
        selected = lines[start_line - 1 : final_line]
        rendered = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, start_line))
        rendered = redact(rendered)
        if len(rendered) > MAX_READ_OUTPUT:
            rendered = rendered[:MAX_READ_OUTPUT] + "\n[truncated by output size limit]"
        notes = []
        if len(lines) > final_line:
            notes.append(f"file has {len(lines)} total lines")
        if end_line is not None and final_line < end_line:
            notes.append(f"requested through line {end_line}; showing through {final_line}")
        if notes:
            rendered += "\n[" + "; ".join(notes) + "]"
        return rendered

    def search_code(self, pattern: str, file_glob: str = "*") -> str:
        if not isinstance(pattern, str) or not pattern or len(pattern) > 200:
            return "Error: pattern must contain 1 to 200 characters."
        if re.search(r"\([^)]*[+*][^)]*\)[+*{]", pattern):
            return "Error: nested repetition is not allowed in search patterns."
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return f"Invalid regex: {exc}"

        matches: list[str] = []
        match_count = 0
        skipped_large = 0
        files_scanned = 0
        bytes_scanned = 0
        scan_capped = False
        file_entries = self._walk_files()
        for relative, path, size, _ in file_entries:
            if not fnmatch.fnmatchcase(relative, file_glob):
                continue
            if files_scanned >= MAX_SEARCH_FILES or bytes_scanned + size > MAX_SEARCH_TOTAL_BYTES:
                scan_capped = True
                break
            files_scanned += 1
            bytes_scanned += size
            if size > MAX_SEARCH_FILE_BYTES:
                skipped_large += 1
                continue
            try:
                safe_file = self.safe_path(relative)
                if not safe_file.is_file():
                    continue
                data = safe_file.read_bytes()
                if b"\x00" in data:
                    continue
                text = data.decode("utf-8")
            except (OSError, UnicodeDecodeError, ValueError):
                continue

            for line_number, line in enumerate(text.splitlines(), 1):
                for _ in compiled.finditer(line):
                    match_count += 1
                    if len(matches) < MAX_SEARCH_MATCHES:
                        snippet = line.strip()
                        if len(snippet) > 200:
                            snippet = snippet[:197] + "..."
                        matches.append(redact(f"{relative}:{line_number}: {snippet}"))
        if not matches:
            result = "No matches found."
        else:
            result = "\n".join(matches)
        if match_count > MAX_SEARCH_MATCHES:
            result += f"\n[{match_count - MAX_SEARCH_MATCHES} more matches omitted]"
        if skipped_large:
            result += f"\n[skipped {skipped_large} files larger than {MAX_SEARCH_FILE_BYTES} bytes]"
        if scan_capped or self.walk_truncated:
            result += "\n[search scan truncated at the configured file or byte limit]"
        return result

    def git_log(self, timeout_seconds: int = 10) -> str:
        command = ["git", "log", "-n", "30", "--pretty=format:%h|%an|%ad|%s", "--date=short"]
        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"},
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return redact(f"Error: could not read git history: {exc}")
        if result.returncode:
            detail = (result.stderr or "git log failed").strip()
            if "does not have any commits" in detail or "your current branch" in detail:
                return "No commits found."
            return redact(f"Error: {detail[:1000]}")
        return redact(result.stdout or "No commits found.")

    def list_workflows(self) -> str:
        workflows = [
            (relative, path)
            for relative, path, size, _ in self._walk_files()
            if relative.startswith(".github/workflows/")
            and Path(relative).suffix.lower() in {".yml", ".yaml"}
            and size <= MAX_READ_BYTES
        ]
        if not workflows:
            if self.walk_truncated:
                return "No workflows found in the scanned entries; repository scan was capped."
            return "No GitHub Actions workflow files found."

        summaries: list[str] = []
        for relative, path in workflows:
            try:
                text = self.safe_path(relative).read_text(encoding="utf-8")
                data = yaml.safe_load(text) or {}
            except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
                summaries.append(f"{relative}: could not parse YAML ({type(exc).__name__})")
                continue
            if not isinstance(data, dict):
                summaries.append(f"{relative}: workflow root is not a mapping")
                continue

            triggers = data.get("on", data.get(True, {}))
            if isinstance(triggers, dict):
                trigger_names = list(triggers)
            elif isinstance(triggers, list):
                trigger_names = triggers
            elif triggers:
                trigger_names = [str(triggers)]
            else:
                trigger_names = []
            trigger_text = ", ".join(map(str, trigger_names)) or "unspecified"
            lines = [f"Workflow {relative}: triggers=[{trigger_text}]"]
            if "permissions" in data:
                lines.append(f"  permissions: {json.dumps(data['permissions'], sort_keys=True)}")

            jobs = data.get("jobs", {})
            if isinstance(jobs, dict):
                for job_name, job in jobs.items():
                    if not isinstance(job, dict):
                        continue
                    lines.append(f"  job {job_name}:")
                    if "permissions" in job:
                        lines.append(
                            f"    permissions: {json.dumps(job['permissions'], sort_keys=True)}"
                        )
                    steps = job.get("steps", [])
                    if isinstance(steps, list):
                        for step in steps:
                            if not isinstance(step, dict):
                                continue
                            uses = step.get("uses")
                            run = step.get("run")
                            label = str(step.get("name") or uses or run or "unnamed step")
                            flags: list[str] = []
                            if step.get("continue-on-error") is True:
                                flags.append("continue-on-error=true")
                            if step.get("if") is False or str(step.get("if", "")).strip().lower() == "false":
                                flags.append("if=false")
                            if isinstance(run, str) and re.search(r"\|\|\s*true\b", run):
                                flags.append("run contains '|| true'")
                            if isinstance(uses, str):
                                ref = uses.rsplit("@", 1)[-1] if "@" in uses else ""
                                if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
                                    flags.append("action is not pinned to a full commit SHA")
                                if (
                                    "pull_request_target" in trigger_names
                                    and uses.startswith("actions/checkout@")
                                ):
                                    checkout_ref = str(step.get("with", {}).get("ref", ""))
                                    if re.search(
                                        r"github\.event\.pull_request\.(head\.(sha|ref)|head_sha)",
                                        checkout_ref,
                                    ):
                                        flags.append("checks out untrusted pull request head")
                            if isinstance(run, str) and re.search(
                                r"\$\{\{\s*github\.event\.(issue|pull_request|comment)\.[^}]+\}\}",
                                run,
                            ):
                                flags.append("run interpolates untrusted GitHub event data")
                            suffix = f" [flags: {', '.join(flags)}]" if flags else ""
                            lines.append(f"    step: {label[:140]}{suffix}")
            summaries.append("\n".join(lines))
        if self.walk_truncated:
            summaries.append("[workflow search truncated at the repository file scan limit]")
        return redact("\n".join(summaries))


INSPECTION_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List tracked-looking files in the cloned repository with sizes. Use this to understand the project structure and committed artifacts. An optional directory narrows the listing.",
        "parameters": {
            "type": "object",
            "properties": {"directory": {"type": "string", "description": "Optional repository-relative directory."}},
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file from the clone with 1-based line numbers. You must read a file before reporting a finding in it. Results are size-capped and secrets are redacted.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository-relative path."},
                "start_line": {"type": "integer", "description": "First line to show, 1-based."},
                "end_line": {"type": "integer", "description": "Optional inclusive last line."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search repository text files with a regular expression and return redacted path:line snippets. Use a narrow file_glob such as '*.yml' when possible.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression, at most 200 characters."},
                "file_glob": {"type": "string", "description": "Optional path glob; defaults to all files."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "git_log",
        "description": "Read up to 30 recent commit summaries. This never runs repository code.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_workflows",
        "description": "Summarize GitHub Actions workflow triggers, permissions, jobs, steps, and common risky patterns. Read the raw workflow file before reporting a finding so the final line and snippet can be verified.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


class RepositoryTools:
    """Dispatch the supported read-only repository inspection tools."""

    def __init__(self, workspace: RepoWorkspace, timeout_seconds: int = 30):
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            return "Error: tool arguments must be an object."
        handlers = {
            "list_files": lambda: self.workspace.list_files(**arguments),
            "read_file": lambda: self.workspace.read_file(**arguments),
            "search_code": lambda: self.workspace.search_code(**arguments),
            "git_log": lambda: self.workspace.git_log(
                timeout_seconds=self.timeout_seconds, **arguments
            ),
            "list_workflows": lambda: self.workspace.list_workflows(),
        }
        handler = handlers.get(name)
        if handler is None:
            return f"Error: unknown inspection tool: {name}"
        try:
            return redact(str(handler()))
        except TypeError as exc:
            return redact(f"Error: invalid arguments for {name}: {exc}")
