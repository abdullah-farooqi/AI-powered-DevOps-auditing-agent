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
            trigger_names = []
            trigger_details = []
            if isinstance(triggers, dict):
                trigger_names = list(triggers)
                for tr_name, tr_cfg in triggers.items():
                    if isinstance(tr_cfg, dict):
                        detail_parts = []
                        if "branches" in tr_cfg:
                            detail_parts.append(f"branches={tr_cfg['branches']}")
                        if "paths" in tr_cfg:
                            detail_parts.append(f"paths={tr_cfg['paths']}")
                        if detail_parts:
                            trigger_details.append(f"{tr_name} ({', '.join(detail_parts)})")
                        else:
                            trigger_details.append(str(tr_name))
                    else:
                        trigger_details.append(str(tr_name))
            elif isinstance(triggers, list):
                trigger_names = triggers
                trigger_details = [str(t) for t in triggers]
            elif triggers:
                trigger_names = [str(triggers)]
                trigger_details = [str(triggers)]

            trigger_text = ", ".join(trigger_details) or "unspecified"
            workflow_flags = []
            if "workflow_run" in trigger_names:
                workflow_flags.append("triggers on workflow_run (privileged execution in default branch context)")

            wf_header = f"Workflow {relative}: triggers=[{trigger_text}]"
            if workflow_flags:
                wf_header += f" [flags: {', '.join(workflow_flags)}]"
            lines = [wf_header]

            top_permissions = data.get("permissions")
            if top_permissions is not None:
                lines.append(f"  permissions: {json.dumps(top_permissions, sort_keys=True)}")
                if top_permissions == "write-all":
                    lines.append("  [flag: top-level permissions: write-all]")

            jobs = data.get("jobs", {})
            test_job_names = set()
            if isinstance(jobs, dict):
                for j_name in jobs:
                    if any(t in j_name.lower() for t in ("test", "check", "lint", "build", "verify")):
                        test_job_names.add(j_name)

                for job_name, job in jobs.items():
                    if not isinstance(job, dict):
                        continue
                    job_flags = []
                    job_if = job.get("if")
                    if job_if is False or str(job_if).strip().lower() == "false":
                        job_flags.append("job if: false (job never runs)")

                    needs = job.get("needs")
                    needs_list = [needs] if isinstance(needs, str) else (needs or [])
                    is_deploy_job = any(k in job_name.lower() for k in ("deploy", "release", "publish"))
                    if is_deploy_job and test_job_names and not any(dep in test_job_names for dep in needs_list):
                        job_flags.append("deploy or release job has no dependency on test jobs")

                    job_permissions = job.get("permissions")
                    if job_permissions == "write-all":
                        job_flags.append("permissions: write-all")
                    elif isinstance(job_permissions, dict):
                        write_scopes = [k for k, v in job_permissions.items() if v == "write"]
                        if write_scopes:
                            job_flags.append(f"GITHUB_TOKEN write scopes: {', '.join(write_scopes)}")

                    runs_on = str(job.get("runs-on", ""))
                    if "self-hosted" in runs_on and any("pull_request" in str(t) for t in trigger_names):
                        job_flags.append("self-hosted runner on pull_request")

                    job_suffix = f" [flags: {', '.join(job_flags)}]" if job_flags else ""
                    lines.append(f"  job {job_name}:{job_suffix}")
                    if needs_list:
                        lines.append(f"    needs: {needs_list}")
                    if job_permissions is not None:
                        lines.append(f"    permissions: {json.dumps(job_permissions, sort_keys=True)}")

                    steps = job.get("steps", [])
                    has_secrets_step = False
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
                            if isinstance(run, str):
                                if re.search(r"\|\|\s*true\b", run):
                                    flags.append("run contains '|| true'")
                                if re.search(r"curl\s+[^|]+\|\s*(ba|z)?sh\b", run) or re.search(r"wget\s+[^|]+\|\s*(ba|z)?sh\b", run):
                                    flags.append("curl | sh execution")
                                if re.search(r"pip\s+install\s+.*git\+[a-zA-Z0-9]+://[^@\s]+", run):
                                    flags.append("pip install of unpinned git URL")
                                if re.search(r"\$\{\{\s*secrets\.[^}]+\}\}", run):
                                    has_secrets_step = True
                                    flags.append("secrets used directly inside run command")
                                if re.search(
                                    r"\$\{\{\s*github\.event\.(issue|pull_request|comment)\.[^}]+\}\}",
                                    run,
                                ):
                                    flags.append("run interpolates untrusted GitHub event data")

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
                                    if "persist-credentials" not in step.get("with", {}):
                                        flags.append("actions/checkout with persist-credentials default in privileged workflow")

                            suffix = f" [flags: {', '.join(flags)}]" if flags else ""
                            lines.append(f"    step: {label[:140]}{suffix}")

                    if has_secrets_step and job_permissions is None and top_permissions is None:
                        lines.append("    [warning: job uses secrets with default/missing permissions]")

            summaries.append("\n".join(lines))
        if self.walk_truncated:
            summaries.append("[workflow search truncated at the repository file scan limit]")
        return redact("\n".join(summaries))

    def scan_secrets(self, directory: str = "") -> str:
        """Scan repository text files for common credential formats and return path:line: rule_name ONLY."""
        normalized_dir = ""
        if directory and directory.strip() not in ("", ".", "./"):
            try:
                base = self.safe_path(directory)
                if not base.is_dir():
                    return f"Error: not a directory: {directory}"
                normalized_dir = self.normalize_relative_path(directory).rstrip("/") + "/"
            except ValueError as exc:
                return f"Error: {exc}"

        secret_rules: list[tuple[str, re.Pattern]] = [
            ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
            ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
            ("openai_or_bearer_token", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
            ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
            ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
            ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
            (
                "connection_string_with_password",
                re.compile(r"[a-zA-Z0-9+.-]+://[^:\s'\"]+:[^@\s'\"]{4,}@[^/\s'\"]+"),
            ),
            (
                "generic_credential_assignment",
                re.compile(
                    r"(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[=:]\s*(['\"])[^\s'\"]{8,}\1"
                ),
            ),
        ]

        matches: list[str] = []
        files_scanned = 0
        file_entries = self._walk_files()
        for relative, path, size, is_symlink in file_entries:
            if normalized_dir and not relative.startswith(normalized_dir):
                continue
            if size > MAX_SEARCH_FILE_BYTES or is_symlink:
                continue
            if files_scanned >= MAX_SEARCH_FILES:
                break
            files_scanned += 1

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

            for line_idx, line in enumerate(text.splitlines(), start=1):
                for rule_name, pattern in secret_rules:
                    if pattern.search(line):
                        matches.append(f"{relative}:{line_idx}: {rule_name}")
                        if len(matches) >= 100:
                            break
                if len(matches) >= 100:
                    break
            if len(matches) >= 100:
                break

        if not matches:
            return "No suspicious credentials found."
        output = "\n".join(matches)
        if len(matches) >= 100:
            output += "\n[scan capped at 100 credential detections]"
        return output

    def list_suspicious_files(self) -> str:
        """List committed artifacts that normally should not be tracked."""
        suspicious: list[str] = []
        ignored_env_samples = {".env.example", ".env.sample", ".env.template"}

        for relative, path, size, is_symlink in self._walk_files():
            filename = relative.rsplit("/", 1)[-1].casefold()
            parts = [p.casefold() for p in relative.split("/")]

            if filename.startswith(".env") and filename not in ignored_env_samples:
                suspicious.append(f"{relative} ({size} bytes): committed environment configuration")
                continue

            if filename.endswith((".pyc", ".pyo", ".pyd")) or "__pycache__" in parts:
                suspicious.append(f"{relative} ({size} bytes): compiled Python bytecode")
                continue

            if filename in {".ds_store", "thumbs.db"}:
                suspicious.append(f"{relative} ({size} bytes): operating system metadata")
                continue

            if "node_modules" in parts:
                suspicious.append(f"{relative} ({size} bytes): committed node_modules dependency")
                continue

            if ".idea" in parts or (len(parts) > 1 and parts[0] == ".vscode" and filename.endswith(".json")):
                suspicious.append(f"{relative} ({size} bytes): committed IDE settings")
                continue

            if filename.endswith(".log"):
                suspicious.append(f"{relative} ({size} bytes): log file")
                continue

            if any(part in {"dist", "build"} for part in parts[:-1]) or filename.endswith(".egg-info"):
                suspicious.append(f"{relative} ({size} bytes): committed build or distribution artifact")
                continue

            if filename in {".coverage", "coverage.xml"} or any(part in {"htmlcov", ".pytest_cache"} for part in parts):
                suspicious.append(f"{relative} ({size} bytes): test coverage / cache artifact")
                continue

            if filename.endswith((".sqlite", ".sqlite3", ".db")):
                suspicious.append(f"{relative} ({size} bytes): local SQLite or database file")
                continue

            if filename.endswith((".pem", ".key", ".pkcs12", ".pfx")) or filename in {"id_rsa", "id_ed25519", "id_dsa"}:
                suspicious.append(f"{relative} ({size} bytes): cryptographic private key or certificate file")
                continue

            if size > 10 * 1024 * 1024 or filename.endswith((".exe", ".dll", ".so", ".bin")):
                suspicious.append(f"{relative} ({size} bytes): binary or oversized file")
                continue

        if not suspicious:
            return "No suspicious files found."
        return "\n".join(suspicious[:200])

    def parse_manifest(self, path: str) -> str:
        """Parse dependency manifests and report pinned/range/unpinned status and lockfile presence."""
        try:
            full_path = self.safe_path(path)
            normalized_path = self.normalize_relative_path(path)
        except ValueError as exc:
            return f"Error: {exc}"

        if not full_path.is_file():
            return f"Error: manifest file does not exist: {normalized_path}"

        try:
            content = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return f"Error: cannot read {normalized_path} as UTF-8 text."

        self.files_read.add(normalized_path)
        filename = full_path.name.casefold()
        parent = full_path.parent

        lockfiles = []
        dependencies: list[tuple[str, str, str]] = []

        def classify_version(spec: str) -> str:
            spec = spec.strip()
            if not spec or spec == "*":
                return "wildcard" if spec == "*" else "unpinned"
            if spec.startswith(("==", "@")) or re.fullmatch(r"v?\d+(\.\d+)*", spec):
                return "pinned"
            if any(spec.startswith(op) for op in (">=", "<=", "~=", "^", ">", "<", "!=")):
                return "range"
            if "*" in spec or "latest" in spec.lower():
                return "wildcard"
            return "pinned"

        if filename == "requirements.txt" or (filename.startswith("requirements") and filename.endswith(".txt")):
            for candidate in ("requirements.lock", "requirements-lock.txt", "Pipfile.lock", "poetry.lock"):
                if (parent / candidate).is_file():
                    lockfiles.append(candidate)
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "-r", "-i", "--")):
                    continue
                m = re.match(r"^([A-Za-z0-9_.-]+)\s*([~=<>!@*].*)?$", line)
                if m:
                    pkg, spec = m.group(1), (m.group(2) or "").strip()
                    dependencies.append((pkg, spec, classify_version(spec)))

        elif filename == "package.json":
            for candidate in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb"):
                if (parent / candidate).is_file():
                    lockfiles.append(candidate)
            try:
                pkg_data = json.loads(content)
                for section in ("dependencies", "devDependencies", "peerDependencies"):
                    deps = pkg_data.get(section, {})
                    if isinstance(deps, dict):
                        for pkg, spec in deps.items():
                            dependencies.append((f"{pkg} ({section})", str(spec), classify_version(str(spec))))
            except json.JSONDecodeError:
                return f"Error: invalid JSON in {normalized_path}"

        elif filename == "pyproject.toml":
            for candidate in ("poetry.lock", "uv.lock", "pdm.lock"):
                if (parent / candidate).is_file():
                    lockfiles.append(candidate)
            in_deps = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("["):
                    in_deps = any(k in stripped for k in ("dependencies", "project.dependencies"))
                    continue
                if in_deps and stripped and not stripped.startswith("#"):
                    m = re.match(r'^["\']?([A-Za-z0-9_.-]+)\s*([~=<>!@^].*?)?["\']?,?$', stripped)
                    if m:
                        pkg, spec = m.group(1), (m.group(2) or "").strip()
                        dependencies.append((pkg, spec, classify_version(spec)))

        elif filename == "pipfile":
            if (parent / "Pipfile.lock").is_file():
                lockfiles.append("Pipfile.lock")
            for line in content.splitlines():
                m = re.match(r'^([A-Za-z0-9_.-]+)\s*=\s*["\']([^"\']+)["\']', line.strip())
                if m:
                    dependencies.append((m.group(1), m.group(2), classify_version(m.group(2))))

        elif filename == "cargo.toml":
            if (parent / "Cargo.lock").is_file():
                lockfiles.append("Cargo.lock")
            for line in content.splitlines():
                m = re.match(r'^([A-Za-z0-9_.-]+)\s*=\s*(?:\{.*version\s*=\s*["\']([^"\']+)["\']|["\']([^"\']+)["\'])', line.strip())
                if m:
                    spec = m.group(2) or m.group(3) or ""
                    dependencies.append((m.group(1), spec, classify_version(spec)))

        elif filename == "go.mod":
            if (parent / "go.sum").is_file():
                lockfiles.append("go.sum")
            for line in content.splitlines():
                m = re.match(r'^\s*([A-Za-z0-9_.\-\/]+)\s+(v[0-9A-Za-z_.\-]+)', line)
                if m:
                    dependencies.append((m.group(1), m.group(2), classify_version(m.group(2))))

        elif filename == "composer.json":
            if (parent / "composer.lock").is_file():
                lockfiles.append("composer.lock")
            try:
                comp_data = json.loads(content)
                for section in ("require", "require-dev"):
                    for pkg, spec in comp_data.get(section, {}).items():
                        dependencies.append((f"{pkg} ({section})", str(spec), classify_version(str(spec))))
            except json.JSONDecodeError:
                return f"Error: invalid JSON in {normalized_path}"
        else:
            return f"Unsupported manifest type: {filename}. Supported: requirements.txt, pyproject.toml, package.json, Pipfile, go.mod, Cargo.toml, composer.json."

        lockfile_status = f"Lockfile present: {', '.join(lockfiles)}" if lockfiles else "No lockfile found in directory."
        lines = [f"Manifest: {normalized_path}", lockfile_status, f"Total dependencies: {len(dependencies)}"]
        for pkg, spec, cls in dependencies[:100]:
            lines.append(f"  - {pkg}: {spec or '[none]'} ({cls})")
        if len(dependencies) > 100:
            lines.append(f"  [and {len(dependencies) - 100} more dependencies]")
        return "\n".join(lines)

    def git_show(self, commit: str, path: str = "", timeout_seconds: int = 10) -> str:
        """Inspect a specific commit or file at that commit in read-only mode."""
        if not re.fullmatch(r"^[0-9a-fA-F]{4,40}$", commit):
            return "Error: commit must be a valid hex commit hash (4 to 40 characters)."

        command = ["git", "show", "--no-patch", "--format=%h|%an|%ad|%s", commit]
        if path:
            try:
                _ = self.safe_path(path)
                norm_path = self.normalize_relative_path(path)
            except ValueError as exc:
                return f"Error: {exc}"
            command = ["git", "show", f"{commit}:{norm_path}"]

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
            return redact(f"Error: could not run git show: {exc}")

        if result.returncode:
            detail = (result.stderr or "git show failed").strip()
            return redact(f"Error: {detail[:1000]}")
        output = result.stdout or ""
        if len(output) > MAX_READ_OUTPUT:
            output = output[:MAX_READ_OUTPUT] + "\n[truncated by output size limit]"
        return redact(output if output else "[empty output]")

    def git_history_search(self, pattern: str, timeout_seconds: int = 10) -> str:
        """Search commit history diffs for a string/regex (read-only)."""
        if not isinstance(pattern, str) or not pattern or len(pattern) > 200:
            return "Error: pattern must contain 1 to 200 characters."

        command = [
            "git",
            "log",
            "-S",
            pattern,
            "-n",
            "10",
            "--pretty=format:%h|%an|%ad|%s",
            "--date=short",
        ]
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
            return redact(f"Error: could not search git history: {exc}")

        if result.returncode:
            detail = (result.stderr or "git log -S failed").strip()
            return redact(f"Error: {detail[:1000]}")
        return redact(result.stdout or "No commits found touching this pattern.")

    def check_equivalents(self) -> str:
        """Report existing tools and configurations to prevent false absence findings."""
        files = [rel.casefold() for rel, _, _, _ in self._walk_files()]

        def has_any(patterns: list[str]) -> list[str]:
            found = []
            for p in patterns:
                for f in files:
                    if fnmatch.fnmatchcase(f, p.casefold()):
                        found.append(f)
                        break
            return found

        linters = has_any(
            [
                "ruff.toml",
                ".ruff.toml",
                "pyproject.toml",
                ".flake8",
                ".pylintrc",
                ".eslintrc*",
                "eslint.config.*",
                ".prettierrc*",
                "prettier.config.*",
                ".pre-commit-config.yaml",
                "biome.json",
            ]
        )
        workflows = self.list_workflows().lower()
        if "lint" in workflows or "ruff" in workflows or "eslint" in workflows:
            linters.append("CI lint step")

        test_configs = has_any(
            [
                "pytest.ini",
                "setup.cfg",
                "conftest.py",
                "jest.config.*",
                "vitest.config.*",
                "karma.conf.js",
                ".rspec",
            ]
        )
        lockfiles = has_any(
            [
                "poetry.lock",
                "uv.lock",
                "pdm.lock",
                "pipfile.lock",
                "requirements.lock",
                "package-lock.json",
                "yarn.lock",
                "pnpm-lock.yaml",
                "bun.lockb",
                "cargo.lock",
                "go.sum",
                "composer.lock",
            ]
        )
        dep_automation = has_any(
            [
                ".github/dependabot.yml",
                ".github/dependabot.yaml",
                ".github/renovate.json",
                ".github/renovate.json5",
                "renovate.json",
            ]
        )
        ownership = has_any(
            [
                ".github/codeowners",
                "codeowners",
                "docs/codeowners",
                ".github/pull_request_template.md",
                ".github/pull_request_template/*",
            ]
        )
        license_files = has_any(["license*", "copying*"])
        docker_files = has_any(["dockerfile*", "docker/dockerfile*", "containerfile*"])

        report = {
            "linter_or_formatter": linters or "None detected",
            "test_runner_config": test_configs or "None detected",
            "ecosystem_lockfile": lockfiles or "None detected",
            "dependency_automation": dep_automation or "None detected",
            "ownership_or_pr_templates": ownership or "None detected",
            "license": license_files or "None detected",
            "container_build": docker_files or "None detected",
        }
        lines = ["Repository Equivalents Report (use this to avoid reporting missing optional files):"]
        for key, val in report.items():
            val_str = ", ".join(val) if isinstance(val, list) else val
            lines.append(f"  {key}: {val_str}")
        return "\n".join(lines)


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
    {
        "name": "scan_secrets",
        "description": "Regex scan repository files for common credential formats (AWS, GitHub, Google, Slack, OpenAI tokens, private keys, connection strings). Returns path:line: rule_name ONLY (never secret values).",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Optional repository-relative directory to narrow scan."}
            },
            "required": [],
        },
    },
    {
        "name": "list_suspicious_files",
        "description": "List committed artifacts that normally should not be tracked, such as .env files, caches, .DS_Store, node_modules, IDE settings, logs, build outputs, coverage, and key files.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "parse_manifest",
        "description": "Parse a dependency manifest (requirements.txt, pyproject.toml, package.json, Pipfile, go.mod, Cargo.toml, composer.json) to list dependencies with pinned/range/unpinned/wildcard classification and note whether a lockfile exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository-relative path to the manifest file."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "git_show",
        "description": "Read-only inspection of a specific commit or file at that commit (e.g. to inspect history or deleted files).",
        "parameters": {
            "type": "object",
            "properties": {
                "commit": {"type": "string", "description": "Hex commit hash (4 to 40 characters)."},
                "path": {"type": "string", "description": "Optional repository-relative file path."}
            },
            "required": ["commit"],
        },
    },
    {
        "name": "git_history_search",
        "description": "Read-only search of commit history diffs for a pattern to find code, configuration, or credentials added or removed in past commits.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "String or regex pattern to search in git commit history."}
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "check_equivalents",
        "description": "Report existing linting, formatting, test runners, lockfiles, dependency automation (Dependabot), ownership, license, and container configurations to prevent false absence findings.",
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
            "scan_secrets": lambda: self.workspace.scan_secrets(**arguments),
            "list_suspicious_files": lambda: self.workspace.list_suspicious_files(),
            "parse_manifest": lambda: self.workspace.parse_manifest(**arguments),
            "git_show": lambda: self.workspace.git_show(
                timeout_seconds=self.timeout_seconds, **arguments
            ),
            "git_history_search": lambda: self.workspace.git_history_search(
                timeout_seconds=self.timeout_seconds, **arguments
            ),
            "check_equivalents": lambda: self.workspace.check_equivalents(),
        }
        handler = handlers.get(name)
        if handler is None:
            return f"Error: unknown inspection tool: {name}"
        try:
            return redact(str(handler()))
        except TypeError as exc:
            return redact(f"Error: invalid arguments for {name}: {exc}")
