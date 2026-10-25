"""Provider-specific Gemini client and provider-neutral chat data types."""

from __future__ import annotations

import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import redact

load_dotenv()

TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_FALLBACK_MODELS = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"


class LLMError(RuntimeError):
    """Raised when a provider call cannot be completed."""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    provider_id: str | None = None


@dataclass
class LLMReply:
    text: str
    tool_calls: list[ToolCall]
    raw: Any
    raw_content: Any
    model_used: str | None = None


def _positive_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise LLMError(f"{name} must be an integer.") from exc
    if parsed < minimum:
        raise LLMError(f"{name} must be at least {minimum}.")
    return parsed


def create_client() -> genai.Client:
    """Create the Gemini client without making a network request."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider != "gemini":
        raise LLMError(f"Unsupported LLM_PROVIDER: {provider}. Only 'gemini' is configured.")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is missing. Add it to the local .env file.")
    timeout_ms = _positive_int("LLM_TIMEOUT_MS", 60_000)
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _function_tools(tool_schemas: list[dict[str, Any]]) -> list[types.Tool]:
    declarations = []
    for schema in tool_schemas:
        declarations.append(
            types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters_json_schema=schema.get(
                    "parameters",
                    {"type": "object", "properties": {}, "required": []},
                ),
            )
        )
    return [types.Tool(function_declarations=declarations)] if declarations else []


def _convert_messages(messages: list[dict[str, Any]]) -> list[types.Content]:
    """Convert neutral conversation records to Gemini Content objects."""
    contents: list[types.Content] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=str(message.get("content", "")))],
                )
            )
            index += 1
            continue

        if role == "assistant":
            raw_content = message.get("raw_content")
            if isinstance(raw_content, types.Content):
                contents.append(raw_content)
            else:
                parts = []
                if message.get("content"):
                    parts.append(types.Part.from_text(text=str(message["content"])))
                for call in message.get("tool_calls", []):
                    parts.append(
                        types.Part.from_function_call(
                            name=call["name"], args=call.get("args", {})
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
            index += 1
            continue

        if role == "tool":
            tool_parts = []
            while index < len(messages) and messages[index].get("role") == "tool":
                tool_message = messages[index]
                response_args: dict[str, Any] = {
                    "name": tool_message["name"],
                    "response": {"result": str(tool_message.get("content", ""))},
                }
                if tool_message.get("tool_call_id"):
                    response_args["id"] = tool_message["tool_call_id"]
                tool_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(**response_args)
                    )
                )
                index += 1
            contents.append(types.Content(role="user", parts=tool_parts))
            continue

        raise LLMError(f"Unsupported conversation role: {role!r}.")
    return contents


def _is_transient(error: Exception) -> bool:
    status_code = getattr(error, "code", None)
    if status_code in TRANSIENT_STATUS_CODES:
        return True
    error_text = f"{type(error).__name__} {error}".lower()
    transient_words = (
        "timeout",
        "connection",
        "network",
        "disconnect",
        "protocol",
        "reset",
        "transport",
        "broken pipe",
        "remote end closed",
        "eof",
        "temporarily unavailable",
        "service unavailable",
        "gateway timeout",
        "resource exhausted",
        "rate limit",
        "too many requests",
    )
    return any(word in error_text for word in transient_words)


def _provider_retry_after(error: Exception) -> float | None:
    """Read a retry delay included in a Gemini quota or transient-error response."""
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        error_body = details.get("error", {})
        provider_details = error_body.get("details", []) if isinstance(error_body, dict) else []
        if isinstance(provider_details, list):
            for detail in provider_details:
                if not isinstance(detail, dict):
                    continue
                if str(detail.get("@type", "")).endswith("RetryInfo"):
                    delay = detail.get("retryDelay", "")
                    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", str(delay))
                    if match:
                        return max(0.0, float(match.group(1)))

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    if headers:
        raw_delay = headers.get("retry-after") or headers.get("Retry-After")
        try:
            if raw_delay is not None:
                return max(0.0, float(raw_delay))
        except (TypeError, ValueError):
            pass
    return None


def _retry_delay(error: Exception, attempt: int) -> float:
    ceiling = min(16.0, 2.0**attempt)
    backoff = random.uniform(ceiling * 0.5, ceiling * 1.5)
    provider_delay = _provider_retry_after(error)
    return max(backoff, provider_delay or 0.0)


def _model_chain(requested_model: str) -> list[str]:
    """Return the active model followed by configured fallbacks, without duplicates."""
    configured = os.getenv("LLM_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS)
    models = [requested_model]
    for fallback in configured.split(","):
        candidate = fallback.strip()
        if candidate and candidate not in models:
            models.append(candidate)
    return models


def chat(
    client: genai.Client,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    *,
    model: str | None = None,
    request_timeout_ms: int | None = None,
    deadline_monotonic: float | None = None,
) -> LLMReply:
    """Ask Gemini for the next response, with manual tool execution and bounded retries."""
    requested_model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()
    if not requested_model:
        raise LLMError("LLM_MODEL cannot be empty.")
    max_retries = _positive_int("LLM_MAX_RETRIES", 4, minimum=0)
    timeout_ms = request_timeout_ms or _positive_int("LLM_TIMEOUT_MS", 60_000)
    function_tools = _function_tools(tool_schemas)
    contents = _convert_messages(messages)
    attempts_per_model = max_retries + 1
    attempts_made = 0
    response = None
    last_error: Exception | None = None
    model_used: str | None = None
    model_errors: list[str] = []

    for model_index, model_name in enumerate(_model_chain(requested_model)):
        response = None
        for attempt in range(attempts_per_model):
            attempt_timeout_ms = timeout_ms
            if deadline_monotonic is not None:
                remaining_ms = int((deadline_monotonic - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    last_error = TimeoutError("Global audit deadline reached.")
                    break
                attempt_timeout_ms = min(attempt_timeout_ms, remaining_ms)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=function_tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0.1,
                max_output_tokens=8192,
                http_options=types.HttpOptions(
                    timeout=attempt_timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            try:
                attempts_made += 1
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                model_used = model_name
                break
            except Exception as exc:  # noqa: BLE001 - the SDK raises different transient transport types.
                last_error = exc
                # A 429 can carry a daily reset delay many hours long. Move to the
                # next model immediately instead of sleeping past this audit's deadline.
                if getattr(exc, "code", None) == 429:
                    break
                if not _is_transient(exc) or attempt + 1 >= attempts_per_model:
                    break
                delay = _retry_delay(exc, attempt)
                if deadline_monotonic is not None:
                    remaining = deadline_monotonic - time.monotonic()
                    if remaining <= delay:
                        break
                    delay = min(delay, remaining)
                time.sleep(delay)

        if response is not None:
            break
        if last_error is not None:
            model_errors.append(f"{model_name}: {redact(str(last_error))[:700]}")
        if (
            model_index + 1 >= len(_model_chain(requested_model))
            or last_error is None
            or not _is_transient(last_error)
        ):
            break

    if response is None:
        error_text = "; ".join(model_errors) or redact(
            str(last_error) if last_error else "unknown provider error"
        )
        raise LLMError(
            f"Gemini request failed after {attempts_made} attempt(s): {error_text}"
        ) from last_error

    candidates = response.candidates or []
    if not candidates or candidates[0].content is None:
        raise LLMError("Gemini returned no candidate content.")
    raw_content = candidates[0].content
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for part_index, part in enumerate(raw_content.parts or []):
        if part.text:
            text_parts.append(part.text)
        function_call = part.function_call
        if function_call is not None:
            args = function_call.args or {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                ToolCall(
                    id=function_call.id or f"call-{uuid.uuid4().hex}-{part_index}",
                    name=function_call.name or "",
                    args=args,
                    provider_id=function_call.id,
                )
            )
    return LLMReply(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        raw=response,
        raw_content=raw_content,
        model_used=model_used,
    )
