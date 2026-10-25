"""Provider-specific Gemini client and provider-neutral chat data types."""

from __future__ import annotations

import json
import os
import random
import re
import socket
import ssl
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import redact

try:
    import httpx
except ImportError:
    httpx = None

try:
    import httpcore
except ImportError:
    httpcore = None

load_dotenv()

TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# Model defaults:
# Note: older models may eventually be retired by the provider. The router and fallback chain
# tolerate 404/NOT_FOUND errors gracefully by skipping unavailable models and falling back to active ones.
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODELS = "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
DEFAULT_CRITIC_MODEL = "gemini-3.1-flash-lite"

_transient_classes: list[type[BaseException]] = [
    TimeoutError,
    ConnectionError,
    socket.timeout,
    socket.gaierror,
    ssl.SSLError,
]

if httpx is not None:
    _transient_classes.extend(
        [
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.TransportError,
            httpx.ProtocolError,
        ]
    )

if httpcore is not None:
    _transient_classes.extend(
        [
            httpcore.TimeoutException,
            httpcore.NetworkError,
            httpcore.ProtocolError,
        ]
    )

TRANSIENT_EXCEPTION_CLASSES = tuple(set(_transient_classes))


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


@dataclass
class ModelHealth:
    model: str
    available: bool = True
    cooldown_until: float = 0.0
    last_error: str | None = None
    consecutive_429: int = 0


class ModelRouter:
    """Per-model health tracking, cooldown handling, and priority routing."""

    def __init__(self, models: list[str]):
        if not models:
            raise LLMError("At least one model must be provided.")
        seen = set()
        self.models: list[str] = []
        for m in models:
            if m and m not in seen:
                seen.add(m)
                self.models.append(m)
        self.health: dict[str, ModelHealth] = {m: ModelHealth(m) for m in self.models}

    def candidate_models(self, now: float | None = None) -> list[str]:
        current = time.monotonic() if now is None else now
        return [
            m
            for m in self.models
            if self.health[m].available and self.health[m].cooldown_until <= current
        ]

    def mark_success(self, model: str) -> None:
        if model in self.health:
            self.health[model].consecutive_429 = 0
            self.health[model].last_error = None

    def mark_cooldown(
        self,
        model: str,
        delay: float,
        error: Exception | str,
        now: float | None = None,
    ) -> None:
        if model in self.health:
            h = self.health[model]
            h.consecutive_429 += 1
            current = time.monotonic() if now is None else now
            h.cooldown_until = current + delay
            h.last_error = redact(str(error))[:700]

    def mark_unavailable(self, model: str, error: Exception | str) -> None:
        if model in self.health:
            h = self.health[model]
            h.available = False
            h.last_error = redact(str(error))[:700]

    def earliest_cooldown_delay(self, now: float | None = None) -> float | None:
        current = time.monotonic() if now is None else now
        cooling = [
            h.cooldown_until - current
            for h in self.health.values()
            if h.available and h.cooldown_until > current
        ]
        return min(cooling) if cooling else None

    def summary(self) -> str:
        parts = []
        for m in self.models:
            h = self.health[m]
            if not h.available:
                status = "unavailable"
            elif h.cooldown_until > time.monotonic():
                remaining = max(0.0, round(h.cooldown_until - time.monotonic(), 1))
                status = f"cooling down ({remaining}s left)"
            else:
                status = "available"
            err = f": {h.last_error}" if h.last_error else ""
            parts.append(f"{m} ({status}{err})")
        return "; ".join(parts)


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


def create_client() -> genai.Client | Any:
    """Create the provider client without making a network request."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "openai_compatible":
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        return SimpleNamespace(
            provider="openai_compatible",
            api_key=api_key,
            base_url=base_url,
            close=lambda: None,
        )
    if provider != "gemini":
        raise LLMError(f"Unsupported LLM_PROVIDER: {provider}. Supported: 'gemini', 'openai_compatible'.")
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


def _convert_messages(
    messages: list[dict[str, Any]], strip_raw_content: bool = False
) -> list[types.Content]:
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
            if isinstance(raw_content, types.Content) and not strip_raw_content:
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
    for attr in ("code", "status_code"):
        code = getattr(error, attr, None)
        if isinstance(code, int) and code in TRANSIENT_STATUS_CODES:
            return True
    response = getattr(error, "response", None)
    if response is not None:
        for attr in ("status_code", "code"):
            code = getattr(response, attr, None)
            if isinstance(code, int) and code in TRANSIENT_STATUS_CODES:
                return True

    if isinstance(error, TRANSIENT_EXCEPTION_CLASSES):
        return True

    if (
        error.__cause__
        and isinstance(error.__cause__, Exception)
        and _is_transient(error.__cause__)
    ):
        return True
    if (
        error.__context__
        and isinstance(error.__context__, Exception)
        and _is_transient(error.__context__)
    ):
        return True

    error_name = type(error).__name__.lower()
    error_msg = str(error).lower()
    if any(word in error_name for word in ("timeout", "connection", "network", "protocol")):
        return True
    return any(
        phrase in error_msg
        for phrase in (
            "server disconnected",
            "connection reset",
            "connection refused",
            "timed out",
            "remote protocol",
        )
    )


def _is_model_unavailable(error: Exception) -> bool:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if response is not None and code is None:
        code = getattr(response, "status_code", None) or getattr(response, "code", None)

    msg = str(error).lower()
    if code == 404 or "not_found" in msg or "not found" in msg:
        return True

    if code == 400:
        model_specific_markers = (
            "not supported for this model",
            "model does not support",
            "unsupported model",
            "model not available",
            "not available for model",
            "unknown model",
            "invalid model",
            "is not found for api version",
        )
        if any(marker in msg for marker in model_specific_markers):
            return True

    return False


def _is_signature_or_reasoning_error(error: Exception) -> bool:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if response is not None and code is None:
        code = getattr(response, "status_code", None) or getattr(response, "code", None)
    if code != 400:
        return False
    msg = str(error).lower()
    markers = (
        "reasoning",
        "thought",
        "signature",
        "raw_content",
        "unexpected part",
        "invalid part",
        "unsupported part",
        "contents[",
    )
    return any(marker in msg for marker in markers)


def _get_thinking_config() -> types.ThinkingConfig | None:
    level_str = os.getenv("LLM_THINKING_LEVEL", "medium").strip().upper()
    try:
        if hasattr(types, "ThinkingLevel") and hasattr(types, "ThinkingConfig"):
            level_enum = getattr(types.ThinkingLevel, level_str, None)
            if level_enum is not None:
                return types.ThinkingConfig(thinking_level=level_enum)
    except (AttributeError, TypeError, ValueError):
        return None
    return None


def _convert_messages_openai(
    system_prompt: str, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    openai_msgs: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            openai_msgs.append({"role": "user", "content": str(msg.get("content", ""))})
        elif role == "assistant":
            m: dict[str, Any] = {"role": "assistant"}
            if msg.get("content"):
                m["content"] = str(msg["content"])
            if msg.get("tool_calls"):
                m["tool_calls"] = [
                    {
                        "id": c.get("id") or f"call-{i}",
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c.get("args", {})),
                        },
                    }
                    for i, c in enumerate(msg["tool_calls"])
                ]
            openai_msgs.append(m)
        elif role == "tool":
            openai_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id") or "call-0",
                    "content": str(msg.get("content", "")),
                }
            )
    return openai_msgs


def _convert_tools_openai(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = []
    for schema in tool_schemas:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema.get(
                        "parameters",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                },
            }
        )
    return tools


def _chat_openai_compatible(
    system_prompt: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout_ms: int = 60_000,
) -> LLMReply:
    url = (
        base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    ).rstrip("/") + "/chat/completions"
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    chosen_model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload: dict[str, Any] = {
        "model": chosen_model,
        "messages": _convert_messages_openai(system_prompt, messages),
        "temperature": temperature,
    }
    tools = _convert_tools_openai(tool_schemas)
    if tools:
        payload["tools"] = tools

    if httpx is None:
        raise LLMError("httpx is required for OpenAI-compatible provider.")

    with httpx.Client(timeout=timeout_ms / 1000.0) as http_client:
        try:
            resp = http_client.post(url, headers=headers, json=payload)
        except Exception as exc:
            raise LLMError(f"OpenAI-compatible request failed: {redact(str(exc))}") from exc

    if resp.status_code != 200:
        raise LLMError(
            f"OpenAI-compatible request returned status {resp.status_code}: {redact(resp.text)[:500]}"
        )

    data = resp.json()
    choice = data.get("choices", [{}])[0].get("message", {})
    text = choice.get("content") or ""
    tool_calls = []
    for c in choice.get("tool_calls", []):
        fn = c.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(
            ToolCall(
                id=c.get("id", ""),
                name=name,
                args=args,
                provider_id=c.get("id"),
            )
        )

    return LLMReply(
        text=text,
        tool_calls=tool_calls,
        raw=data,
        raw_content=None,
        model_used=chosen_model,
    )


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
    client: genai.Client | Any,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    request_timeout_ms: int | None = None,
    deadline_monotonic: float | None = None,
    router: ModelRouter | None = None,
    trace: list[dict[str, Any]] | None = None,
    step: int = 1,
) -> LLMReply:
    """Ask Gemini for the next response, with manual tool execution and bounded retries."""
    requested_model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if getattr(client, "provider", None) == "openai_compatible" or provider == "openai_compatible":
        return _chat_openai_compatible(
            system_prompt,
            messages,
            tool_schemas,
            model=requested_model,
            temperature=temperature,
            timeout_ms=request_timeout_ms or _positive_int("LLM_TIMEOUT_MS", 60_000),
        )

    max_retries = _positive_int("LLM_MAX_RETRIES", 4, minimum=0)
    timeout_ms = request_timeout_ms or _positive_int("LLM_TIMEOUT_MS", 60_000)
    function_tools = _function_tools(tool_schemas)
    attempts_per_model = max_retries + 1
    attempts_made = 0
    response = None
    last_error: Exception | None = None
    model_used: str | None = None
    model_errors: list[str] = []

    if router is None:
        router = ModelRouter(_model_chain(requested_model))

    candidate_models = router.candidate_models()
    if not candidate_models:
        earliest_delay = router.earliest_cooldown_delay()
        if earliest_delay is not None:
            if deadline_monotonic is not None:
                remaining_total = deadline_monotonic - time.monotonic()
                if remaining_total > earliest_delay + 0.1:
                    time.sleep(earliest_delay)
                    candidate_models = router.candidate_models()
            else:
                time.sleep(earliest_delay)
                candidate_models = router.candidate_models()

    if not candidate_models:
        if os.getenv("LLM_FALLBACK_PROVIDER") == "openai_compatible" and os.getenv("OPENAI_API_KEY"):
            if trace is not None:
                trace.append(
                    {
                        "step": step,
                        "type": "provider_fallback",
                        "from_provider": "gemini",
                        "to_provider": "openai_compatible",
                    }
                )
            return _chat_openai_compatible(
                system_prompt,
                messages,
                tool_schemas,
                model=os.getenv("LLM_MODEL"),
                timeout_ms=timeout_ms,
            )
        raise LLMError(f"All models currently unavailable or cooling down: {router.summary()}")

    contents = _convert_messages(messages)
    for model_index, model_name in enumerate(candidate_models):
        response = None
        for attempt in range(attempts_per_model):
            attempt_timeout_ms = timeout_ms
            if deadline_monotonic is not None:
                remaining_ms = int((deadline_monotonic - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    last_error = TimeoutError("Global audit deadline reached.")
                    break
                attempt_timeout_ms = min(attempt_timeout_ms, remaining_ms)

            config_kwargs: dict[str, Any] = {
                "system_instruction": system_prompt,
                "tools": function_tools,
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
                "temperature": temperature,
                "max_output_tokens": 8192,
                "http_options": types.HttpOptions(
                    timeout=attempt_timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            }
            thinking_cfg = _get_thinking_config()
            if thinking_cfg is not None:
                config_kwargs["thinking_config"] = thinking_cfg

            config = types.GenerateContentConfig(**config_kwargs)
            try:
                attempts_made += 1
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                model_used = model_name
                router.mark_success(model_name)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc

                # Mid-conversation switch: retry once with stripped raw content
                if _is_signature_or_reasoning_error(exc):
                    clean_contents = _convert_messages(messages, strip_raw_content=True)
                    try:
                        attempts_made += 1
                        response = client.models.generate_content(
                            model=model_name,
                            contents=clean_contents,
                            config=config,
                        )
                        model_used = model_name
                        router.mark_success(model_name)
                        break
                    except Exception as retry_exc:  # noqa: BLE001
                        last_error = retry_exc
                        exc = retry_exc

                # Gracefully retry without thinking_config if unsupported
                if "thinking_config" in config_kwargs and "thinking" in str(exc).lower():
                    del config_kwargs["thinking_config"]
                    try:
                        config = types.GenerateContentConfig(**config_kwargs)
                        attempts_made += 1
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=config,
                        )
                        model_used = model_name
                        router.mark_success(model_name)
                        break
                    except Exception as thinking_retry_exc:  # noqa: BLE001
                        last_error = thinking_retry_exc
                        exc = thinking_retry_exc

                if _is_model_unavailable(exc):
                    router.mark_unavailable(model_name, exc)
                    break

                if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
                    consec = router.health[model_name].consecutive_429
                    cooldown_delay = _provider_retry_after(exc) or min(
                        60.0, float(2**consec * 5.0)
                    )
                    router.mark_cooldown(model_name, cooldown_delay, exc)
                    if trace is not None:
                        trace.append(
                            {
                                "step": step,
                                "type": "model_cooldown",
                                "model": model_name,
                                "duration_seconds": cooldown_delay,
                                "reason": redact(str(exc))[:500],
                            }
                        )
                    break

                if not _is_transient(exc):
                    router.mark_unavailable(model_name, exc)
                    break

                if attempt + 1 >= attempts_per_model:
                    break

                delay = _retry_delay(exc, attempt)
                if trace is not None:
                    trace.append(
                        {
                            "step": step,
                            "type": "model_retry",
                            "model": model_name,
                            "attempt": attempt + 1,
                            "delay": delay,
                            "reason": redact(str(exc))[:500],
                        }
                    )
                if deadline_monotonic is not None:
                    remaining = deadline_monotonic - time.monotonic()
                    if remaining <= delay:
                        break
                    delay = min(delay, remaining)
                time.sleep(delay)

        if response is not None:
            break
        if last_error is not None:
            unavailable_note = (
                " (model unavailable)" if _is_model_unavailable(last_error) else ""
            )
            model_errors.append(
                f"{model_name}: {redact(str(last_error))[:700]}{unavailable_note}"
            )
        if (
            model_index + 1 >= len(candidate_models)
            or last_error is None
            or not (_is_transient(last_error) or _is_model_unavailable(last_error))
        ):
            break

    if response is None:
        if os.getenv("LLM_FALLBACK_PROVIDER") == "openai_compatible" and os.getenv("OPENAI_API_KEY"):
            if trace is not None:
                trace.append(
                    {
                        "step": step,
                        "type": "provider_fallback",
                        "from_provider": "gemini",
                        "to_provider": "openai_compatible",
                    }
                )
            return _chat_openai_compatible(
                system_prompt,
                messages,
                tool_schemas,
                model=os.getenv("LLM_MODEL"),
                timeout_ms=timeout_ms,
            )
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
