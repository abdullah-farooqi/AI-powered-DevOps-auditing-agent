from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import types

import llm
from llm import LLMError, chat


def _response_with_call():
    content = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    id="call-1",
                    name="read_file",
                    args={"path": "README.md"},
                )
            )
        ],
    )
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def _response_with_text(text: str = "done"):
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def test_chat_uses_manual_function_calling_and_parses_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")

    class ModelAPI:
        config = None

        def generate_content(self, **kwargs):
            self.config = kwargs["config"]
            return _response_with_call()

    models = ModelAPI()
    reply = chat(
        SimpleNamespace(models=models),
        "system prompt",
        [{"role": "user", "content": "Read README."}],
        [
            {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    )
    assert models.config.automatic_function_calling.disable is True
    assert models.config.tools[0].function_declarations[0].name == "read_file"
    assert reply.tool_calls[0].name == "read_file"
    assert reply.tool_calls[0].args == {"path": "README.md"}
    assert reply.tool_calls[0].provider_id == "call-1"


def test_convert_messages_preserves_model_content_and_function_response() -> None:
    raw_model_content = types.Content(
        role="model",
        parts=[
            types.Part.from_function_call(name="read_file", args={"path": "README.md"})
        ],
    )
    converted = llm._convert_messages(
        [
            {"role": "user", "content": "Read it."},
            {"role": "assistant", "raw_content": raw_model_content},
            {
                "role": "tool",
                "name": "read_file",
                "tool_call_id": "call-1",
                "content": "1: # README",
            },
        ]
    )
    assert converted[0].role == "user"
    assert converted[1] is raw_model_content
    assert converted[2].role == "user"
    assert converted[2].parts[0].function_response.name == "read_file"
    assert converted[2].parts[0].function_response.id == "call-1"
    assert converted[2].parts[0].function_response.response == {"result": "1: # README"}


def test_chat_retries_transient_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    class BusyError(Exception):
        code = 503

    class ModelAPI:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise BusyError("temporarily unavailable")
            return _response_with_text("recovered")

    models = ModelAPI()
    reply = chat(
        SimpleNamespace(models=models),
        "system prompt",
        [{"role": "user", "content": "continue"}],
        [],
    )
    assert models.calls == 2
    assert sleeps
    assert reply.text == "recovered"


def test_retry_delay_honors_provider_retry_info() -> None:
    error = Exception("quota")
    error.details = {
        "error": {
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "56.5s",
                }
            ]
        }
    }
    assert llm._retry_delay(error, 0) == 56.5


def test_chat_falls_back_immediately_on_quota_and_reuses_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "4")
    monkeypatch.setenv(
        "LLM_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    )
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    class DailyQuotaError(Exception):
        def __init__(self):
            super().__init__("daily quota exhausted")
            self.code = 429
            self.details = {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "70774s",
                        }
                    ],
                }
            }

    class ModelAPI:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["model"] == "gemini-3.6-flash":
                raise DailyQuotaError()
            return _response_with_text("resumed")

    models = ModelAPI()
    reply = chat(
        SimpleNamespace(models=models),
        "system prompt",
        [{"role": "user", "content": "continue the audit"}],
        [],
    )

    assert [call["model"] for call in models.calls] == [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ]
    assert models.calls[0]["contents"] is models.calls[1]["contents"]
    assert reply.model_used == "gemini-3.5-flash-lite"
    assert reply.text == "resumed"
    assert sleeps == []


def test_chat_does_not_retry_non_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")

    class BadRequest(Exception):
        code = 400

    class ModelAPI:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise BadRequest("invalid request")

    models = ModelAPI()
    with pytest.raises(LLMError, match=r"after 1 attempt\(s\).*invalid request"):
        chat(
            SimpleNamespace(models=models),
            "system prompt",
            [{"role": "user", "content": "bad"}],
            [],
        )
    assert models.calls == 1


def test_create_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY is missing"):
        llm.create_client()
