from __future__ import annotations

import socket
import ssl
from types import SimpleNamespace

import httpcore
import httpx
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
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv(
        "LLM_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite"
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
            if kwargs["model"] == "gemini-3.8-flash":
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
        "gemini-3.8-flash",
        "gemini-3.6-flash",
    ]
    assert models.calls[0]["contents"] is models.calls[1]["contents"]
    assert reply.model_used == "gemini-3.6-flash"
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


def test_is_transient_classification() -> None:
    # Status codes
    for code in (408, 429, 500, 502, 503, 504):
        err = Exception(f"status {code}")
        err.code = code
        assert llm._is_transient(err) is True

    # Non-transient status codes
    for code in (400, 401, 403, 404):
        err = Exception(f"status {code}")
        err.code = code
        assert llm._is_transient(err) is False

    # Exception classes
    assert (
        llm._is_transient(
            httpx.RemoteProtocolError("Server disconnected without sending a response")
        )
        is True
    )
    assert llm._is_transient(httpx.ConnectError("connection refused")) is True
    assert llm._is_transient(httpcore.RemoteProtocolError("disconnected")) is True
    assert llm._is_transient(socket.timeout("timed out")) is True  # noqa: UP041
    assert llm._is_transient(ssl.SSLError("handshake failed")) is True


def test_chat_404_moves_immediately_to_next_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "gemini-3.6-flash")

    class NotFoundError(Exception):
        code = 404

    class ModelAPI:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.8-flash":
                raise NotFoundError("model not found")
            return _response_with_text("resumed from fallback")

    models = ModelAPI()
    reply = chat(
        SimpleNamespace(models=models),
        "system prompt",
        [{"role": "user", "content": "hello"}],
        [],
    )
    # 404 should not retry 3 times; it should immediately try fallback
    assert models.calls == ["gemini-3.8-flash", "gemini-3.6-flash"]
    assert reply.model_used == "gemini-3.6-flash"
    assert reply.text == "resumed from fallback"


def test_chat_model_specific_400_moves_to_next_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "gemini-3.6-flash")

    class ModelIncompatibleError(Exception):
        code = 400

    class ModelAPI:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.8-flash":
                raise ModelIncompatibleError("feature is not supported for this model")
            return _response_with_text("resumed from fallback")

    models = ModelAPI()
    reply = chat(
        SimpleNamespace(models=models),
        "system prompt",
        [{"role": "user", "content": "hello"}],
        [],
    )
    assert models.calls == ["gemini-3.8-flash", "gemini-3.6-flash"]
    assert reply.model_used == "gemini-3.6-flash"


def test_chat_auth_error_fails_fast_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "gemini-3.6-flash")

    class AuthError(Exception):
        code = 401

    class ModelAPI:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs["model"])
            raise AuthError("API_KEY_INVALID")

    models = ModelAPI()
    with pytest.raises(LLMError, match="API_KEY_INVALID"):
        chat(
            SimpleNamespace(models=models),
            "system prompt",
            [{"role": "user", "content": "hello"}],
            [],
        )
    # Must fail fast after 1 attempt on primary model without trying fallback
    assert models.calls == ["gemini-3.8-flash"]


def test_model_router_health_cooldown_and_recovery() -> None:
    router = llm.ModelRouter(["model-a", "model-b"])
    now = 100.0

    # Both available initially; model-a is first
    assert router.candidate_models(now) == ["model-a", "model-b"]

    # 429 sets cooldown on model-a
    router.mark_cooldown("model-a", 10.0, "rate limit error with AKIA1234567890123456", now=now)
    # Secret must be redacted
    assert "AKIA1234567890123456" not in router.health["model-a"].last_error
    assert "[REDACTED_AWS_KEY]" in router.health["model-a"].last_error

    # While cooling down, model-a is skipped; model-b is chosen
    assert router.candidate_models(now + 2.0) == ["model-b"]

    # After cooldown expires, model-a is available again (no permanent downgrade)
    assert router.candidate_models(now + 11.0) == ["model-a", "model-b"]

    # 404 marks model-a permanently unavailable
    router.mark_unavailable("model-a", "404 not found")
    assert router.candidate_models(now + 20.0) == ["model-b"]

    # When all models cool down and deadline does not allow
    router.mark_cooldown("model-b", 50.0, "rate limit")
    assert router.candidate_models(now + 20.0) == []
    summary = router.summary()
    assert "model-a (unavailable" in summary
    assert "model-b (cooling down" in summary


def test_mid_conversation_switch_strips_raw_content_on_signature_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")

    class SignatureMismatchError(Exception):
        code = 400

    raw_model_content = types.Content(
        role="model",
        parts=[types.Part.from_text(text="old model internal reasoning")],
    )

    calls = []

    class ModelAPI:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            # If contents has the raw_model_content instance, fail with reasoning 400
            if kwargs["contents"][1] is raw_model_content:
                raise SignatureMismatchError("contents[1] contains unsupported reasoning part")
            return _response_with_text("recovered after strip")

    models = ModelAPI()
    reply = chat(
        SimpleNamespace(models=models),
        "system prompt",
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "raw_content": raw_model_content, "content": "thought"},
            {"role": "tool", "name": "read_file", "content": "file data"},
        ],
        [],
    )

    assert len(calls) == 2
    # Second call had contents rebuilt without raw_model_content
    assert calls[1]["contents"][1] is not raw_model_content
    assert reply.text == "recovered after strip"


def test_openai_compatible_provider_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Analyzing repository",
                            "tool_calls": [
                                {
                                    "id": "call-123",
                                    "type": "function",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": json.dumps({"directory": ""}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers=None, json=None):
            assert "/chat/completions" in url
            assert headers.get("Authorization") == "Bearer fake-openai-key"
            assert json["model"] == "gpt-4o-mini"
            assert json["messages"][0]["role"] == "system"
            assert json["tools"][0]["function"]["name"] == "list_files"
            return FakeResponse()

    monkeypatch.setattr(llm.httpx, "Client", FakeHttpxClient)
    reply = llm._chat_openai_compatible(
        "You are an auditor",
        [{"role": "user", "content": "list repo"}],
        [
            {
                "name": "list_files",
                "description": "List files",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="fake-openai-key",
    )

    assert reply.text == "Analyzing repository"
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "list_files"
    assert reply.tool_calls[0].args == {"directory": ""}
    assert reply.tool_calls[0].provider_id == "call-123"


def test_chat_observability_records_events_in_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "gemini-3.6-flash")

    class BusyError(Exception):
        code = 503

    class ModelAPI:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise BusyError("server busy")
            return _response_with_text("recovered")

    monkeypatch.setattr(llm.time, "sleep", lambda d: None)
    trace = []
    models = ModelAPI()
    reply = chat(
        SimpleNamespace(models=models),
        "system prompt",
        [{"role": "user", "content": "hello"}],
        [],
        trace=trace,
        step=1,
    )

    assert reply.text == "recovered"
    assert any(event.get("type") == "model_retry" for event in trace)


def test_thinking_config_and_provider_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm import _get_thinking_config, _provider_retry_after

    # Thinking levels
    monkeypatch.setenv("LLM_THINKING_LEVEL", "low")
    cfg_low = _get_thinking_config()
    assert cfg_low is not None

    monkeypatch.setenv("LLM_THINKING_LEVEL", "off")
    cfg_off = _get_thinking_config()
    assert cfg_off is None

    # Retry-after headers
    class HeaderError(Exception):
        response = SimpleNamespace(headers={"Retry-After": "2.5"})

    assert _provider_retry_after(HeaderError()) == 2.5
