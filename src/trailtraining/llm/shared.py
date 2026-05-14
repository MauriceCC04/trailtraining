from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from trailtraining.util.errors import LLMUnsupportedParameterError
from trailtraining.util.llm_helpers import _classify_and_raise

log = logging.getLogger(__name__)

_MONTH_MAP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_MONTH_PATTERN = (
    r"january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

_STRUCTURED_DEBUG_ENV = "TRAILTRAINING_DEBUG_STRUCTURED_OUTPUTS"
_STRUCTURED_DEBUG_ONCE_ENV = "TRAILTRAINING_DEBUG_STRUCTURED_OUTPUTS_ONCE"
_STRUCTURED_DEBUG_EMITTED = False
_LOCAL_VLLM_FORCED_RESPONSE_WARNING_EMITTED = False


@dataclass(frozen=True)
class _StructuredChatResponse:
    output_text: str
    model: Optional[str]
    response: Any

    def __str__(self) -> str:
        return self.output_text

    def __getattr__(self, name: str) -> Any:
        return getattr(self.response, name)


class StructuredOutputUnavailableError(RuntimeError):
    def __init__(
        self,
        *,
        schema_name: str,
        attempted_modes: list[str],
        last_error: Exception | None,
    ) -> None:
        detail = f"; last_error={last_error}" if last_error is not None else ""
        super().__init__(
            f"No structured-output mode succeeded for schema '{schema_name}'. "
            f"attempted_modes={attempted_modes}{detail}"
        )
        self.schema_name = schema_name
        self.attempted_modes = list(attempted_modes)
        self.last_error = last_error


def _client_base_url(client: OpenAI) -> str:
    raw = getattr(client, "base_url", None)
    if raw is None:
        raw = getattr(client, "_base_url", None)
    return str(raw or "").strip()


def _is_local_openai_compatible_base_url(base_url: str) -> bool:
    low = base_url.strip().lower()
    if not low:
        return False
    return not ("openrouter.ai" in low or "api.openai.com" in low or "api.anthropic.com" in low)


def _is_local_vllm_client(client: OpenAI) -> bool:
    return _is_local_openai_compatible_base_url(_client_base_url(client))


def _structured_api_preference(client: OpenAI) -> str:
    global _LOCAL_VLLM_FORCED_RESPONSE_WARNING_EMITTED

    forced = (os.getenv("TRAILTRAINING_FORCE_API") or "").strip().lower()
    if _is_local_vllm_client(client):
        if forced == "responses" and not _LOCAL_VLLM_FORCED_RESPONSE_WARNING_EMITTED:
            log.warning(
                "Ignoring TRAILTRAINING_FORCE_API=responses for local OpenAI-compatible endpoint %s; "
                "using chat.completions because vLLM structured outputs are documented on the chat/completions path.",
                _client_base_url(client),
            )
            _LOCAL_VLLM_FORCED_RESPONSE_WARNING_EMITTED = True
        return "chat"

    if forced in {"chat", "responses"}:
        return forced

    base_url = _client_base_url(client).lower()
    return "responses" if "openrouter.ai" in base_url else "chat"


def _merge_structured_extra_body(
    client: OpenAI,
    kwargs: dict[str, Any],
    schema_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra_body = dict(kwargs.get("extra_body") or {})
    if schema_body is not None and _is_local_vllm_client(client):
        structured = dict(extra_body.get("structured_outputs") or {})
        structured.setdefault("json", schema_body)
        extra_body["structured_outputs"] = structured
    return extra_body


def _coerce_input_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _extract_chat_completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                continue

            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str) and text_attr:
                parts.append(text_attr)
        if parts:
            return "".join(parts)

    parsed = getattr(message, "parsed", None)
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed
    try:
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        return str(parsed)


def _structured_debug_enabled() -> bool:
    raw = (os.getenv(_STRUCTURED_DEBUG_ENV) or "").strip().lower()
    return raw in {"1", "true", "yes", "on", "debug"}


def _structured_debug_once_only() -> bool:
    raw = (os.getenv(_STRUCTURED_DEBUG_ONCE_ENV) or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _safe_json_dump(payload: Any) -> str:
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return repr(payload)


def _debug_structured_exchange(
    *,
    api_path: str,
    client: OpenAI,
    schema_name: str,
    request_payload: dict[str, Any],
    raw_text: str | None = None,
) -> None:
    global _STRUCTURED_DEBUG_EMITTED

    if not _structured_debug_enabled():
        return
    if _structured_debug_once_only() and _STRUCTURED_DEBUG_EMITTED:
        return

    log.warning(
        "Structured-output debug\napi_path=%s\nbase_url=%s\nschema=%s\nrequest=%s",
        api_path,
        _client_base_url(client),
        schema_name,
        _safe_json_dump(request_payload),
    )
    if raw_text is not None:
        log.warning("Structured-output raw response\nschema=%s\nraw_text=%s", schema_name, raw_text)
        _STRUCTURED_DEBUG_EMITTED = True


def _call_with_param_fallback(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    def _strip_verbosity(kw: dict[str, Any]) -> dict[str, Any]:
        text = {k: v for k, v in kw.get("text", {}).items() if k != "verbosity"}
        return {**kw, "text": text} if text else {k: v for k, v in kw.items() if k != "text"}

    def _strip_reasoning(kw: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in kw.items() if k != "reasoning"}

    attempts = [
        ("full", kwargs),
        ("no_text_verbosity", _strip_verbosity(kwargs)),
        ("no_reasoning", _strip_reasoning(kwargs)),
        ("bare_minimum", _strip_reasoning(_strip_verbosity(kwargs))),
    ]

    last_exc: Optional[Exception] = None
    for label, attempt_kwargs in attempts:
        try:
            return client.responses.create(**attempt_kwargs)
        except Exception as exc:
            try:
                _classify_and_raise(exc)
            except LLMUnsupportedParameterError as unsupported:
                log.warning(
                    "LLM call rejected %s attempt due to unsupported parameters: %s",
                    label,
                    unsupported,
                )
                last_exc = unsupported
                continue
            raise

    assert last_exc is not None
    raise last_exc


# Backward-compatible alias for old call sites.
call_with_param_fallback = _call_with_param_fallback


def _call_chat_completion_with_schema(
    client: OpenAI,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
) -> Any:
    name = str(schema.get("name") or "unnamed_schema")
    body = schema.get("schema")
    if not isinstance(body, dict):
        raise ValueError(f"Schema '{name}' is missing a dict-valued 'schema' body.")

    messages: list[dict[str, str]] = []
    instructions = kwargs.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": _coerce_input_to_text(kwargs.get("input", ""))})

    chat_kwargs: dict[str, Any] = {
        "model": kwargs.get("model"),
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": body,
                "strict": True,
            },
        },
    }

    for key in (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "stop",
        "n",
    ):
        if key in kwargs and kwargs[key] is not None:
            chat_kwargs[key] = kwargs[key]

    extra_body = _merge_structured_extra_body(client, kwargs, schema_body=body)
    if extra_body:
        chat_kwargs["extra_body"] = extra_body

    _debug_structured_exchange(
        api_path="chat.completions",
        client=client,
        schema_name=name,
        request_payload=chat_kwargs,
    )

    try:
        response = client.chat.completions.create(**chat_kwargs)
    except Exception as exc:
        _classify_and_raise(exc)
        raise

    output_text = _extract_chat_completion_text(response)
    _debug_structured_exchange(
        api_path="chat.completions",
        client=client,
        schema_name=name,
        request_payload=chat_kwargs,
        raw_text=output_text,
    )
    if not output_text:
        raise RuntimeError(
            f"Chat Completions structured-output response was empty for schema '{name}'."
        )

    return _StructuredChatResponse(
        output_text=output_text,
        model=getattr(response, "model", None),
        response=response,
    )


def _call_responses_with_schema(
    client: OpenAI,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
) -> Any:
    name = str(schema.get("name") or "unnamed_schema")
    body = schema.get("schema")
    if not isinstance(body, dict):
        raise ValueError(f"Schema '{name}' is missing a dict-valued 'schema' body.")

    response_kwargs: dict[str, Any] = {
        **kwargs,
        "text": {
            **kwargs.get("text", {}),
            "format": {
                "type": "json_schema",
                "name": name,
                "schema": body,
                "strict": True,
            },
        },
    }

    extra_body = _merge_structured_extra_body(client, kwargs)
    if extra_body:
        response_kwargs["extra_body"] = extra_body

    _debug_structured_exchange(
        api_path="responses.create",
        client=client,
        schema_name=name,
        request_payload=response_kwargs,
    )
    response = _call_with_param_fallback(client, response_kwargs)
    raw_text = getattr(response, "output_text", None)
    _debug_structured_exchange(
        api_path="responses.create",
        client=client,
        schema_name=name,
        request_payload=response_kwargs,
        raw_text=raw_text if isinstance(raw_text, str) else str(response),
    )
    return response


# keep the rest of the file unchanged
