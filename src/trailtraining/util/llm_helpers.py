# src/trailtraining/util/llm_helpers.py
"""LLM call helpers with targeted fallback logic.

Only unsupported-parameter / schema-shape errors trigger parameter
stripping.  Auth, network, timeout, rate-limit, and server errors
are raised immediately.
"""

from __future__ import annotations

import logging
import re

from trailtraining.util.errors import LLMUnsupportedParameterError

log = logging.getLogger(__name__)

# Patterns that indicate a provider rejected an unsupported parameter
# (vs auth, rate-limit, or server errors).
_UNSUPPORTED_PARAM_PATTERNS = (
    re.compile(r"unsupported\s+param", re.IGNORECASE),
    re.compile(r"unknown\s+param", re.IGNORECASE),
    re.compile(r"not\s+support", re.IGNORECASE),
    re.compile(r"invalid.*param", re.IGNORECASE),
    re.compile(r"invalid\s+argument", re.IGNORECASE),
    re.compile(r"INVALID_ARGUMENT", re.IGNORECASE),
    re.compile(r"unexpected.*key", re.IGNORECASE),
    re.compile(r"unrecognized.*field", re.IGNORECASE),
    re.compile(r"additional\s+properties", re.IGNORECASE),
    re.compile(r"schema.*not.*valid", re.IGNORECASE),
    re.compile(r"response_format.*not.*support", re.IGNORECASE),
    re.compile(r"json_schema.*not.*support", re.IGNORECASE),
)


def _is_unsupported_parameter_error(exc: Exception) -> bool:
    """Return True if the exception indicates an unsupported parameter.

    This is the ONLY condition under which parameter-stripping fallback
    should be attempted.
    """
    msg = str(exc).lower()
    for pattern in _UNSUPPORTED_PARAM_PATTERNS:
        if pattern.search(msg):
            return True

    # OpenAI SDK BadRequestError with status 400 + param-related message
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 400:
        for pattern in _UNSUPPORTED_PARAM_PATTERNS:
            if pattern.search(msg):
                return True

    return False


def _classify_and_raise(exc: Exception) -> None:
    """Re-raise as LLMUnsupportedParameterError if appropriate, else re-raise original."""
    if _is_unsupported_parameter_error(exc):
        raise LLMUnsupportedParameterError(
            message=f"LLM provider rejected parameter: {exc}",
            hint="Will attempt fallback with stripped parameters.",
        ) from exc
    raise exc
