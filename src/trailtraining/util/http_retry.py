# src/trailtraining/util/http_retry.py
"""Shared HTTP retry logic for external provider integrations.

Retries on:
  - 429 (rate limit): uses Retry-After header when present
  - 5xx: exponential backoff
  - Timeouts / connection errors: exponential backoff

Does NOT retry on 4xx (except 429) — these are raised immediately
as ExternalServiceError with context.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import requests

from trailtraining.util.errors import ExternalServiceError

DEFAULT_MAX_RETRIES = 6
DEFAULT_TIMEOUT = 30


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    service_name: str = "external service",
    **kwargs: Any,
) -> requests.Response:
    """Execute an HTTP request with retry logic for transient failures.

    Raises ExternalServiceError with context on permanent failures.
    """
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as err:
            last_err = err
            time.sleep(min(30, 2**attempt))
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if (retry_after and retry_after.isdigit()) else (2**attempt)
            time.sleep(min(60, max(1, wait)))
            continue

        if 500 <= resp.status_code <= 599:
            last_err = ExternalServiceError(
                message=f"{service_name} server error ({resp.status_code}) for {method} {url}",
                hint=resp.text[:300]
                if resp.text
                else "The service may be temporarily unavailable.",
            )
            time.sleep(min(30, 2**attempt))
            continue

        if 400 <= resp.status_code <= 499:
            raise ExternalServiceError(
                message=f"{service_name} request failed with HTTP {resp.status_code}",
                hint=resp.text[:300] if resp.text else f"{method} {url}",
            )

        return resp

    if isinstance(last_err, ExternalServiceError):
        raise last_err

    raise ExternalServiceError(
        message=f"{service_name} request failed after {max_retries} retries: {method} {url}",
        hint=str(last_err) if last_err else "Check network access and service availability.",
    )
