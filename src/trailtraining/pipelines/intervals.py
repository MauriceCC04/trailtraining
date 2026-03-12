# src/trailtraining/pipelines/intervals.py
from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from typing import Any, Optional

import requests

from trailtraining import config

BASE_URL = os.getenv("INTERVALS_BASE_URL", "https://intervals.icu/api/v1")
_SESSION = requests.Session()


def _pick(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _to_int(v: Any, default: int = -1) -> int:
    if v is None:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _auth_headers() -> dict[str, str]:
    # If you ever add OAuth later:
    bearer = os.getenv("INTERVALS_ACCESS_TOKEN", "").strip()
    if bearer:
        return {"Authorization": f"Bearer {bearer}"}

    # API key (personal use): Basic auth where username is literally "API_KEY"
    api_key = (os.getenv("INTERVALS_API_KEY") or getattr(config, "INTERVALS_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("Missing INTERVALS_API_KEY (or INTERVALS_ACCESS_TOKEN).")

    import base64

    basic = base64.b64encode(f"API_KEY:{api_key}".encode()).decode("ascii")
    return {"Authorization": f"Basic {basic}"}


def _request_with_retry(
    session: requests.Session, method: str, url: str, **kwargs
) -> requests.Response:
    """
    Retries on:
      - 429 (rate limit): uses Retry-After when present
      - 5xx: exponential backoff
      - timeouts/connection errors: exponential backoff
    """
    last_err: Optional[Exception] = None

    for attempt in range(0, 6):
        try:
            resp = session.request(method, url, timeout=30, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            time.sleep(min(30, 2**attempt))
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if (retry_after and retry_after.isdigit()) else (2**attempt)
            time.sleep(min(60, max(1, wait)))
            continue

        if 500 <= resp.status_code <= 599:
            time.sleep(min(30, 2**attempt))
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError(f"HTTP request failed after retries: {method} {url} ({last_err})")


def fetch_wellness(oldest: str, newest: str) -> list[dict[str, Any]]:
    athlete_id = (
        os.getenv("INTERVALS_ATHLETE_ID") or getattr(config, "INTERVALS_ATHLETE_ID", "0")
    ).strip() or "0"
    url = f"{BASE_URL}/athlete/{athlete_id}/wellness"
    params = {"oldest": oldest, "newest": newest}

    resp = _request_with_retry(
        _SESSION,
        "GET",
        url,
        params=params,
        headers={**_auth_headers(), "Accept": "application/json"},
    )

    try:
        data = resp.json()
    except Exception as err:
        raise RuntimeError(
            f"Intervals wellness JSON parse failed: {err}. HTTP {resp.status_code}: {resp.text[:500]}"
        ) from err

    if not isinstance(data, list):
        raise ValueError(
            f"Unexpected Intervals wellness response (expected list). Got: {type(data)}"
        )
    return data


def normalize_to_filtered_sleep(entry: dict[str, Any]) -> dict[str, Any]:
    # Date key: Intervals often uses `id` for the day (YYYY-MM-DD)
    day = str(_pick(entry, "id", "day", "date", "calendarDate", default=""))[:10]
    if not day:
        raise ValueError(f"Wellness entry missing date/id: {entry}")

    sleep_secs = _to_int(_pick(entry, "sleepSecs", "sleep_seconds", "sleepTimeSeconds"))
    resting_hr = _to_int(_pick(entry, "restingHR", "restingHr", "restingHeartRate"))

    # HRV/status/body battery are Garmin-ish fields expected by your current schema
    avg_hrv = _to_int(_pick(entry, "avgOvernightHrv", "hrv", "hrvRmssd", "rmssd"))

    return {
        "calendarDate": day,
        "sleepTimeSeconds": sleep_secs,
        "restingHeartRate": resting_hr,
        "avgOvernightHrv": avg_hrv,
    }


def ensure_personal_stub() -> None:
    out_path = os.path.join(config.PROMPTING_DIRECTORY, "formatted_personal_data.json")
    if os.path.exists(out_path):
        return
    stub = {"userInfo": {}, "biometricProfile": {}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stub, f, indent=2)


def _validate_ymd(s: str, name: str) -> str:
    s = (s or "").strip()
    try:
        date.fromisoformat(s)
    except ValueError as err:
        raise RuntimeError(f"{name} must be YYYY-MM-DD. Got: {s!r}") from err
    return s


def main(*, oldest: Optional[str] = None, newest: Optional[str] = None) -> None:
    config.ensure_directories()

    # Priority:
    # 1) CLI args
    # 2) New env names: TRAILTRAINING_WELLNESS_OLDEST / TRAILTRAINING_WELLNESS_NEWEST
    # 3) Back-compat env names used by README/node script: OLDEST / NEWEST
    # 4) Fallback: lookback window to today
    newest_raw = (
        newest
        or os.getenv("TRAILTRAINING_WELLNESS_NEWEST")
        or os.getenv("NEWEST")
        or date.today().isoformat()
    )

    lookback_days = int(os.getenv("TRAILTRAINING_WELLNESS_LOOKBACK_DAYS", "200"))
    oldest_default = (date.today() - timedelta(days=lookback_days)).isoformat()
    oldest_raw = (
        oldest
        or os.getenv("TRAILTRAINING_WELLNESS_OLDEST")
        or os.getenv("OLDEST")
        or oldest_default
    )

    newest_v = _validate_ymd(newest_raw, "newest")
    oldest_v = _validate_ymd(oldest_raw, "oldest")

    if date.fromisoformat(oldest_v) > date.fromisoformat(newest_v):
        raise RuntimeError(f"oldest must be <= newest (got {oldest_v} > {newest_v})")

    print(f"Fetching Intervals wellness {oldest_v} → {newest_v} ...")
    raw = fetch_wellness(oldest=oldest_v, newest=newest_v)
    normalized = [normalize_to_filtered_sleep(x) for x in raw]
    normalized.sort(key=lambda r: r["calendarDate"])

    out = os.path.join(config.PROCESSING_DIRECTORY, "filtered_sleep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=4)

    ensure_personal_stub()
    print(f"Intervals wellness saved: {out}")
