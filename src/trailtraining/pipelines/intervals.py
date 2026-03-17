from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Any, Optional

import requests

from trailtraining import config
from trailtraining.util.errors import (
    ConfigError,
    DataValidationError,
    ExternalServiceError,
)
from trailtraining.util.http_retry import request_with_retry

BASE_URL = os.getenv("INTERVALS_BASE_URL", "https://intervals.icu/api/v1")
_SESSION = requests.Session()


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) else default


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
    bearer = _env_str("INTERVALS_ACCESS_TOKEN")
    if bearer:
        return {"Authorization": f"Bearer {bearer}"}

    runtime = config.current()
    api_key = _env_str("INTERVALS_API_KEY") or runtime.intervals_api_key

    if not api_key:
        raise ConfigError(
            message="Missing INTERVALS_API_KEY (or INTERVALS_ACCESS_TOKEN).",
            hint="Set INTERVALS_API_KEY in your profile env, or run `trailtraining doctor`.",
        )

    import base64

    basic = base64.b64encode(f"API_KEY:{api_key}".encode()).decode("ascii")
    return {"Authorization": f"Basic {basic}"}


def _request_with_retry(
    session: requests.Session, method: str, url: str, **kwargs: Any
) -> requests.Response:
    return request_with_retry(session, method, url, service_name="Intervals.icu", **kwargs)


def fetch_wellness(oldest: str, newest: str) -> list[dict[str, Any]]:
    athlete_id_env = _env_str("INTERVALS_ATHLETE_ID")
    runtime = config.current()
    athlete_id = athlete_id_env or runtime.intervals_athlete_id or "0"

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
        raise ExternalServiceError(
            message="Intervals wellness JSON parse failed.",
            hint=f"HTTP {resp.status_code}: {resp.text[:300]}",
        ) from err

    if not isinstance(data, list):
        raise DataValidationError(
            message="Unexpected Intervals wellness response.",
            hint=f"Expected a list, got {type(data).__name__}.",
        )
    return data


def normalize_to_filtered_sleep(entry: dict[str, Any]) -> dict[str, Any]:
    day = str(_pick(entry, "id", "day", "date", "calendarDate", default=""))[:10]
    if not day:
        raise DataValidationError(
            message="Wellness entry missing date/id.",
            hint=str(entry)[:300],
        )

    sleep_secs = _to_int(_pick(entry, "sleepSecs", "sleep_seconds", "sleepTimeSeconds"))
    resting_hr = _to_int(_pick(entry, "restingHR", "restingHr", "restingHeartRate"))
    avg_hrv = _to_int(_pick(entry, "avgOvernightHrv", "hrv", "hrvRmssd", "rmssd"))

    return {
        "calendarDate": day,
        "sleepTimeSeconds": sleep_secs,
        "restingHeartRate": resting_hr,
        "avgOvernightHrv": avg_hrv,
    }


def ensure_personal_stub(runtime: Optional[config.RuntimeConfig] = None) -> None:
    runtime = runtime or config.current()
    out_path = runtime.paths.prompting_directory / "formatted_personal_data.json"
    if out_path.exists():
        return

    stub: dict[str, dict[str, Any]] = {
        "userInfo": {},
        "biometricProfile": {},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stub, f, indent=2)


def _validate_ymd(s: str, name: str) -> str:
    s = (s or "").strip()
    try:
        date.fromisoformat(s)
    except ValueError as err:
        raise DataValidationError(
            message=f"{name} must be YYYY-MM-DD.",
            hint=f"Got {s!r}.",
        ) from err
    return s


def main(*, oldest: Optional[str] = None, newest: Optional[str] = None) -> None:
    runtime = config.current()
    config.ensure_directories(runtime)
    paths = runtime.paths

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
        raise DataValidationError(
            message="oldest must be <= newest.",
            hint=f"Got {oldest_v} > {newest_v}.",
        )

    print(f"Fetching Intervals wellness {oldest_v} -> {newest_v} ...")
    raw = fetch_wellness(oldest=oldest_v, newest=newest_v)
    normalized = [normalize_to_filtered_sleep(x) for x in raw]
    normalized.sort(key=lambda r: r["calendarDate"])

    out = paths.processing_directory / "filtered_sleep.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=4)

    ensure_personal_stub(runtime)
    print(f"Intervals wellness saved: {out}")
