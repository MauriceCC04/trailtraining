"""
Pipeline: fetch Strava activities and write processing/strava_activities.json

Implements:
- D: removes pandas/stravalib/pickle usage
- E: requests.Session reuse, retries/backoff, incremental fetch via `after` + meta file
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from trailtraining import config
from trailtraining.data.strava import (
    StravaOAuthConfig,
    build_authorize_url,
    exchange_code_for_token,
    get_valid_token,
    save_token,
)
from trailtraining.util.state import load_json, save_json
from trailtraining.web.auth_server import start_auth_server, wait_for_code

STRAVA_API_BASE = "https://www.strava.com/api/v3"
ACTIVITIES_PATH = "/athlete/activities"

# Files written/used by this pipeline
ACTIVITIES_JSON = lambda: os.path.join(config.PROCESSING_DIRECTORY, "strava_activities.json")
META_JSON = lambda: os.path.join(config.PROCESSING_DIRECTORY, "strava_meta.json")

# Defaults (tweak via env)
DEFAULT_LOOKBACK_DAYS = int(os.getenv("TRAILTRAINING_STRAVA_LOOKBACK_DAYS", "365"))
MAX_PAGES = int(os.getenv("TRAILTRAINING_STRAVA_MAX_PAGES", "10"))  # 10 * 200 = 2000 activities max
PER_PAGE = int(os.getenv("TRAILTRAINING_STRAVA_PER_PAGE", "200"))  # Strava max is 200
AFTER_BUFFER_SECONDS = int(os.getenv("TRAILTRAINING_STRAVA_AFTER_BUFFER_SECONDS", str(7 * 24 * 3600)))  # 7 days


def _parse_strava_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # Strava uses ISO8601; start_date is often "...Z"
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        # treat as UTC if missing tz
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _request_with_retry(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """
    Retries on:
      - 429 (rate limit): uses Retry-After when present
      - 5xx: exponential backoff
      - request timeouts: exponential backoff
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


def _api_get(session: requests.Session, path: str, access_token: str, params: Optional[Dict[str, Any]] = None) -> Any:
    resp = _request_with_retry(
        session,
        "GET",
        f"{STRAVA_API_BASE}{path}",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params or {},
    )
    return resp.json()


def _slim_activity(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only fields commonly needed downstream.
    Include BOTH start_date (UTC) and start_date_local.
    """
    return {
        "id": a.get("id"),
        "name": a.get("name"),
        "start_date": a.get("start_date"),
        "start_date_local": a.get("start_date_local"),
        "sport_type": a.get("sport_type"),
        "type": a.get("type"),
        "distance": a.get("distance"),
        "moving_time": a.get("moving_time"),
        "elapsed_time": a.get("elapsed_time"),
        "total_elevation_gain": a.get("total_elevation_gain"),
        "average_heartrate": a.get("average_heartrate"),
        "max_heartrate": a.get("max_heartrate"),
        "elev_low": a.get("elev_low"),
        "elev_high": a.get("elev_high"),
        "workout_type": a.get("workout_type"),
    }


def _get_or_auth_token(cfg: StravaOAuthConfig) -> Dict[str, Any]:
    token = get_valid_token(cfg)
    if token:
        return token

    # First-time OAuth
    start_auth_server(host="127.0.0.1", port=5000)
    url, _state = build_authorize_url(cfg)
    print("\nOpen this URL to authorize Strava:\n")
    print(url)
    print("\nWaiting for authorization callback...\n")
    code = wait_for_code(timeout=300.0)
    token = exchange_code_for_token(cfg, code)
    save_token(token)
    return token


def _compute_after_unix(existing: List[Dict[str, Any]], meta: Dict[str, Any]) -> int:
    """
    Prefer meta.max_start_date_ts (UTC), otherwise compute from existing activities,
    otherwise fall back to NOW - lookback_days.
    """
    now_ts = int(time.time())

    max_ts = meta.get("max_start_date_ts")
    if isinstance(max_ts, (int, float)) and max_ts > 0:
        return max(0, int(max_ts) - AFTER_BUFFER_SECONDS)

    best: int = 0
    for a in existing:
        dt = _parse_strava_datetime(a.get("start_date"))
        if dt:
            ts = int(dt.timestamp())
            if ts > best:
                best = ts

    if best > 0:
        return max(0, best - AFTER_BUFFER_SECONDS)

    return max(0, now_ts - DEFAULT_LOOKBACK_DAYS * 24 * 3600)


def fetch_activities_incremental(
    session: requests.Session,
    access_token: str,
    *,
    after_unix: int,
    per_page: int = PER_PAGE,
    max_pages: int = MAX_PAGES,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        items = _api_get(
            session,
            ACTIVITIES_PATH,
            access_token,
            params={"page": page, "per_page": per_page, "after": after_unix},
        )
        if not items:
            break
        out.extend(items)
        if len(items) < per_page:
            break
    return out


def _merge_by_id(existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for a in existing:
        if a.get("id") is not None:
            merged[str(a["id"])] = a
    for a in new_items:
        if a.get("id") is not None:
            merged[str(a["id"])] = a

    # stable order by start_date desc (fallback by id)
    def key_fn(x: Dict[str, Any]):
        dt = _parse_strava_datetime(x.get("start_date"))
        ts = int(dt.timestamp()) if dt else 0
        return (ts, int(x.get("id") or 0))

    return sorted(merged.values(), key=key_fn, reverse=True)


def main() -> None:
    print("Fetching Strava data...")
    config.ensure_directories()

    cfg = StravaOAuthConfig.from_env()
    token = _get_or_auth_token(cfg)

    access_token = token.get("access_token")
    if not access_token:
        raise RuntimeError("Strava token did not include access_token.")

    existing: List[Dict[str, Any]] = load_json(ACTIVITIES_JSON(), default=[]) or []
    meta: Dict[str, Any] = load_json(META_JSON(), default={}) or {}

    after_unix = _compute_after_unix(existing, meta)

    session = requests.Session()
    raw_new = fetch_activities_incremental(session, access_token, after_unix=after_unix)
    slim_new = [_slim_activity(a) for a in raw_new]

    merged = _merge_by_id(existing, slim_new)

    # update meta
    max_start_ts = 0
    for a in merged:
        dt = _parse_strava_datetime(a.get("start_date"))
        if dt:
            max_start_ts = max(max_start_ts, int(dt.timestamp()))

    meta_out = {
        "last_fetched_at": int(time.time()),
        "after_used": after_unix,
        "max_start_date_ts": max_start_ts,
        "count": len(merged),
        "new_count": len(slim_new),
    }

    save_json(ACTIVITIES_JSON(), merged, compact=True)
    save_json(META_JSON(), meta_out, compact=True)

    print(f"Saved {len(merged)} activities → {ACTIVITIES_JSON()}")
    if slim_new:
        print(f"(Fetched {len(slim_new)} new/updated since after={after_unix})")


if __name__ == "__main__":
    main()