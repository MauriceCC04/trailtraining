"""
Pipeline: fetch Strava activities and write processing/strava_activities.json

Implements incremental fetch via `after` + meta file, connection pooling,
retries/backoff, and removes legacy pandas/stravalib/pickle dependencies.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from trailtraining import config
from trailtraining.data.strava import (
    StravaOAuthConfig,
    build_authorize_url,
    default_token_path,
    exchange_code_for_token,
    get_valid_token,
    save_token,
)
from trailtraining.util.errors import DataValidationError, ExternalServiceError
from trailtraining.util.http_retry import request_with_retry
from trailtraining.util.state import load_json, save_json
from trailtraining.web.auth_server import start_auth_server, wait_for_code

STRAVA_API_BASE = "https://www.strava.com/api/v3"
ACTIVITIES_PATH = "/athlete/activities"

DEFAULT_LOOKBACK_DAYS = int(os.getenv("TRAILTRAINING_STRAVA_LOOKBACK_DAYS", "365"))
MAX_PAGES = int(os.getenv("TRAILTRAINING_STRAVA_MAX_PAGES", "0"))
HARD_MAX_PAGES = int(os.getenv("TRAILTRAINING_STRAVA_HARD_MAX_PAGES", "1000"))
PER_PAGE = int(os.getenv("TRAILTRAINING_STRAVA_PER_PAGE", "200"))
AFTER_BUFFER_SECONDS = int(
    os.getenv("TRAILTRAINING_STRAVA_AFTER_BUFFER_SECONDS", str(7 * 24 * 3600))
)


def _parse_strava_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _request_with_retry(
    session: requests.Session, method: str, url: str, **kwargs: Any
) -> requests.Response:
    return request_with_retry(session, method, url, service_name="Strava", **kwargs)


def _api_get(
    session: requests.Session, path: str, access_token: str, params: Optional[dict[str, Any]] = None
) -> Any:
    resp = _request_with_retry(
        session,
        "GET",
        f"{STRAVA_API_BASE}{path}",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params or {},
    )
    try:
        return resp.json()
    except ValueError as err:
        raise ExternalServiceError(
            message="Strava JSON parse failed.",
            hint=f"HTTP {resp.status_code}: {resp.text[:300]}",
        ) from err


def _slim_activity(a: dict[str, Any]) -> dict[str, Any]:
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


def _get_or_auth_token(cfg: StravaOAuthConfig) -> dict[str, Any]:
    token = get_valid_token(cfg)
    if token:
        return token

    start_auth_server(host="127.0.0.1", port=5000)
    url, _state = build_authorize_url(cfg)
    print("\nOpen this URL to authorize Strava:\n")
    print(url)
    print("\nWaiting for authorization callback...\n")
    code = wait_for_code(timeout=300.0)
    token = exchange_code_for_token(cfg, code)
    save_token(token)
    return token


def auth_main(*, force: bool = False) -> None:
    config.ensure_directories()
    cfg = StravaOAuthConfig.from_env()
    token_path = default_token_path()

    if not force:
        token = get_valid_token(cfg, token_path=token_path)
        if token:
            print(f"✅ Strava already authorized -> {token_path}")
            return

    start_auth_server(host="127.0.0.1", port=5000)
    url, _state = build_authorize_url(cfg)
    print("\nOpen this URL to authorize Strava:\n")
    print(url)
    print("\nWaiting for authorization callback...\n")
    code = wait_for_code(timeout=300.0)
    token = exchange_code_for_token(cfg, code)
    save_token(token, token_path)
    print(f"✅ Saved Strava token -> {token_path}")


def _compute_after_unix(
    existing: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> int:
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

    now = int(datetime.now(tz=timezone.utc).timestamp())
    return max(0, now - lookback_days * 86400)


def fetch_activities_incremental(
    session: requests.Session,
    access_token: str,
    *,
    after_unix: int,
    per_page: int = PER_PAGE,
    max_pages: int = MAX_PAGES,
    hard_max_pages: int = HARD_MAX_PAGES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []

    page = 1
    pages_fetched = 0
    hit_max_pages = False

    while True:
        if hard_max_pages > 0 and pages_fetched >= hard_max_pages:
            raise RuntimeError(
                f"Strava pagination exceeded TRAILTRAINING_STRAVA_HARD_MAX_PAGES={hard_max_pages}. "
                "Increase it if this is expected."
            )

        if max_pages > 0 and pages_fetched >= max_pages:
            hit_max_pages = True
            break

        items = _api_get(
            session,
            ACTIVITIES_PATH,
            access_token,
            params={"page": page, "per_page": per_page, "after": after_unix},
        )
        if not isinstance(items, list):
            raise DataValidationError(
                message="Unexpected Strava activities response.",
                hint=f"Expected a list for page {page}, got {type(items).__name__}.",
            )
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise DataValidationError(
                    message="Unexpected Strava activity item.",
                    hint=f"Page {page} item {idx} is {type(item).__name__}, expected dict.",
                )

        if not items:
            break

        out.extend(items)
        pages_fetched += 1

        if len(items) < per_page:
            break

        page += 1

    info = {
        "pages_fetched": pages_fetched,
        "per_page": per_page,
        "max_pages": max_pages,
        "hard_max_pages": hard_max_pages,
        "hit_max_pages": hit_max_pages,
    }
    return out, info


def _merge_by_id(
    existing: list[dict[str, Any]], new_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for a in existing:
        if a.get("id") is not None:
            merged[str(a["id"])] = a
    for a in new_items:
        if a.get("id") is not None:
            merged[str(a["id"])] = a

    def key_fn(x: dict[str, Any]) -> tuple[int, int]:
        dt = _parse_strava_datetime(x.get("start_date"))
        ts = int(dt.timestamp()) if dt else 0
        return (ts, int(x.get("id") or 0))

    return sorted(merged.values(), key=key_fn, reverse=True)


def main() -> None:
    print("Fetching Strava data...")
    runtime = config.current()
    config.ensure_directories(runtime)
    processing_dir = runtime.paths.processing_directory

    activities_path = str(processing_dir / "strava_activities.json")
    meta_path = str(processing_dir / "strava_meta.json")

    cfg = StravaOAuthConfig.from_env()
    token = _get_or_auth_token(cfg)

    access_token = token.get("access_token")
    if not access_token:
        raise RuntimeError("Strava token did not include access_token.")

    existing: list[dict[str, Any]] = load_json(activities_path, default=[]) or []
    meta: dict[str, Any] = load_json(meta_path, default={}) or {}

    after_unix = _compute_after_unix(existing, meta)

    session = requests.Session()
    raw_new, page_info = fetch_activities_incremental(session, access_token, after_unix=after_unix)

    if page_info.get("hit_max_pages"):
        print(
            "⚠️  Strava fetch hit TRAILTRAINING_STRAVA_MAX_PAGES limit. "
            "Set TRAILTRAINING_STRAVA_MAX_PAGES=0 (unlimited) or increase it to avoid truncation."
        )

    slim_new = [_slim_activity(a) for a in raw_new]
    merged = _merge_by_id(existing, slim_new)

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
        "pagination": page_info,
    }

    save_json(activities_path, merged, compact=True)
    save_json(meta_path, meta_out, compact=True)

    print(f"Saved {len(merged)} activities -> {activities_path}")
    if slim_new:
        print(f"(Fetched {len(slim_new)} new/updated since after={after_unix})")


if __name__ == "__main__":
    main()
