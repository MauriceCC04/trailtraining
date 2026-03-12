# src/trailtraining/data/combine.py

from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional

from trailtraining import config
from trailtraining.metrics.training_load import activity_training_load_hours
from trailtraining.util.state import load_json, save_json


def _as_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _extract_sleep_date(entry: dict[str, Any]) -> Optional[str]:
    """
    Garmin/Intervals filtered_sleep.json uses 'calendarDate' (YYYY-MM-DD).
    Be tolerant of other key names and nested dailySleepDTO.
    """
    for k in ("calendarDate", "date", "day", "calendar_date", "id"):
        v = entry.get(k)
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]

    dto = entry.get("dailySleepDTO")
    if isinstance(dto, dict):
        v = dto.get("calendarDate") or dto.get("date") or dto.get("day")
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]

    return None


def _date_key_from_activity(a: dict[str, Any]) -> Optional[str]:
    # Prefer local date for grouping
    s = a.get("start_date_local") or a.get("start_date")
    if not isinstance(s, str) or len(s) < 10:
        return None
    return s[:10]


def _load_sleep_by_date(path: str) -> dict[str, dict[str, Any]]:
    raw = load_json(path, default=None)
    if raw is None:
        return {}

    # If it's already dict keyed by date, normalize keys to YYYY-MM-DD
    if isinstance(raw, dict):
        out: dict[str, dict[str, Any]] = {}
        for k, v in raw.items():
            if isinstance(k, str) and len(k) >= 10 and isinstance(v, dict):
                out[k[:10]] = v
        return out

    # Typical case: list of dict entries with calendarDate
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            d = _extract_sleep_date(item)
            if d:
                out[d] = item
    return out


def _load_activities_by_date(path: str) -> dict[str, list[dict[str, Any]]]:
    raw = load_json(path, default=[])
    if not isinstance(raw, list):
        return {}

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in raw:
        if not isinstance(a, dict):
            continue
        d = _date_key_from_activity(a)
        if d:
            out[d].append(a)
    return out


def _compute_rollup(
    combined: list[dict[str, Any]],
    *,
    end_date: date,
    window_days: int,
) -> dict[str, Any]:
    start_date = end_date - timedelta(days=window_days - 1)

    total_distance_m = 0.0
    total_elev_m = 0.0
    total_moving_s = 0.0
    total_training_load_h = 0.0

    hr_sum = 0.0
    hr_n = 0
    activity_count = 0

    sports = Counter()
    sleep_days_with_data = 0

    # Per-sport aggregation (distance/time/elev + training load)
    by_sport: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "distance_m": 0.0,
            "elev_m": 0.0,
            "moving_s": 0.0,
            "training_load_h": 0.0,
        }
    )

    for day_obj in combined:
        d_str = day_obj.get("date")
        if not isinstance(d_str, str):
            continue
        d = _as_date(d_str)
        if not d or d < start_date or d > end_date:
            continue

        if day_obj.get("sleep") is not None:
            sleep_days_with_data += 1

        acts = day_obj.get("activities") or []
        if not isinstance(acts, list):
            continue

        for a in acts:
            if not isinstance(a, dict):
                continue

            activity_count += 1
            sport = str(a.get("sport_type") or a.get("type") or "unknown")
            sports[sport] += 1
            by_sport[sport]["count"] += 1.0

            dist = a.get("distance")
            if isinstance(dist, (int, float)):
                total_distance_m += float(dist)
                by_sport[sport]["distance_m"] += float(dist)

            elev = a.get("total_elevation_gain")
            if isinstance(elev, (int, float)):
                total_elev_m += float(elev)
                by_sport[sport]["elev_m"] += float(elev)

            mv = a.get("moving_time")
            if isinstance(mv, (int, float)):
                total_moving_s += float(mv)
                by_sport[sport]["moving_s"] += float(mv)

            # Training load: moving_time_hours * load_factor
            # Distance can be 0: as long as moving_time > 0, this counts as load.
            tl_h = float(activity_training_load_hours(a))
            if tl_h > 0:
                total_training_load_h += tl_h
                by_sport[sport]["training_load_h"] += tl_h

            hr = a.get("average_heartrate")
            if isinstance(hr, (int, float)):
                hr_sum += float(hr)
                hr_n += 1

    avg_hr = (hr_sum / hr_n) if hr_n else None

    by_sport_out: dict[str, Any] = {}
    for sport, agg in by_sport.items():
        by_sport_out[sport] = {
            "count": int(agg["count"]),
            "distance_km": round(float(agg["distance_m"]) / 1000.0, 3),
            "total_elevation_m": round(float(agg["elev_m"]), 1),
            "total_moving_time_hours": round(float(agg["moving_s"]) / 3600.0, 3),
            "training_load_hours": round(float(agg["training_load_h"]), 3),
        }

    return {
        "window_days": window_days,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "activities": {
            "count": activity_count,
            "total_distance_km": round(total_distance_m / 1000.0, 3),
            "total_elevation_m": round(total_elev_m, 1),
            "total_moving_time_hours": round(total_moving_s / 3600.0, 3),
            "total_training_load_hours": round(total_training_load_h, 3),
            "average_heartrate_mean": (round(avg_hr, 2) if avg_hr is not None else None),
            "count_by_sport": dict(sports),
            "by_sport": by_sport_out,
        },
        "sleep_days_with_data": sleep_days_with_data,
    }


def main() -> None:
    config.ensure_directories()

    sleep_path = os.path.join(config.PROCESSING_DIRECTORY, "filtered_sleep.json")
    activities_path = os.path.join(config.PROCESSING_DIRECTORY, "strava_activities.json")

    sleep_by_date = _load_sleep_by_date(sleep_path)
    activities_by_date = _load_activities_by_date(activities_path)

    print(f"Loaded sleep days: {len(sleep_by_date)} from {sleep_path}")
    print(f"Loaded activity days: {len(activities_by_date)} from {activities_path}")

    all_dates = sorted(set(sleep_by_date.keys()) | set(activities_by_date.keys()))
    combined: list[dict[str, Any]] = []

    for d in all_dates:
        combined.append(
            {
                "date": d,
                "sleep": sleep_by_date.get(d),
                "activities": activities_by_date.get(d, []),
            }
        )

    out_summary = os.path.join(config.PROMPTING_DIRECTORY, "combined_summary.json")
    save_json(out_summary, combined, compact=True)

    # Rollups (7d + 28d) ending at the most recent date we have
    if combined:
        last_date = _as_date(combined[-1]["date"])
        if last_date:
            rollups = {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "windows": {
                    "7": _compute_rollup(combined, end_date=last_date, window_days=7),
                    "28": _compute_rollup(combined, end_date=last_date, window_days=28),
                },
            }
            out_rollups = os.path.join(config.PROMPTING_DIRECTORY, "combined_rollups.json")
            save_json(out_rollups, rollups, compact=True)

    print(f"Wrote: {out_summary}")
    print(f"Wrote: {os.path.join(config.PROMPTING_DIRECTORY, 'combined_rollups.json')}")


if __name__ == "__main__":
    main()
