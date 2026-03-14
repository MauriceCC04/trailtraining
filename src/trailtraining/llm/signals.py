# src/trailtraining/llm/signals.py
from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any, Optional, cast

from trailtraining.metrics.training_load import day_training_load_hours

# move any other top-level imports here


def _as_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            out.append(float(item))
    return out


def _mean(xs: Sequence[float | None]) -> Optional[float]:
    xs2 = [float(x) for x in xs if x is not None]
    if not xs2:
        return None
    return sum(xs2) / len(xs2)


def _round_or_none(x: float | None, ndigits: int = 2) -> float | None:
    return round(x, ndigits) if x is not None else None


def _sleep_hours(day_obj: dict[str, Any]) -> Optional[float]:
    sleep = day_obj.get("sleep")
    if not isinstance(sleep, dict):
        return None
    secs = sleep.get("sleepTimeSeconds")
    if isinstance(secs, (int, float)) and secs > 0:
        return float(secs) / 3600.0
    return None


def _sleep_int(day_obj: dict[str, Any], key: str) -> Optional[int]:
    sleep = day_obj.get("sleep")
    if not isinstance(sleep, dict):
        return None
    v = sleep.get(key)
    if isinstance(v, (int, float)) and int(v) != -1:
        return int(v)
    return None


def _sum_activity_fields(day_obj: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    Returns (distance_km, moving_time_hours, elevation_m, training_load_hours) for one day.
    training_load_hours is computed from moving_time * load_factor, so distance==0 can still contribute.
    """
    acts = day_obj.get("activities") or []
    if not isinstance(acts, list):
        return 0.0, 0.0, 0.0, 0.0

    dist_m = 0.0
    mv_s = 0.0
    elev_m = 0.0
    for a in acts:
        if not isinstance(a, dict):
            continue
        d = a.get("distance")
        if isinstance(d, (int, float)):
            dist_m += float(d)
        mv = a.get("moving_time")
        if isinstance(mv, (int, float)):
            mv_s += float(mv)
        el = a.get("total_elevation_gain")
        if isinstance(el, (int, float)):
            elev_m += float(el)

    tlh = float(day_training_load_hours(day_obj))
    return dist_m / 1000.0, mv_s / 3600.0, elev_m, tlh


def build_weekly_history(combined: list[dict[str, Any]], *, weeks: int) -> list[dict[str, Any]]:
    if not combined:
        return []

    last_date_raw = combined[-1].get("date")
    if not isinstance(last_date_raw, str):
        return []

    last_d = _as_date(last_date_raw)
    if not last_d:
        return []

    buckets: dict[str, dict[str, Any]] = {}

    for day_obj in combined:
        ds = day_obj.get("date")
        if not isinstance(ds, str):
            continue
        d = _as_date(ds)
        if not d:
            continue

        iso = d.isocalendar()
        week_id = f"{iso[0]}-W{int(iso[1]):02d}"

        bucket = buckets.get(week_id)
        if bucket is None:
            bucket = {
                "iso_week": week_id,
                "min_date": d,
                "max_date": d,
                "distance_km": 0.0,
                "moving_time_hours": 0.0,
                "elevation_m": 0.0,
                "training_load_hours": 0.0,
                "sleep_hours": [],
                "hrv": [],
                "rhr": [],
                "days_with_sleep": 0,
            }
            buckets[week_id] = bucket

        bucket["min_date"] = min(cast(date, bucket["min_date"]), d)
        bucket["max_date"] = max(cast(date, bucket["max_date"]), d)

        dk, mh, em, tlh = _sum_activity_fields(day_obj)
        bucket["distance_km"] = float(bucket["distance_km"]) + dk
        bucket["moving_time_hours"] = float(bucket["moving_time_hours"]) + mh
        bucket["elevation_m"] = float(bucket["elevation_m"]) + em
        bucket["training_load_hours"] = float(bucket["training_load_hours"]) + tlh

        sh = _sleep_hours(day_obj)
        if sh is not None:
            sleep_hours = _as_float_list(bucket["sleep_hours"])
            sleep_hours.append(sh)
            bucket["sleep_hours"] = sleep_hours
            bucket["days_with_sleep"] = int(bucket["days_with_sleep"]) + 1

        hrv = _sleep_int(day_obj, "avgOvernightHrv")
        if hrv is not None:
            hrv_vals = _as_float_list(bucket["hrv"])
            hrv_vals.append(float(hrv))
            bucket["hrv"] = hrv_vals

        rhr = _sleep_int(day_obj, "restingHeartRate")
        if rhr is not None:
            rhr_vals = _as_float_list(bucket["rhr"])
            rhr_vals.append(float(rhr))
            bucket["rhr"] = rhr_vals

    all_weeks = sorted(buckets.keys())
    keep = set(all_weeks[-weeks:]) if weeks > 0 and len(all_weeks) > weeks else set(all_weeks)

    out: list[dict[str, Any]] = []
    for week_id in all_weeks:
        if week_id not in keep:
            continue

        bucket = buckets[week_id]
        sleep_hours = _as_float_list(bucket["sleep_hours"])
        hrv_vals = _as_float_list(bucket["hrv"])
        rhr_vals = _as_float_list(bucket["rhr"])

        out.append(
            {
                "iso_week": str(bucket["iso_week"]),
                "date_range": f"{cast(date, bucket['min_date']).isoformat()}..{cast(date, bucket['max_date']).isoformat()}",
                "distance_km": round(float(bucket["distance_km"]), 3),
                "moving_time_hours": round(float(bucket["moving_time_hours"]), 3),
                "elevation_m": round(float(bucket["elevation_m"]), 1),
                "training_load_hours": round(float(bucket["training_load_hours"]), 3),
                "sleep_hours_mean": _round_or_none(_mean(sleep_hours), 2),
                "hrv_mean": _round_or_none(_mean(hrv_vals), 2),
                "rhr_mean": _round_or_none(_mean(rhr_vals), 2),
                "days_with_sleep": int(bucket["days_with_sleep"]),
            }
        )

    return out


def build_signal_registry(
    combined: list[dict[str, Any]], rollups: Optional[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not combined:
        return []

    last_date_raw = combined[-1].get("date")
    if not isinstance(last_date_raw, str):
        return []

    last_d = _as_date(last_date_raw)
    if not last_d:
        return []

    reg: list[dict[str, Any]] = []

    def add(signal_id: str, value: Any, source: str, date_range: str, unit: str = "") -> None:
        reg.append(
            {
                "signal_id": signal_id,
                "value": value,
                "unit": unit,
                "source": source,
                "date_range": date_range,
            }
        )

    if isinstance(rollups, dict):
        windows = _as_dict(rollups.get("windows"))

        w7_raw = windows.get("7")
        w7 = _as_dict(w7_raw)
        if w7:
            acts = _as_dict(w7.get("activities"))
            dr = f"{w7.get('start_date')}..{w7.get('end_date')}"
            add(
                "load.last7.distance_km",
                acts.get("total_distance_km"),
                "combined_rollups.json:windows.7.activities.total_distance_km",
                dr,
                "km",
            )
            add(
                "load.last7.moving_time_hours",
                acts.get("total_moving_time_hours"),
                "combined_rollups.json:windows.7.activities.total_moving_time_hours",
                dr,
                "h",
            )
            add(
                "load.last7.elevation_m",
                acts.get("total_elevation_m"),
                "combined_rollups.json:windows.7.activities.total_elevation_m",
                dr,
                "m",
            )
            add(
                "load.last7.training_load_hours",
                acts.get("total_training_load_hours"),
                "combined_rollups.json:windows.7.activities.total_training_load_hours",
                dr,
                "load_h",
            )
            add(
                "load.last7.activity_count",
                acts.get("count"),
                "combined_rollups.json:windows.7.activities.count",
                dr,
                "",
            )
            add(
                "load.last7.sleep_days_with_data",
                w7.get("sleep_days_with_data"),
                "combined_rollups.json:windows.7.sleep_days_with_data",
                dr,
                "days",
            )

        w28_raw = windows.get("28")
        w28 = _as_dict(w28_raw)
        if w28:
            acts = _as_dict(w28.get("activities"))
            dr = f"{w28.get('start_date')}..{w28.get('end_date')}"
            add(
                "load.baseline28.distance_km",
                acts.get("total_distance_km"),
                "combined_rollups.json:windows.28.activities.total_distance_km",
                dr,
                "km",
            )
            add(
                "load.baseline28.moving_time_hours",
                acts.get("total_moving_time_hours"),
                "combined_rollups.json:windows.28.activities.total_moving_time_hours",
                dr,
                "h",
            )
            add(
                "load.baseline28.elevation_m",
                acts.get("total_elevation_m"),
                "combined_rollups.json:windows.28.activities.total_elevation_m",
                dr,
                "m",
            )
            add(
                "load.baseline28.training_load_hours",
                acts.get("total_training_load_hours"),
                "combined_rollups.json:windows.28.activities.total_training_load_hours",
                dr,
                "load_h",
            )

    def window(days: int) -> list[dict[str, Any]]:
        start = last_d - timedelta(days=days - 1)
        out_days: list[dict[str, Any]] = []
        for day_obj in combined:
            ds = day_obj.get("date")
            if not isinstance(ds, str):
                continue
            dd = _as_date(ds)
            if dd and start <= dd <= last_d:
                out_days.append(day_obj)
        return out_days

    w7_days = window(7)
    w28_days = window(28)

    sleep7_vals = [_sleep_hours(day_obj) for day_obj in w7_days]
    sleep28_vals = [_sleep_hours(day_obj) for day_obj in w28_days]
    hrv7_vals = [
        float(v)
        for v in (_sleep_int(day_obj, "avgOvernightHrv") for day_obj in w7_days)
        if v is not None
    ]
    rhr7_vals = [
        float(v)
        for v in (_sleep_int(day_obj, "restingHeartRate") for day_obj in w7_days)
        if v is not None
    ]

    sleep7 = _mean(sleep7_vals)
    sleep28 = _mean(sleep28_vals)
    hrv7 = _mean(hrv7_vals)
    rhr7 = _mean(rhr7_vals)

    add(
        "recovery.last7.sleep_hours_mean",
        _round_or_none(sleep7, 2),
        "combined_summary.json:sleep.sleepTimeSeconds",
        f"{(last_d - timedelta(days=6)).isoformat()}..{last_d.isoformat()}",
        "h",
    )
    add(
        "recovery.last28.sleep_hours_mean",
        _round_or_none(sleep28, 2),
        "combined_summary.json:sleep.sleepTimeSeconds",
        f"{(last_d - timedelta(days=27)).isoformat()}..{last_d.isoformat()}",
        "h",
    )
    add(
        "recovery.last7.hrv_mean",
        _round_or_none(hrv7, 2),
        "combined_summary.json:sleep.avgOvernightHrv",
        f"{(last_d - timedelta(days=6)).isoformat()}..{last_d.isoformat()}",
        "ms",
    )
    add(
        "recovery.last7.rhr_mean",
        _round_or_none(rhr7, 2),
        "combined_summary.json:sleep.restingHeartRate",
        f"{(last_d - timedelta(days=6)).isoformat()}..{last_d.isoformat()}",
        "bpm",
    )

    return reg


def build_retrieval_context(
    combined: list[dict[str, Any]],
    rollups: Optional[dict[str, Any]],
    *,
    retrieval_weeks: int,
) -> dict[str, Any]:
    return {
        "weekly_history": build_weekly_history(combined, weeks=retrieval_weeks),
        "signal_registry": build_signal_registry(
            combined, rollups if isinstance(rollups, dict) else None
        ),
    }
