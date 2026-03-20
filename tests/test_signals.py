# src/trailtraining/llm/signals.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

from trailtraining.metrics.training_load import day_training_load_hours, latest_atl_ctl_tsb
from trailtraining.util.dates import _as_date


@dataclass
class _WeekBucket:
    iso_week: str
    min_date: date
    max_date: date
    distance_km: float = 0.0
    moving_time_hours: float = 0.0
    elevation_m: float = 0.0
    training_load_hours: float = 0.0
    sleep_hours: list[float] = field(default_factory=list)
    hrv: list[float] = field(default_factory=list)
    rhr: list[float] = field(default_factory=list)
    days_with_sleep: int = 0


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _mean(xs: Sequence[float | None]) -> Optional[float]:
    xs2 = [float(x) for x in xs if x is not None]
    if not xs2:
        return None
    return sum(xs2) / len(xs2)


def _round_or_none(x: Optional[float], ndigits: int = 2) -> Optional[float]:
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

    buckets: dict[str, _WeekBucket] = {}

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
            bucket = _WeekBucket(iso_week=week_id, min_date=d, max_date=d)
            buckets[week_id] = bucket

        bucket.min_date = min(bucket.min_date, d)
        bucket.max_date = max(bucket.max_date, d)

        dk, mh, em, tlh = _sum_activity_fields(day_obj)
        bucket.distance_km += dk
        bucket.moving_time_hours += mh
        bucket.elevation_m += em
        bucket.training_load_hours += tlh

        sh = _sleep_hours(day_obj)
        if sh is not None:
            bucket.sleep_hours.append(sh)
            bucket.days_with_sleep += 1

        hrv = _sleep_int(day_obj, "avgOvernightHrv")
        if hrv is not None:
            bucket.hrv.append(float(hrv))

        rhr = _sleep_int(day_obj, "restingHeartRate")
        if rhr is not None:
            bucket.rhr.append(float(rhr))

    all_weeks = sorted(buckets.keys())
    keep = set(all_weeks[-weeks:]) if weeks > 0 and len(all_weeks) > weeks else set(all_weeks)

    out: list[dict[str, Any]] = []
    for week_id in all_weeks:
        if week_id not in keep:
            continue

        bucket = buckets[week_id]
        out.append(
            {
                "iso_week": bucket.iso_week,
                "date_range": f"{bucket.min_date.isoformat()}..{bucket.max_date.isoformat()}",
                "distance_km": round(bucket.distance_km, 3),
                "moving_time_hours": round(bucket.moving_time_hours, 3),
                "elevation_m": round(bucket.elevation_m, 1),
                "training_load_hours": round(bucket.training_load_hours, 3),
                "sleep_hours_mean": _round_or_none(_mean(bucket.sleep_hours), 2),
                "hrv_mean": _round_or_none(_mean(bucket.hrv), 2),
                "rhr_mean": _round_or_none(_mean(bucket.rhr), 2),
                "days_with_sleep": bucket.days_with_sleep,
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

    # Load window signals from rollups if present
    if isinstance(rollups, dict):
        windows = _as_dict(rollups.get("windows"))

        w7 = _as_dict(windows.get("7"))
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

        w28 = _as_dict(windows.get("28"))
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

    # ATL / CTL / TSB
    load_model: dict[str, Any] = {}
    load_model_source = "combined_summary.json:derived.latest_atl_ctl_tsb"

    if isinstance(rollups, dict):
        lm = _as_dict(rollups.get("load_model"))
        if lm:
            load_model = lm
            load_model_source = "combined_rollups.json:load_model"

    if not load_model:
        try:
            computed = latest_atl_ctl_tsb(combined)
        except Exception:
            computed = None
        load_model = _as_dict(computed)

    if load_model:
        dr = f"{last_d.isoformat()}..{last_d.isoformat()}"

        add(
            "load.model.atl_hours",
            load_model.get("atl_load_h"),
            f"{load_model_source}.atl_load_h",
            dr,
            "load_h",
        )
        add(
            "load.model.ctl_hours",
            load_model.get("ctl_load_h"),
            f"{load_model_source}.ctl_load_h",
            dr,
            "load_h",
        )
        add(
            "load.model.tsb_hours",
            load_model.get("tsb_load_h"),
            f"{load_model_source}.tsb_load_h",
            dr,
            "load_h",
        )

        if "tau_atl_days" in load_model:
            add(
                "load.model.atl_tau_days",
                load_model.get("tau_atl_days"),
                f"{load_model_source}.tau_atl_days",
                dr,
                "days",
            )
        if "tau_ctl_days" in load_model:
            add(
                "load.model.ctl_tau_days",
                load_model.get("tau_ctl_days"),
                f"{load_model_source}.tau_ctl_days",
                dr,
                "days",
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
