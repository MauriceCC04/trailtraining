# src/trailtraining/llm/signals.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

from trailtraining.metrics.training_load import (
    day_training_load_hours,
    latest_atl_ctl_tsb,
)


def _as_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _mean(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


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
    # Treat -1 as missing (per prompts)
    if isinstance(v, (int, float)) and int(v) != -1:
        return int(v)
    return None


def _sum_activity_fields(day_obj: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    Returns (distance_km, moving_time_hours, elevation_m, training_load_hours) for one day.

    training_load_hours is computed from moving_time * load_factor, so distance==0 can still
    contribute.
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


def _window_days(
    combined: list[dict[str, Any]],
    last_d: date,
    days: int,
) -> list[dict[str, Any]]:
    start = last_d - timedelta(days=days - 1)
    out: list[dict[str, Any]] = []

    for d in combined:
        ds = d.get("date")
        if not isinstance(ds, str):
            continue
        dd = _as_date(ds)
        if dd and start <= dd <= last_d:
            out.append(d)

    return out


def _summarize_window(days: list[dict[str, Any]]) -> dict[str, Any]:
    total_distance_km = 0.0
    total_moving_time_hours = 0.0
    total_elevation_m = 0.0
    total_training_load_hours = 0.0
    activity_count = 0
    sleep_days_with_data = 0

    for day_obj in days:
        acts = day_obj.get("activities") or []
        if isinstance(acts, list):
            activity_count += sum(1 for a in acts if isinstance(a, dict))

        dk, mh, em, tlh = _sum_activity_fields(day_obj)
        total_distance_km += dk
        total_moving_time_hours += mh
        total_elevation_m += em
        total_training_load_hours += tlh

        if _sleep_hours(day_obj) is not None:
            sleep_days_with_data += 1

    return {
        "total_distance_km": round(float(total_distance_km), 3),
        "total_moving_time_hours": round(float(total_moving_time_hours), 3),
        "total_elevation_m": round(float(total_elevation_m), 1),
        "total_training_load_hours": round(float(total_training_load_hours), 3),
        "count": int(activity_count),
        "sleep_days_with_data": int(sleep_days_with_data),
    }


def build_weekly_history(combined: list[dict[str, Any]], *, weeks: int) -> list[dict[str, Any]]:
    """
    Weekly retrieval summary for the last N weeks.

    Output: list sorted oldest -> newest:
    {
      "iso_week": "2026-W10",
      "date_range": "YYYY-MM-DD..YYYY-MM-DD",
      "distance_km": float,
      "moving_time_hours": float,
      "elevation_m": float,
      "training_load_hours": float,
      "sleep_hours_mean": float|None,
      "hrv_mean": float|None,
      "rhr_mean": float|None,
      "days_with_sleep": int
    }
    """
    if not combined:
        return []

    last_d = _as_date(combined[-1].get("date", ""))
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

        b = buckets.get(week_id)
        if b is None:
            b = {
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
            buckets[week_id] = b

        b["min_date"] = min(b["min_date"], d)
        b["max_date"] = max(b["max_date"], d)

        dk, mh, em, tlh = _sum_activity_fields(day_obj)
        b["distance_km"] += dk
        b["moving_time_hours"] += mh
        b["elevation_m"] += em
        b["training_load_hours"] += tlh

        sh = _sleep_hours(day_obj)
        if sh is not None:
            b["sleep_hours"].append(sh)
            b["days_with_sleep"] += 1

        hrv = _sleep_int(day_obj, "avgOvernightHrv")
        if hrv is not None:
            b["hrv"].append(float(hrv))

        rhr = _sleep_int(day_obj, "restingHeartRate")
        if rhr is not None:
            b["rhr"].append(float(rhr))

    all_weeks = sorted(buckets.keys())
    keep = set(all_weeks[-weeks:]) if weeks > 0 and len(all_weeks) > weeks else set(all_weeks)

    out: list[dict[str, Any]] = []
    for w in all_weeks:
        if w not in keep:
            continue
        b = buckets[w]
        out.append(
            {
                "iso_week": b["iso_week"],
                "date_range": f"{b['min_date'].isoformat()}..{b['max_date'].isoformat()}",
                "distance_km": round(float(b["distance_km"]), 3),
                "moving_time_hours": round(float(b["moving_time_hours"]), 3),
                "elevation_m": round(float(b["elevation_m"]), 1),
                "training_load_hours": round(float(b["training_load_hours"]), 3),
                "sleep_hours_mean": _round_or_none(_mean(b["sleep_hours"]), 2),
                "hrv_mean": _round_or_none(_mean(b["hrv"]), 2),
                "rhr_mean": _round_or_none(_mean(b["rhr"]), 2),
                "days_with_sleep": int(b["days_with_sleep"]),
            }
        )

    return out


def build_signal_registry(
    combined: list[dict[str, Any]],
    rollups: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Creates a citeable registry of signals.

    The coach must reference signal_ids from this list.
    """
    if not combined:
        return []

    last_d = _as_date(combined[-1].get("date", ""))
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

    w7_days = _window_days(combined, last_d, 7)
    w28_days = _window_days(combined, last_d, 28)

    # Rollups (authoritative if present); otherwise derive from combined_summary.
    have_w7 = False
    have_w28 = False

    try:
        if isinstance(rollups, dict):
            windows = _as_dict(rollups.get("windows"))
            w7 = _as_dict(windows.get("7"))
            w28 = _as_dict(windows.get("28"))

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
                have_w7 = True

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
                have_w28 = True
    except Exception:
        pass

    if not have_w7:
        s7 = _summarize_window(w7_days)
        dr7 = f"{(last_d - timedelta(days=6)).isoformat()}..{last_d.isoformat()}"
        add(
            "load.last7.distance_km",
            s7["total_distance_km"],
            "combined_summary.json:activities.distance",
            dr7,
            "km",
        )
        add(
            "load.last7.moving_time_hours",
            s7["total_moving_time_hours"],
            "combined_summary.json:activities.moving_time",
            dr7,
            "h",
        )
        add(
            "load.last7.elevation_m",
            s7["total_elevation_m"],
            "combined_summary.json:activities.total_elevation_gain",
            dr7,
            "m",
        )
        add(
            "load.last7.training_load_hours",
            s7["total_training_load_hours"],
            "combined_summary.json:derived.day_training_load_hours",
            dr7,
            "load_h",
        )
        add(
            "load.last7.activity_count",
            s7["count"],
            "combined_summary.json:activities",
            dr7,
            "",
        )
        add(
            "load.last7.sleep_days_with_data",
            s7["sleep_days_with_data"],
            "combined_summary.json:sleep.sleepTimeSeconds",
            dr7,
            "days",
        )

    if not have_w28:
        s28 = _summarize_window(w28_days)
        dr28 = f"{(last_d - timedelta(days=27)).isoformat()}..{last_d.isoformat()}"
        add(
            "load.baseline28.distance_km",
            s28["total_distance_km"],
            "combined_summary.json:activities.distance",
            dr28,
            "km",
        )
        add(
            "load.baseline28.moving_time_hours",
            s28["total_moving_time_hours"],
            "combined_summary.json:activities.moving_time",
            dr28,
            "h",
        )
        add(
            "load.baseline28.elevation_m",
            s28["total_elevation_m"],
            "combined_summary.json:activities.total_elevation_gain",
            dr28,
            "m",
        )
        add(
            "load.baseline28.training_load_hours",
            s28["total_training_load_hours"],
            "combined_summary.json:derived.day_training_load_hours",
            dr28,
            "load_h",
        )

    # ATL / CTL / TSB model state.
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

    # Derived recovery signals from combined_summary (last 7 / 28).
    sleep7_vals = [v for v in (_sleep_hours(d) for d in w7_days) if v is not None]
    sleep28_vals = [v for v in (_sleep_hours(d) for d in w28_days) if v is not None]
    hrv7_vals = [
        float(v) for v in (_sleep_int(d, "avgOvernightHrv") for d in w7_days) if v is not None
    ]
    rhr7_vals = [
        float(v) for v in (_sleep_int(d, "restingHeartRate") for d in w7_days) if v is not None
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
            combined,
            rollups if isinstance(rollups, dict) else None,
        ),
    }
