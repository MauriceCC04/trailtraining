from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from trailtraining.metrics.training_load import activity_training_load_hours
from trailtraining.util.state import load_json, save_json

DISCIPLINE_MAP: dict[str, str] = {
    "run": "running",
    "trailrun": "trailrunning",
    "ride": "cycling",
    "virtualride": "cycling",
    "ebikeride": "cycling",
    "mountainbikeride": "cycling",
    "gravelride": "cycling",
    "handcycle": "cycling",
    "velomobile": "cycling",
    "swim": "swimming",
    "openwaterswim": "swimming",
    "walk": "walking",
    "hike": "hiking",
    "alpineski": "skiing",
    "backcountryski": "skiing",
    "nordicski": "skiing",
    "snowboard": "skiing",
    "rowing": "rowing",
    "virtualrow": "rowing",
    "workout": "strength",
    "weighttraining": "strength",
    "crossfit": "strength",
    "yoga": "mobility",
    "elliptical": "elliptical",
    "stairstepper": "stairs",
    "rockclimbing": "climbing",
}

FAMILY_MAP: dict[str, str] = {
    "running": "running",
    "trailrunning": "running",
    "cycling": "cycling",
    "swimming": "swimming",
    "walking": "walking",
    "hiking": "hiking",
    "skiing": "skiing",
    "rowing": "rowing",
    "strength": "strength",
    "mobility": "mobility",
    "elliptical": "elliptical",
    "stairs": "stairs",
    "climbing": "climbing",
}


def _as_date(s: Any) -> date | None:
    if not isinstance(s, str) or len(s) < 10:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _latest_valid_date(values: list[Any]) -> date:
    valid_dates: list[date] = []
    for value in values:
        d = _as_date(value)
        if d is not None:
            valid_dates.append(d)
    return max(valid_dates) if valid_dates else date.today()


def _canonical_discipline(activity: dict[str, Any]) -> str:
    raw = str(activity.get("sport_type") or activity.get("type") or "unknown").strip()
    if not raw:
        return "unknown"
    key = "".join(ch for ch in raw.lower() if ch.isalnum())
    return DISCIPLINE_MAP.get(key, raw.lower())


def _sport_family(discipline: str) -> str:
    return FAMILY_MAP.get(discipline, discipline)


def _activity_date(day_date: date, activity: dict[str, Any]) -> date:
    for key in ("start_date_local", "start_date"):
        d = _as_date(activity.get(key))
        if d:
            return d
    return day_date


def _float_or_zero(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _fallback_activity_key(day_date: date, activity: dict[str, Any]) -> str:
    parts = [
        day_date.isoformat(),
        str(activity.get("start_date_local") or activity.get("start_date") or ""),
        str(activity.get("sport_type") or activity.get("type") or ""),
        str(activity.get("name") or ""),
        str(activity.get("distance") or ""),
        str(activity.get("moving_time") or ""),
    ]
    return "|".join(parts)


def _dedup_flatten_activities(
    combined: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], date | None]:
    seen: set[str] = set()
    flat: list[dict[str, Any]] = []
    as_of_date: date | None = None

    for day in combined:
        day_date = _as_date(day.get("date"))
        if not day_date:
            continue
        as_of_date = max(as_of_date, day_date) if as_of_date else day_date

        acts = day.get("activities")
        if not isinstance(acts, list):
            continue

        for activity in acts:
            if not isinstance(activity, dict):
                continue

            aid = activity.get("id")
            key = f"id:{aid}" if aid is not None else _fallback_activity_key(day_date, activity)
            if key in seen:
                continue
            seen.add(key)

            activity_date = _activity_date(day_date, activity)
            discipline = _canonical_discipline(activity)
            family = _sport_family(discipline)
            distance_m = _float_or_zero(activity.get("distance"))
            moving_s = _float_or_zero(activity.get("moving_time"))
            elevation_m = _float_or_zero(activity.get("total_elevation_gain"))
            training_load_h = float(activity_training_load_hours(activity))
            load_basis = training_load_h
            if load_basis <= 0 and moving_s > 0:
                load_basis = moving_s / 3600.0
            if load_basis <= 0 and distance_m > 0:
                load_basis = distance_m / 1000.0

            flat.append(
                {
                    "date": activity_date,
                    "discipline": discipline,
                    "family": family,
                    "distance_m": distance_m,
                    "moving_s": moving_s,
                    "elevation_m": elevation_m,
                    "training_load_h": training_load_h,
                    "load_basis": load_basis,
                }
            )

    flat.sort(key=lambda x: (x["date"], x["discipline"], x["distance_m"], x["moving_s"]))
    return flat, as_of_date


def _build_sports_stats(flat: list[dict[str, Any]], as_of_date: date) -> dict[str, Any]:
    by_sport: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "first_observed_date": None,
            "last_observed_date": None,
            "activity_count": 0,
            "total_distance_m": 0.0,
            "total_moving_s": 0.0,
            "total_elevation_m": 0.0,
            "total_training_load_h": 0.0,
        }
    )

    for item in flat:
        discipline = item["discipline"]
        agg = by_sport[discipline]
        d = item["date"]
        first = agg["first_observed_date"]
        last = agg["last_observed_date"]
        agg["first_observed_date"] = d if first is None or d < first else first
        agg["last_observed_date"] = d if last is None or d > last else last
        agg["activity_count"] += 1
        agg["total_distance_m"] += item["distance_m"]
        agg["total_moving_s"] += item["moving_s"]
        agg["total_elevation_m"] += item["elevation_m"]
        agg["total_training_load_h"] += item["training_load_h"]

    out: dict[str, Any] = {}
    for discipline in sorted(by_sport.keys()):
        agg = by_sport[discipline]
        first = agg["first_observed_date"]
        last = agg["last_observed_date"]
        claimed_years = None
        if isinstance(first, date):
            claimed_years = round(max(0.0, (as_of_date - first).days / 365.25), 2)
        out[discipline] = {
            "sport_family": _sport_family(discipline),
            "first_observed_date": first.isoformat() if isinstance(first, date) else None,
            "last_observed_date": last.isoformat() if isinstance(last, date) else None,
            "claimed_years_sport": claimed_years,
            "activity_count": int(agg["activity_count"]),
            "total_distance_km": round(agg["total_distance_m"] / 1000.0, 3),
            "total_moving_time_hours": round(agg["total_moving_s"] / 3600.0, 3),
            "total_elevation_m": round(agg["total_elevation_m"], 1),
            "total_training_load_hours": round(agg["total_training_load_h"], 3),
        }
    return out


def _top_sports_for_window(
    flat: list[dict[str, Any]], *, as_of_date: date, window_days: int
) -> dict[str, Any]:
    start_date = as_of_date - timedelta(days=window_days - 1)
    filtered = [x for x in flat if start_date <= x["date"] <= as_of_date]

    discipline_loads: dict[str, dict[str, float]] = defaultdict(lambda: {"load": 0.0, "count": 0.0})
    family_loads: dict[str, dict[str, float]] = defaultdict(lambda: {"load": 0.0, "count": 0.0})

    for item in filtered:
        discipline = item["discipline"]
        family = item["family"]
        discipline_loads[discipline]["load"] += item["load_basis"]
        discipline_loads[discipline]["count"] += 1.0
        family_loads[family]["load"] += item["load_basis"]
        family_loads[family]["count"] += 1.0

    def _pick_top(agg: dict[str, dict[str, float]]) -> list[str]:
        if not agg:
            return []
        total_load = sum(max(0.0, v["load"]) for v in agg.values())
        ranked = sorted(agg.items(), key=lambda kv: (-kv[1]["load"], -kv[1]["count"], kv[0]))
        chosen: list[str] = []
        for name, values in ranked:
            share = (values["load"] / total_load) if total_load > 0 else 0.0
            if values["count"] >= 3 and (share >= 0.15 or not chosen):
                chosen.append(name)
        if not chosen and ranked:
            chosen.append(ranked[0][0])
        return chosen

    top_disciplines = _pick_top(discipline_loads)
    top_families = _pick_top(family_loads)
    primary_discipline = top_disciplines[0] if top_disciplines else None
    primary_family = top_families[0] if top_families else None

    def _meets_tri(name: str) -> bool:
        values = family_loads.get(name) or {}
        total_load = sum(max(0.0, v["load"]) for v in family_loads.values())
        share = (float(values.get("load", 0.0)) / total_load) if total_load > 0 else 0.0
        count = int(values.get("count", 0.0))
        return share >= 0.10 and count >= 6

    if all(_meets_tri(k) for k in ("running", "cycling", "swimming")):
        profile_style = "triathlon"
    elif primary_discipline == "trailrunning":
        profile_style = "trailrunning"
    elif primary_family:
        profile_style = primary_family
    elif primary_discipline:
        profile_style = primary_discipline
    else:
        profile_style = "unknown"

    return {
        "window_days": window_days,
        "start_date": start_date.isoformat(),
        "end_date": as_of_date.isoformat(),
        "top_sports": top_disciplines,
        "top_sport_families": top_families,
        "top_sport_disciplines": top_disciplines,
        "primary_sport_family": primary_family,
        "primary_sport_discipline": primary_discipline,
        "profile_style": profile_style,
    }


def _daily_metric_maps(
    flat: list[dict[str, Any]],
) -> tuple[dict[date, dict[str, float]], dict[date, dict[str, float]]]:
    all_daily: dict[date, dict[str, float]] = defaultdict(
        lambda: {
            "training_load_h": 0.0,
            "distance_m": 0.0,
            "elevation_m": 0.0,
        }
    )
    running_daily: dict[date, dict[str, float]] = defaultdict(
        lambda: {
            "training_load_h": 0.0,
            "distance_m": 0.0,
            "elevation_m": 0.0,
        }
    )

    for item in flat:
        d = item["date"]
        all_daily[d]["training_load_h"] += item["training_load_h"]
        all_daily[d]["distance_m"] += item["distance_m"]
        all_daily[d]["elevation_m"] += item["elevation_m"]

        if item["family"] == "running":
            running_daily[d]["training_load_h"] += item["training_load_h"]
            running_daily[d]["distance_m"] += item["distance_m"]
            running_daily[d]["elevation_m"] += item["elevation_m"]

    return all_daily, running_daily


def _peak_rolling_metrics(
    daily: dict[date, dict[str, float]],
    *,
    as_of_date: date,
    history_window_days: int,
) -> dict[str, Any]:
    history_start = as_of_date - timedelta(days=history_window_days - 1)
    day_count = (as_of_date - history_start).days + 1
    ordered_days = [history_start + timedelta(days=i) for i in range(day_count)]

    def _rolling_peak(metric: str, width: int) -> float:
        values = [float((daily.get(d) or {}).get(metric, 0.0)) for d in ordered_days]
        if not values:
            return 0.0
        window_sum = 0.0
        best = 0.0
        for i, value in enumerate(values):
            window_sum += value
            if i >= width:
                window_sum -= values[i - width]
            if i >= width - 1:
                best = max(best, window_sum)
        return best

    return {
        "peak_7d_training_load_hours": round(_rolling_peak("training_load_h", 7), 3),
        "peak_28d_training_load_hours": round(_rolling_peak("training_load_h", 28), 3),
        "peak_7d_distance_km": round(_rolling_peak("distance_m", 7) / 1000.0, 3),
        "peak_28d_distance_km": round(_rolling_peak("distance_m", 28) / 1000.0, 3),
        "peak_7d_elevation_m": round(_rolling_peak("elevation_m", 7), 1),
        "peak_28d_elevation_m": round(_rolling_peak("elevation_m", 28), 1),
    }


def _build_historical_capacities(flat: list[dict[str, Any]], *, as_of_date: date) -> dict[str, Any]:
    all_daily, running_daily = _daily_metric_maps(flat)
    out: dict[str, Any] = {}
    for history_days in (90, 365):
        key = f"{history_days}d"
        out[key] = {
            "all_sports": _peak_rolling_metrics(
                all_daily,
                as_of_date=as_of_date,
                history_window_days=history_days,
            ),
            "running_family": _peak_rolling_metrics(
                running_daily,
                as_of_date=as_of_date,
                history_window_days=history_days,
            ),
        }
    return out


def _merge_base_profile(base: dict[str, Any], derived: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    if not isinstance(merged.get("userInfo"), dict):
        merged["userInfo"] = {}
    if not isinstance(merged.get("biometricProfile"), dict):
        merged["biometricProfile"] = {}
    merged["derived_activity_profile"] = derived
    return merged


def build_formatted_personal_profile(
    *,
    combined_summary_path: str | Path,
    output_path: str | Path,
    base_personal_path: str | Path | None = None,
) -> dict[str, Any]:
    combined = load_json(combined_summary_path, default=[])
    if not isinstance(combined, list):
        raise RuntimeError("combined_summary.json must be a list")

    base: dict[str, Any] = {}
    if base_personal_path is not None:
        loaded = load_json(base_personal_path, default={})
        if isinstance(loaded, dict):
            base = loaded

    flat, as_of_date = _dedup_flatten_activities(combined)
    if as_of_date is None:
        as_of_date = date.today()

    observed_first = flat[0]["date"] if flat else None
    observed_last = flat[-1]["date"] if flat else None
    sports = _build_sports_stats(flat, as_of_date=as_of_date) if flat else {}

    sources: list[str] = ["combined_summary"]
    if isinstance(base.get("userInfo"), dict) and base.get("userInfo"):
        sources.append("garmin_user_info")
    if isinstance(base.get("biometricProfile"), dict) and base.get("biometricProfile"):
        sources.append("garmin_biometric_profile")

    derived_activity_profile: dict[str, Any] = {
        "observed_window": {
            "as_of_date": as_of_date.isoformat(),
            "first_activity_date": observed_first.isoformat() if observed_first else None,
            "last_activity_date": observed_last.isoformat() if observed_last else None,
            "activity_span_days": (
                ((observed_last - observed_first).days + 1)
                if observed_first is not None and observed_last is not None
                else 0
            ),
        },
        "sports": sports,
        "top_sports": {
            "90d": _top_sports_for_window(flat, as_of_date=as_of_date, window_days=90),
            "365d": _top_sports_for_window(flat, as_of_date=as_of_date, window_days=365),
        }
        if flat
        else {},
        "historical_capacities": _build_historical_capacities(flat, as_of_date=as_of_date)
        if flat
        else {
            "90d": {"all_sports": {}, "running_family": {}},
            "365d": {"all_sports": {}, "running_family": {}},
        },
    }

    merged = _merge_base_profile(base, derived_activity_profile)
    merged["profile_metadata"] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generated_from": sources,
        "activity_history_assumption": (
            "claimed_years_sport assumes the athlete had Strava when they started each sport"
        ),
    }

    save_json(output_path, merged, compact=False)
    return merged
