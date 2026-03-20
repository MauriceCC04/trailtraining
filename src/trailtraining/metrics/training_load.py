# src/trailtraining/metrics/training_load.py
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def activity_load_factor(activity: dict[str, Any]) -> float:
    avg = activity.get("average_heartrate")
    mx = activity.get("max_heartrate")

    if isinstance(avg, (int, float)) and isinstance(mx, (int, float)) and float(mx) > 0:
        r = float(avg) / float(mx)
        r = _clamp(r, 0.0, 1.0)
        return 0.5 + 2.0 * r

    return 1.0


def activity_training_load_hours(activity: dict[str, Any]) -> float:
    mv = activity.get("moving_time")
    if not isinstance(mv, (int, float)) or float(mv) <= 0:
        return 0.0

    moving_hours = float(mv) / 3600.0
    return moving_hours * float(activity_load_factor(activity))


def day_training_load_hours(day_obj: dict[str, Any]) -> float:
    acts = day_obj.get("activities") or []
    if not isinstance(acts, list):
        return 0.0
    total = 0.0
    for a in acts:
        if isinstance(a, dict):
            total += activity_training_load_hours(a)
    return total


@dataclass(frozen=True)
class LoadModelPoint:
    date: str
    load_h: float
    atl_load_h: float
    ctl_load_h: float
    tsb_load_h: float


def _exp_alpha(tau_days: float) -> float:
    if tau_days <= 0:
        raise ValueError("tau_days must be > 0")
    return 1.0 - math.exp(-1.0 / tau_days)


def _ewma_update(prev: float, x: float, tau_days: float) -> float:
    a = _exp_alpha(tau_days)
    return prev + a * (x - prev)


def build_daily_training_load_series(
    combined: list[dict[str, Any]],
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for day in combined:
        ds = day.get("date")
        if isinstance(ds, str):
            out.append((ds, float(day_training_load_hours(day))))
    return out


def build_atl_ctl_tsb_series(
    combined: list[dict[str, Any]],
    *,
    tau_atl_days: float = 7.0,
    tau_ctl_days: float = 42.0,
) -> list[LoadModelPoint]:
    series = build_daily_training_load_series(combined)
    if not series:
        return []

    # Seed from first observed daily load to reduce startup bias.
    first_date, first_load = series[0]
    atl = first_load
    ctl = first_load

    out: list[LoadModelPoint] = [
        LoadModelPoint(
            date=first_date,
            load_h=round(first_load, 3),
            atl_load_h=round(atl, 3),
            ctl_load_h=round(ctl, 3),
            tsb_load_h=0.0,
        )
    ]

    for ds, load_h in series[1:]:
        atl = _ewma_update(atl, load_h, tau_atl_days)
        ctl = _ewma_update(ctl, load_h, tau_ctl_days)
        tsb = ctl - atl
        out.append(
            LoadModelPoint(
                date=ds,
                load_h=round(load_h, 3),
                atl_load_h=round(atl, 3),
                ctl_load_h=round(ctl, 3),
                tsb_load_h=round(tsb, 3),
            )
        )

    return out


def latest_atl_ctl_tsb(
    combined: list[dict[str, Any]],
    *,
    tau_atl_days: float = 7.0,
    tau_ctl_days: float = 42.0,
) -> Optional[dict[str, Any]]:
    series = build_atl_ctl_tsb_series(
        combined,
        tau_atl_days=tau_atl_days,
        tau_ctl_days=tau_ctl_days,
    )
    if not series:
        return None

    last = asdict(series[-1])
    last["tau_atl_days"] = tau_atl_days
    last["tau_ctl_days"] = tau_ctl_days
    last["metric"] = "training_load_hours"
    last["unit"] = "load_h"
    return last
