# src/trailtraining/metrics/training_load.py
from __future__ import annotations

from typing import Any


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def activity_load_factor(activity: dict[str, Any]) -> float:
    """
    Intensity proxy ("load") used for training load:
      training_load_hours = moving_time_hours * activity_load_factor

    We keep this deliberately simple + robust:
    - If avg HR and max HR exist: use ratio avg/max and map to ~[0.5..2.5]
    - Else: default 1.0

    This ensures distance==0 but time>0 activities still contribute (via moving_time).
    """
    avg = activity.get("average_heartrate")
    mx = activity.get("max_heartrate")

    if isinstance(avg, (int, float)) and isinstance(mx, (int, float)) and float(mx) > 0:
        r = float(avg) / float(mx)
        r = _clamp(r, 0.0, 1.0)  # keep sane even if data is noisy
        # Map HR ratio to a modest multiplier (easy~0.5, hard~2.5)
        return 0.5 + 2.0 * r

    return 1.0


def activity_training_load_hours(activity: dict[str, Any]) -> float:
    """
    Returns training load in "load-hours":
      moving_time_hours * load_factor
    """
    mv = activity.get("moving_time")
    if not isinstance(mv, (int, float)) or float(mv) <= 0:
        return 0.0

    moving_hours = float(mv) / 3600.0
    return moving_hours * float(activity_load_factor(activity))


def day_training_load_hours(day_obj: dict[str, Any]) -> float:
    """
    Sum of activity_training_load_hours across a day record from combined_summary.json.
    """
    acts = day_obj.get("activities") or []
    if not isinstance(acts, list):
        return 0.0
    total = 0.0
    for a in acts:
        if isinstance(a, dict):
            total += activity_training_load_hours(a)
    return total
