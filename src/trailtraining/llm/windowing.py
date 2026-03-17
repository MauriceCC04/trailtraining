from __future__ import annotations

from typing import Any, Optional

from trailtraining.util.dates import _as_date


def normalize_plan_days(plan_obj: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (plan_obj.get("plan") or {}).get("days")
    if not isinstance(raw, list):
        return []

    days = [d for d in raw if isinstance(d, dict)]

    def key(d: dict[str, Any]) -> tuple[int, str]:
        dd = _as_date(d.get("date"))
        return (0, dd.isoformat()) if dd else (1, str(d.get("date") or "9999-99-99"))

    return sorted(days, key=key)


def rolling_windows(days: list[dict[str, Any]], *, size: int = 7) -> list[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if not days:
        return []
    if len(days) <= size:
        return [days]
    return [days[i : i + size] for i in range(0, len(days) - size + 1)]


def extract_last7_hours(rollups: Optional[dict[str, Any]]) -> Optional[float]:
    windows = (rollups or {}).get("windows")
    if not isinstance(windows, dict):
        return None

    seven = windows.get("7")
    if not isinstance(seven, dict):
        return None

    activities = seven.get("activities")
    if not isinstance(activities, dict):
        return None

    value = activities.get("total_moving_time_hours")
    return float(value) if isinstance(value, (int, float)) else None
