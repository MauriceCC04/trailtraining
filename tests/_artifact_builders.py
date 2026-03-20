from __future__ import annotations

from copy import deepcopy
from typing import Any


def make_training_plan_artifact(*, plan_days: int = 7) -> dict[str, Any]:
    days = []
    session_types = ["easy", "tempo", "easy", "rest", "long", "easy", "strength"]
    hard_days = [False, True, False, False, False, False, False]
    rest_days = [False, False, False, True, False, False, False]
    durations = [45, 50, 45, 0, 90, 40, 30]

    for idx in range(plan_days):
        base_i = idx % len(session_types)
        day_no = idx + 1
        days.append(
            {
                "date": f"2026-03-{day_no:02d}",
                "title": f"Day {day_no}",
                "session_type": session_types[base_i],
                "is_rest_day": rest_days[base_i],
                "is_hard_day": hard_days[base_i],
                "duration_minutes": durations[base_i],
                "target_intensity": "easy" if not hard_days[base_i] else "threshold",
                "terrain": "road",
                "workout": "Session details",
                "purpose": "training purpose",
                "signal_ids": ["forecast.readiness.status"],
            }
        )

    return {
        "meta": {
            "today": "2026-03-01",
            "plan_start": "2026-03-01",
            "plan_days": plan_days,
            "style": "trailrunning",
            "primary_goal": "build trail endurance",
            "lifestyle_notes": "weekday road only",
        },
        "snapshot": {
            "last7": {
                "distance_km": "30",
                "moving_time_hours": "5.0",
                "elevation_m": "500",
                "activity_count": "5",
                "sleep_hours_mean": "7.5",
                "hrv_mean": "65",
                "rhr_mean": "45",
            },
            "baseline28": {
                "distance_km": "25",
                "moving_time_hours": "4.5",
                "elevation_m": "400",
                "activity_count": "4",
                "sleep_hours_mean": "7.8",
                "hrv_mean": "68",
                "rhr_mean": "44",
            },
            "notes": "snapshot notes",
        },
        "readiness": {
            "status": "steady",
            "rationale": "Baseline rationale.",
            "signal_ids": ["forecast.readiness.status"],
        },
        "plan": {
            "weekly_totals": {
                "planned_distance_km": 42.0,
                "planned_moving_time_hours": 5.0,
                "planned_elevation_m": 900.0,
            },
            "days": days,
        },
        "recovery": {
            "actions": ["Sleep 8h"],
            "signal_ids": ["forecast.readiness.status"],
        },
        "risks": [
            {
                "severity": "low",
                "message": "Monitor fatigue",
                "signal_ids": ["forecast.overreach_risk.level"],
            }
        ],
        "data_notes": [],
        "citations": [
            {
                "signal_id": "forecast.readiness.status",
                "source": "readiness_and_risk_forecast.json:result.readiness.status",
                "date_range": "2026-03-01..2026-03-01",
                "value": "steady",
            },
            {
                "signal_id": "forecast.overreach_risk.level",
                "source": "readiness_and_risk_forecast.json:result.overreach_risk.level",
                "date_range": "2026-03-01..2026-03-01",
                "value": "moderate",
            },
            {
                "signal_id": "forecast.recovery_capability.key",
                "source": "readiness_and_risk_forecast.json:result.inputs.recovery_capability_key",
                "date_range": "2026-03-01..2026-03-01",
                "value": "load_sleep_resting_hr_hrv",
            },
        ],
    }


def make_evaluation_report_artifact() -> dict[str, Any]:
    return {
        "score": 82,
        "grade": "B",
        "subscores": {"safety": 85, "structure": 80},
        "stats": {"days": 7, "hard_days": 1, "rest_days": 1},
        "violations": [
            {
                "code": "NOT_ENOUGH_REST",
                "severity": "medium",
                "category": "safety",
                "penalty": 15,
                "message": "Rolling 7-day window 0 has 0 rest days.",
                "details": {"window_index": 0},
            }
        ],
    }


def deep_copy_artifact(obj: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(obj)
