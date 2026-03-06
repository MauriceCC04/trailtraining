# src/trailtraining/llm/constraints.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ConstraintConfig:
    max_ramp_pct: float = 10.0
    max_consecutive_hard: int = 2  # i.e., "no 3 hard days in a row"


def _pct_increase(new: float, old: float) -> Optional[float]:
    if old <= 0:
        return None
    return (new - old) / old * 100.0


def validate_training_plan(
    plan_obj: Dict[str, Any],
    rollups: Optional[Dict[str, Any]],
    cfg: ConstraintConfig,
) -> List[Dict[str, Any]]:
    """
    Returns a list of violations:
      {code, severity, message, details}
    """
    violations: List[Dict[str, Any]] = []

    plan = plan_obj.get("plan")
    if not isinstance(plan, dict):
        return [
            {"code": "MISSING_PLAN", "severity": "high", "message": "Missing plan object in coach output.", "details": {}}
        ]

    weekly = plan.get("weekly_totals")
    if not isinstance(weekly, dict):
        violations.append(
            {
                "code": "MISSING_WEEKLY_TOTALS",
                "severity": "high",
                "message": "plan.weekly_totals missing; cannot evaluate ramp.",
                "details": {},
            }
        )
        weekly = {}

    planned_hours = weekly.get("planned_moving_time_hours")
    last7_hours = None
    try:
        if isinstance(rollups, dict):
            w7 = rollups.get("windows", {}).get("7")
            if isinstance(w7, dict):
                acts = w7.get("activities") if isinstance(w7.get("activities"), dict) else {}
                last7_hours = acts.get("total_moving_time_hours")
    except Exception:
        last7_hours = None

    # Ramp check (planned moving time vs last7 moving time)
    if isinstance(planned_hours, (int, float)) and isinstance(last7_hours, (int, float)):
        inc = _pct_increase(float(planned_hours), float(last7_hours))
        if inc is not None and inc > float(cfg.max_ramp_pct):
            violations.append(
                {
                    "code": "MAX_RAMP_PCT",
                    "severity": "high",
                    "message": f"Planned moving time ramps {inc:.1f}% vs last 7 days (max {cfg.max_ramp_pct:.1f}%).",
                    "details": {"planned_hours": float(planned_hours), "last7_hours": float(last7_hours), "ramp_pct": float(inc)},
                }
            )
    else:
        violations.append(
            {
                "code": "RAMP_NOT_EVALUATED",
                "severity": "medium",
                "message": "Could not evaluate ramp: missing planned_hours or last7_hours.",
                "details": {"planned_hours": planned_hours, "last7_hours": last7_hours},
            }
        )

    # No 3 hard days in a row (max_consecutive_hard=2 by default)
    days = plan.get("days")
    if not isinstance(days, list):
        violations.append(
            {
                "code": "MISSING_DAYS",
                "severity": "high",
                "message": "plan.days missing; cannot evaluate consecutive hard days.",
                "details": {},
            }
        )
        return violations

    consec = 0
    for d in days:
        if not isinstance(d, dict):
            continue
        hard = bool(d.get("is_hard_day"))
        if hard:
            consec += 1
            if consec > int(cfg.max_consecutive_hard):
                violations.append(
                    {
                        "code": "TOO_MANY_CONSEC_HARD",
                        "severity": "high",
                        "message": f"More than {cfg.max_consecutive_hard} hard days in a row (hit {consec}).",
                        "details": {"date": d.get("date"), "consecutive_hard": consec},
                    }
                )
        else:
            consec = 0

    return violations

def build_retrieval_context(
    combined: List[Dict[str, Any]],
    rollups: Optional[Dict[str, Any]],
    *,
    retrieval_weeks: int,
) -> Dict[str, Any]:
    return {
        "weekly_history": build_weekly_history(combined, weeks=retrieval_weeks),
        "signal_registry": build_signal_registry(combined, rollups if isinstance(rollups, dict) else None),
    }