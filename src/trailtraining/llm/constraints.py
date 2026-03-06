# src/trailtraining/llm/constraints.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class ConstraintConfig:
    max_ramp_pct: float = 10.0
    max_consecutive_hard: int = 2

def _pct_increase(new: float, old: float) -> Optional[float]:
    if old is None or old <= 0:
        return None
    return (new - old) / old * 100.0

def validate_training_plan(plan_obj: Dict[str, Any], rollups: Optional[Dict[str, Any]], cfg: ConstraintConfig) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []

    # --- Ramp rate ---
    planned = plan_obj.get("plan", {}).get("weekly_totals", {})
    planned_hours = planned.get("planned_moving_time_hours")
    last7_hours = None
    try:
        w7 = (rollups or {}).get("windows", {}).get("7", {})
        last7_hours = w7.get("activities", {}).get("total_moving_time_hours")
    except Exception:
        last7_hours = None

    if isinstance(planned_hours, (int, float)) and isinstance(last7_hours, (int, float)):
        inc = _pct_increase(float(planned_hours), float(last7_hours))
        if inc is not None and inc > cfg.max_ramp_pct:
            violations.append({
                "code": "MAX_RAMP_PCT",
                "severity": "high",
                "message": f"Planned moving time ramps {inc:.1f}% vs last 7 days (max {cfg.max_ramp_pct:.1f}%).",
                "details": {"planned_hours": planned_hours, "last7_hours": last7_hours, "ramp_pct": inc},
            })

    # --- 3 hard days in a row ---
    days = plan_obj.get("plan", {}).get("days", [])
    consec = 0
    for d in days:
        hard = bool(d.get("is_hard_day"))
        if hard:
            consec += 1
            if consec > cfg.max_consecutive_hard:
                violations.append({
                    "code": "TOO_MANY_CONSEC_HARD",
                    "severity": "high",
                    "message": f"More than {cfg.max_consecutive_hard} hard days in a row (hit {consec}).",
                    "details": {"date": d.get("date")},
                })
        else:
            consec = 0

    return violations