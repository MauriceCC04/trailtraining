# src/trailtraining/llm/constraints.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional


@dataclass(frozen=True)
class ConstraintConfig:
    # existing
    max_ramp_pct: float = 10.0
    max_consecutive_hard: int = 2

    # existing quality knobs
    max_hard_per_7d: int = 3
    min_rest_per_7d: int = 1
    min_signal_ids_per_day: int = 1

    # Compare weekly_totals.planned_moving_time_hours to sum(day.duration_minutes)/60
    weekly_time_tolerance_pct: float = 30.0  # allow mismatch (plans often round)

    # Rest-day expectations
    rest_day_max_minutes: int = 30
    require_rest_session_type: bool = True

    # --- new forecast-aware conservative overlays ---
    fatigued_max_hard_per_7d: int = 1
    high_risk_max_hard_per_7d: int = 1
    high_risk_max_consecutive_hard: int = 1
    high_risk_max_ramp_pct: float = 0.0

    # When telemetry is sparse, be somewhat more conservative
    sparse_data_max_hard_per_7d: int = 2
    sparse_data_max_ramp_pct: float = 5.0


def _pct_increase(new: float, old: Optional[float]) -> Optional[float]:
    if old is None or old <= 0:
        return None
    return (new - old) / old * 100.0


def _as_date(s: Any) -> Optional[date]:
    if not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _default_penalty(severity: str) -> int:
    return {"low": 3, "medium": 10, "high": 30}.get(severity, 10)


def _v(
    code: str,
    severity: str,
    category: str,
    message: str,
    *,
    penalty: Optional[int] = None,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "category": category,
        "penalty": int(_default_penalty(severity) if penalty is None else penalty),
        "message": message,
        "details": details or {},
    }


def _rolling7(days: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not days:
        return []
    if len(days) <= 7:
        return [days]
    return [days[i : i + 7] for i in range(0, len(days) - 7 + 1)]


def _as_clean_str(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    return s or None


def _citation_value(plan_obj: dict[str, Any], signal_id: str) -> Any:
    cits = plan_obj.get("citations")
    if not isinstance(cits, list):
        return None
    for c in cits:
        if not isinstance(c, dict):
            continue
        if c.get("signal_id") != signal_id:
            continue
        if "value" in c:
            return c.get("value")
        if isinstance(c.get("quote"), str):
            return c.get("quote")
        if isinstance(c.get("text"), str):
            return c.get("text")
    return None


def _extract_forecast_context(plan_obj: dict[str, Any]) -> dict[str, Optional[str]]:
    readiness = _as_clean_str(_citation_value(plan_obj, "forecast.readiness.status"))
    overreach = _as_clean_str(_citation_value(plan_obj, "forecast.overreach_risk.level"))
    capability_key = _as_clean_str(_citation_value(plan_obj, "forecast.recovery_capability.key"))
    capability_label = _as_clean_str(
        _citation_value(plan_obj, "forecast.recovery_capability.label")
    )

    return {
        "readiness_status": readiness.lower() if readiness else None,
        "overreach_risk_level": overreach.lower() if overreach else None,
        "recovery_capability_key": capability_key.lower() if capability_key else None,
        "recovery_capability_label": capability_label,
    }


def _is_sparse_capability(ctx: dict[str, Optional[str]]) -> bool:
    key = ctx.get("recovery_capability_key")
    label = (ctx.get("recovery_capability_label") or "").lower()

    if key in {"load_only", "load_sleep", "load_resting_hr", "load_hrv"}:
        return True

    # fallback for older artifacts that only carry the label
    if "only have training data" in label:
        return True

    return label.startswith("i have load + ") and " only" in label


def _forecast_reason_text(ctx: dict[str, Optional[str]]) -> str:
    reasons: list[str] = []

    if ctx.get("readiness_status") == "fatigued":
        reasons.append("readiness is fatigued")
    if ctx.get("overreach_risk_level") == "high":
        reasons.append("overreach risk is high")
    if _is_sparse_capability(ctx):
        reasons.append("recovery telemetry is sparse")

    if not reasons:
        return "current forecast context"

    return "; ".join(reasons)


def _normalize_days(plan_obj: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (plan_obj.get("plan") or {}).get("days")
    if not isinstance(raw, list):
        return []
    days: list[dict[str, Any]] = [d for d in raw if isinstance(d, dict)]

    # Sort by date if possible; otherwise keep stable-ish order
    def key(d: dict[str, Any]) -> tuple[int, str]:
        dd = _as_date(d.get("date"))
        return (0, dd.isoformat()) if dd else (1, str(d.get("date") or ""))

    return sorted(days, key=key)


def _planned_week_hours(plan_obj: dict[str, Any]) -> Optional[float]:
    wt = (plan_obj.get("plan") or {}).get("weekly_totals") or {}
    v = wt.get("planned_moving_time_hours")
    return float(v) if isinstance(v, (int, float)) else None


def _sum_hours(days: list[dict[str, Any]]) -> float:
    total_min = 0.0
    for d in days:
        m = d.get("duration_minutes")
        if isinstance(m, (int, float)):
            total_min += float(m)
    return total_min / 60.0


def _pct_diff(a: float, b: float) -> Optional[float]:
    if b <= 0:
        return None
    return abs(a - b) / b * 100.0


# -----------------------------
# Existing constraint function
# -----------------------------
def validate_training_plan(
    plan_obj: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    cfg: ConstraintConfig,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    days = _normalize_days(plan_obj)

    # --- Ramp rate: use actual first-7 planned durations, not weekly_totals ---
    declared_planned_hours = _planned_week_hours(plan_obj)
    actual_first7_hours = _sum_hours(days[: min(7, len(days))]) if days else None

    last7_hours = None
    try:
        w7 = (rollups or {}).get("windows", {}).get("7", {})
        last7_hours = w7.get("activities", {}).get("total_moving_time_hours")
    except Exception:
        last7_hours = None

    ramp_basis_hours = (
        actual_first7_hours if actual_first7_hours is not None else declared_planned_hours
    )

    if isinstance(ramp_basis_hours, (int, float)) and isinstance(last7_hours, (int, float)):
        inc = _pct_increase(float(ramp_basis_hours), float(last7_hours))
        if inc is not None and inc > cfg.max_ramp_pct:
            violations.append(
                _v(
                    "MAX_RAMP_PCT",
                    "high",
                    "safety",
                    f"Planned first-7-day moving time ramps {inc:.1f}% vs last 7 days "
                    f"(max {cfg.max_ramp_pct:.1f}%).",
                    details={
                        "ramp_basis_hours": ramp_basis_hours,
                        "declared_planned_hours": declared_planned_hours,
                        "last7_hours": last7_hours,
                        "ramp_pct": inc,
                    },
                )
            )

    # --- Too many hard days in a row ---
    consec = 0
    for d in days:
        hard = bool(d.get("is_hard_day"))
        if hard:
            consec += 1
            if consec > cfg.max_consecutive_hard:
                violations.append(
                    _v(
                        "TOO_MANY_CONSEC_HARD",
                        "high",
                        "safety",
                        f"More than {cfg.max_consecutive_hard} hard days in a row (hit {consec}).",
                        details={"date": d.get("date")},
                    )
                )
        else:
            consec = 0

    return violations


# -----------------------------
# New: quality scoring
# -----------------------------
def evaluate_training_plan_quality(
    plan_obj: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    cfg: ConstraintConfig,
) -> dict[str, Any]:
    """
    Returns a report dict:
      {
        "score": int,
        "grade": str,
        "subscores": {category: int},
        "stats": {...},
        "violations": [ ... ]
      }
    """
    # Start with existing safety constraints
    violations: list[dict[str, Any]] = []
    for v0 in validate_training_plan(plan_obj, rollups, cfg):
        if isinstance(v0, dict):
            # already normalized via _v(), but keep robust
            v0.setdefault("category", "safety")
            v0.setdefault("penalty", _default_penalty(str(v0.get("severity", "medium"))))
            violations.append(v0)

    days = _normalize_days(plan_obj)

    # ---- Stats ----
    hard_days = sum(1 for d in days if bool(d.get("is_hard_day")))
    rest_days = sum(1 for d in days if bool(d.get("is_rest_day")))
    stats: dict[str, Any] = {"days": len(days), "hard_days": hard_days, "rest_days": rest_days}
    # ---- Forecast-aware context ----
    fx = _extract_forecast_context(plan_obj)
    for k, v in fx.items():
        if v is not None:
            stats[k] = v

    effective_max_hard_per_7d = cfg.max_hard_per_7d
    effective_max_consecutive_hard = cfg.max_consecutive_hard
    effective_max_ramp_pct = cfg.max_ramp_pct

    if fx.get("readiness_status") == "fatigued":
        effective_max_hard_per_7d = min(effective_max_hard_per_7d, cfg.fatigued_max_hard_per_7d)

    if fx.get("overreach_risk_level") == "high":
        effective_max_hard_per_7d = min(effective_max_hard_per_7d, cfg.high_risk_max_hard_per_7d)
        effective_max_consecutive_hard = min(
            effective_max_consecutive_hard, cfg.high_risk_max_consecutive_hard
        )
        effective_max_ramp_pct = min(effective_max_ramp_pct, cfg.high_risk_max_ramp_pct)

    if _is_sparse_capability(fx):
        effective_max_hard_per_7d = min(effective_max_hard_per_7d, cfg.sparse_data_max_hard_per_7d)
        effective_max_ramp_pct = min(effective_max_ramp_pct, cfg.sparse_data_max_ramp_pct)

    stats["effective_max_hard_per_7d"] = effective_max_hard_per_7d
    stats["effective_max_consecutive_hard"] = effective_max_consecutive_hard
    stats["effective_max_ramp_pct"] = effective_max_ramp_pct
    # ---- Structure checks ----
    seen = set()
    prev: Optional[date] = None
    for d in days:
        ds = d.get("date")
        dd = _as_date(ds)
        if not dd:
            violations.append(
                _v(
                    "BAD_DATE",
                    "low",
                    "structure",
                    "Day has invalid/missing date.",
                    details={"date": ds},
                )
            )
            continue

        if dd in seen:
            violations.append(
                _v(
                    "DUPLICATE_DATE",
                    "high",
                    "structure",
                    "Duplicate date in plan.days.",
                    details={"date": ds},
                )
            )
        seen.add(dd)

        if prev and (dd - prev).days != 1:
            violations.append(
                _v(
                    "NON_CONSECUTIVE_DATES",
                    "medium",
                    "structure",
                    "Plan dates are not consecutive (gap or reorder).",
                    details={"prev": prev.isoformat(), "curr": dd.isoformat()},
                )
            )
        prev = dd

    planned_hours = _planned_week_hours(plan_obj)
    if planned_hours is not None and days:
        first7 = days[: min(7, len(days))]
        sum_hours = _sum_hours(first7)
        diff = _pct_diff(planned_hours, sum_hours)
        if diff is not None and diff > cfg.weekly_time_tolerance_pct:
            violations.append(
                _v(
                    "WEEKLY_TOTALS_MISMATCH",
                    "low",
                    "structure",
                    f"weekly_totals.planned_moving_time_hours ({planned_hours:.1f}h) doesn't match "
                    f"sum(duration) ({sum_hours:.1f}h) within {cfg.weekly_time_tolerance_pct:.0f}%.",
                    details={
                        "planned_hours": planned_hours,
                        "sum_hours_first7": sum_hours,
                        "pct_diff": diff,
                    },
                )
            )
    # ---- Forecast-aware ramp overlay ----
    last7_hours = None
    try:
        w7 = (rollups or {}).get("windows", {}).get("7", {})
        last7_hours = w7.get("activities", {}).get("total_moving_time_hours")
    except Exception:
        last7_hours = None

    actual_first7_hours = _sum_hours(days[: min(7, len(days))]) if days else None
    ramp_basis_hours = actual_first7_hours if actual_first7_hours is not None else planned_hours

    if isinstance(ramp_basis_hours, (int, float)) and isinstance(last7_hours, (int, float)):
        inc = _pct_increase(float(ramp_basis_hours), float(last7_hours))
        if (
            inc is not None
            and inc > effective_max_ramp_pct
            and effective_max_ramp_pct < cfg.max_ramp_pct
        ):
            reason_text = _forecast_reason_text(fx)
            sev = "high" if fx.get("overreach_risk_level") == "high" else "medium"
            violations.append(
                _v(
                    "FORECAST_RAMP_TOO_AGGRESSIVE",
                    sev,
                    "safety",
                    f"Planned first-7-day moving time ramps {inc:.1f}% vs last 7 days, "
                    f"but limit is {effective_max_ramp_pct:.1f}% because {reason_text}.",
                    details={
                        "ramp_basis_hours": ramp_basis_hours,
                        "last7_hours": last7_hours,
                        "ramp_pct": inc,
                        "effective_max_ramp_pct": effective_max_ramp_pct,
                        "forecast_context": fx,
                    },
                    penalty=35 if sev == "high" else 15,
                )
            )
    # ---- Safety/consistency checks (new) ----
    # ---- Safety/consistency checks (rolling 7-day windows) ----
    for i, wk in enumerate(_rolling7(days)):
        h = sum(1 for d in wk if bool(d.get("is_hard_day")))
        if h > cfg.max_hard_per_7d:
            violations.append(
                _v(
                    "TOO_MANY_HARD_PER_7D",
                    "high",
                    "safety",
                    f"Rolling 7-day window {i} has {h} hard days (max {cfg.max_hard_per_7d}).",
                    details={
                        "window_index": i,
                        "window_start": wk[0].get("date"),
                        "window_end": wk[-1].get("date"),
                        "hard_days": h,
                    },
                )
            )

    for i, wk in enumerate(_rolling7(days)):
        r = sum(1 for d in wk if bool(d.get("is_rest_day")))
        if r < cfg.min_rest_per_7d:
            sev = "high" if r == 0 else "medium"
            violations.append(
                _v(
                    "NOT_ENOUGH_REST",
                    sev,
                    "safety",
                    f"Rolling 7-day window {i} has {r} rest days (min {cfg.min_rest_per_7d}).",
                    details={
                        "window_index": i,
                        "window_start": wk[0].get("date"),
                        "window_end": wk[-1].get("date"),
                        "rest_days": r,
                    },
                    penalty=35 if sev == "high" else 15,
                )
            )

    # ---- Forecast-aware hard-day overlay ----
    if effective_max_hard_per_7d < cfg.max_hard_per_7d:
        reason_text = _forecast_reason_text(fx)
        sev = "high" if fx.get("overreach_risk_level") == "high" else "medium"

        for i, wk in enumerate(_rolling7(days)):
            h = sum(1 for d in wk if bool(d.get("is_hard_day")))
            if h > effective_max_hard_per_7d:
                violations.append(
                    _v(
                        "FORECAST_HARD_DAY_LIMIT",
                        sev,
                        "safety",
                        f"Rolling 7-day window {i} has {h} hard days, but limit is "
                        f"{effective_max_hard_per_7d} because {reason_text}.",
                        details={
                            "window_index": i,
                            "window_start": wk[0].get("date"),
                            "window_end": wk[-1].get("date"),
                            "hard_days": h,
                            "effective_max_hard_per_7d": effective_max_hard_per_7d,
                            "forecast_context": fx,
                        },
                        penalty=35 if sev == "high" else 15,
                    )
                )

    # ---- Forecast-aware consecutive-hard overlay ----
    if effective_max_consecutive_hard < cfg.max_consecutive_hard:
        reason_text = _forecast_reason_text(fx)
        consec = 0
        streak_start: Optional[str] = None

        for d in days:
            if bool(d.get("is_hard_day")):
                consec += 1
                if consec == 1:
                    streak_start = d.get("date")
                if consec > effective_max_consecutive_hard:
                    violations.append(
                        _v(
                            "FORECAST_CONSEC_HARD_LIMIT",
                            "high",
                            "safety",
                            f"More than {effective_max_consecutive_hard} hard days in a row "
                            f"because {reason_text}.",
                            details={
                                "streak_start": streak_start,
                                "date": d.get("date"),
                                "consecutive_hard_days": consec,
                                "effective_max_consecutive_hard": effective_max_consecutive_hard,
                                "forecast_context": fx,
                            },
                            penalty=35,
                        )
                    )
            else:
                consec = 0
                streak_start = None
    for d in days:
        if not bool(d.get("is_rest_day")):
            continue
        mins = d.get("duration_minutes")
        if isinstance(mins, (int, float)) and float(mins) > float(cfg.rest_day_max_minutes):
            violations.append(
                _v(
                    "REST_DAY_TOO_LONG",
                    "medium",
                    "structure",
                    f"Rest day exceeds {cfg.rest_day_max_minutes} minutes.",
                    details={"date": d.get("date"), "duration_minutes": mins},
                )
            )
        if cfg.require_rest_session_type:
            st = d.get("session_type")
            if isinstance(st, str) and st != "rest":
                violations.append(
                    _v(
                        "REST_DAY_BAD_SESSION_TYPE",
                        "low",
                        "structure",
                        "Rest day should have session_type == 'rest'.",
                        details={"date": d.get("date"), "session_type": st},
                    )
                )

    # ---- Justification checks (new) ----
    for idx, d in enumerate(days):
        sig = d.get("signal_ids")
        n = len(sig) if isinstance(sig, list) else 0
        if n < cfg.min_signal_ids_per_day:
            violations.append(
                _v(
                    "MISSING_SIGNAL_IDS",
                    "medium",
                    "justification",
                    f"plan.days[{idx}] has empty/insufficient signal_ids (min {cfg.min_signal_ids_per_day}).",
                    details={"date": d.get("date")},
                )
            )

    cited = set()
    cits = plan_obj.get("citations")
    if isinstance(cits, list):
        for c in cits:
            if isinstance(c, dict) and isinstance(c.get("signal_id"), str):
                cited.add(c["signal_id"])

    used = set()
    for d in days:
        sig = d.get("signal_ids")
        if isinstance(sig, list):
            for s in sig:
                if isinstance(s, str):
                    used.add(s)

    if used and not cited:
        violations.append(
            _v(
                "MISSING_CITATIONS",
                "medium",
                "justification",
                "Plan uses signal_ids but citations[] is empty/missing.",
                details={"used_signal_ids_count": len(used)},
            )
        )
    elif used and cited:
        missing = sorted(list(used - cited))
        if missing:
            violations.append(
                _v(
                    "UNCITED_SIGNAL_IDS",
                    "medium",
                    "justification",
                    "Some signal_ids used in plan.days are not present in citations[].signal_id.",
                    details={"missing_signal_ids": missing[:50], "missing_count": len(missing)},
                )
            )

    return score_from_violations(violations, stats=stats)


def score_from_violations(
    violations: list[dict[str, Any]],
    *,
    stats: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    total_pen = 0
    by_cat: dict[str, int] = {}
    for v in violations:
        if not isinstance(v, dict):
            continue
        try:
            pen = int(v.get("penalty", _default_penalty(str(v.get("severity", "medium")))))
        except Exception:
            pen = 10
        total_pen += pen
        cat = str(v.get("category") or "other")
        by_cat[cat] = by_cat.get(cat, 0) + pen

    score = max(0, 100 - total_pen)

    subscores = {cat: max(0, 100 - pen) for cat, pen in by_cat.items()}

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": score,
        "grade": grade,
        "subscores": subscores,
        "stats": stats or {},
        "violations": violations,
    }
