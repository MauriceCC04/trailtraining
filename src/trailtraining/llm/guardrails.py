# src/trailtraining/llm/guardrails.py
from __future__ import annotations

from typing import Any, Optional

from trailtraining.llm.constraints import ConstraintConfig, constraint_config_from_env


def _get_cfg() -> ConstraintConfig:
    return constraint_config_from_env()


def _get_last7_hours(rollups: Optional[dict[str, Any]]) -> Optional[float]:
    try:
        v = (
            (rollups or {})
            .get("windows", {})
            .get("7", {})
            .get("activities", {})
            .get("total_moving_time_hours")
        )
        return float(v) if isinstance(v, (int, float)) else None
    except Exception:
        return None


def _normalize_days(plan_obj: dict[str, Any]) -> list[dict[str, Any]]:
    days = (plan_obj.get("plan") or {}).get("days")
    if not isinstance(days, list):
        return []
    out = [d for d in days if isinstance(d, dict)]
    out.sort(key=lambda d: str(d.get("date") or "9999-99-99"))
    return out


def _sum_minutes(days: list[dict[str, Any]]) -> int:
    total = 0
    for d in days:
        m = d.get("duration_minutes")
        if isinstance(m, (int, float)):
            total += int(round(float(m)))
    return total


def _set_weekly_hours(plan_obj: dict[str, Any]) -> float:
    days = _normalize_days(plan_obj)
    first7 = days[: min(7, len(days))]
    total_min = _sum_minutes(first7)
    hours = total_min / 60.0

    wt = (plan_obj.get("plan") or {}).get("weekly_totals") or {}
    if isinstance(wt, dict):
        wt["planned_moving_time_hours"] = round(hours, 1)

    return hours


def _min_minutes_for_day(d: dict[str, Any]) -> int:
    if bool(d.get("is_rest_day")) or str(d.get("session_type") or "") == "rest":
        return 0

    st = str(d.get("session_type") or "")
    if st == "strength":
        return 20
    if st in ("easy", "aerobic"):
        return 30
    if st == "tempo":
        return 40
    if st in ("intervals", "hills"):
        return 40
    if st == "cross":
        return 50
    if st == "long":
        return 60
    return 30


def _reduction_priority(d: dict[str, Any]) -> tuple[int, int]:
    st = str(d.get("session_type") or "")
    dur = d.get("duration_minutes")
    dur_i = int(round(float(dur))) if isinstance(dur, (int, float)) else 0

    if bool(d.get("is_rest_day")) or st == "rest":
        return (0, -dur_i)

    if not bool(d.get("is_hard_day")):
        band = 1
        if st in ("easy", "aerobic", "strength"):
            band = 1
        elif st in ("cross", "long"):
            band = 2
        else:
            band = 3
        return (band, -dur_i)

    if st in ("cross", "long"):
        return (4, -dur_i)
    return (5, -dur_i)


def _reduce_total_minutes(days: list[dict[str, Any]], reduce_by: int) -> int:
    if reduce_by <= 0:
        return 0

    cands = sorted(days, key=_reduction_priority)

    remaining = int(reduce_by)
    for d in cands:
        if remaining <= 0:
            break
        cur = d.get("duration_minutes")
        if not isinstance(cur, (int, float)):
            continue
        cur_i = int(round(float(cur)))
        min_i = _min_minutes_for_day(d)
        cap = cur_i - min_i
        if cap <= 0:
            continue
        cut = cap if cap < remaining else remaining
        new_i = cur_i - cut
        d["duration_minutes"] = int(new_i)
        remaining -= cut

    return remaining


def _hard_downgrade_score(d: dict[str, Any]) -> int:
    st = str(d.get("session_type") or "")
    score = 0

    if st in ("long", "cross"):
        score += 20
    if st in ("easy", "aerobic", "strength"):
        score += 10

    txt = " ".join(
        [
            str(d.get("target_intensity") or ""),
            str(d.get("workout") or ""),
            str(d.get("title") or ""),
        ]
    ).lower()

    if "easy" in txt or "aerobic" in txt or "conversational" in txt:
        score += 8
    if "tempo" in txt or "interval" in txt or "hard" in txt or "threshold" in txt:
        score -= 15

    dur = d.get("duration_minutes")
    if isinstance(dur, (int, float)) and float(dur) >= 90:
        score += 3

    return score


def _enforce_max_hard_per_7d(days: list[dict[str, Any]], max_hard: int) -> list[str]:
    changed: list[str] = []
    if max_hard <= 0:
        return changed

    wk = days[: min(7, len(days))]
    hard = [d for d in wk if bool(d.get("is_hard_day")) and not bool(d.get("is_rest_day"))]

    while len(hard) > max_hard:
        cand = max(hard, key=_hard_downgrade_score)
        cand["is_hard_day"] = False
        changed.append(str(cand.get("date") or ""))
        hard = [d for d in wk if bool(d.get("is_hard_day")) and not bool(d.get("is_rest_day"))]

    return changed


def _enforce_max_consecutive_hard(days: list[dict[str, Any]], max_consec: int) -> list[str]:
    changed: list[str] = []
    if max_consec <= 0:
        return changed

    streak: list[dict[str, Any]] = []
    for d in days:
        if bool(d.get("is_hard_day")) and not bool(d.get("is_rest_day")):
            streak.append(d)
            if len(streak) > max_consec:
                cand = max(streak, key=_hard_downgrade_score)
                cand["is_hard_day"] = False
                changed.append(str(cand.get("date") or ""))
                streak = [
                    x
                    for x in streak
                    if bool(x.get("is_hard_day")) and not bool(x.get("is_rest_day"))
                ]
        else:
            streak = []

    return changed


def build_eval_constraints_block(rollups: Optional[dict[str, Any]]) -> str:
    cfg = _get_cfg()
    last7 = _get_last7_hours(rollups)
    allowed = (
        (last7 * (1.0 + cfg.max_ramp_pct / 100.0)) if isinstance(last7, (int, float)) else None
    )

    lines = []
    if allowed is not None:
        lines.append(
            f"- MAX_RAMP_PCT: planned_moving_time_hours MUST be <= {allowed:.2f}h "
            f"(last7={last7:.2f}h, max_ramp_pct={cfg.max_ramp_pct:.1f}%)."
        )
    else:
        lines.append(
            f"- MAX_RAMP_PCT: max_ramp_pct={cfg.max_ramp_pct:.1f}% (rollups last7 hours unavailable; be conservative)."
        )

    lines.append(
        f"- TOO_MANY_CONSEC_HARD: NEVER exceed {cfg.max_consecutive_hard} consecutive hard days."
    )
    lines.append(
        f"- TOO_MANY_HARD_PER_WEEK: hard days in any 7-day chunk MUST be <= {cfg.max_hard_per_7d}."
    )
    lines.append(
        f"- NOT_ENOUGH_REST: rest days in any 7-day chunk MUST be >= {cfg.min_rest_per_7d}."
    )
    lines.append(
        "- Definition: set is_hard_day=true ONLY for high-intensity sessions (intervals/tempo/hills). Long EASY aerobic should usually be is_hard_day=false."
    )
    return "\n".join(lines)


def apply_eval_coach_guardrails(
    plan_obj: dict[str, Any], rollups: Optional[dict[str, Any]]
) -> None:
    if not isinstance(plan_obj, dict):
        return

    cfg = _get_cfg()
    days = _normalize_days(plan_obj)

    for d in days:
        if bool(d.get("is_rest_day")):
            d["session_type"] = "rest"
            m = d.get("duration_minutes")
            if isinstance(m, (int, float)) and float(m) > cfg.rest_day_max_minutes:
                d["duration_minutes"] = int(cfg.rest_day_max_minutes)

    changed_hard_week = _enforce_max_hard_per_7d(days, cfg.max_hard_per_7d)
    changed_hard_consec = _enforce_max_consecutive_hard(days, cfg.max_consecutive_hard)

    last7 = _get_last7_hours(rollups)
    if isinstance(last7, (int, float)) and last7 > 0:
        allowed_hours = float(last7) * (1.0 + cfg.max_ramp_pct / 100.0)

        wk = days[: min(7, len(days))]
        cur_min = _sum_minutes(wk)
        allowed_min = int(round(allowed_hours * 60.0))
        excess = cur_min - allowed_min

        if excess > 0:
            leftover = _reduce_total_minutes(wk, excess)

            if leftover > 0:
                non_rest = [
                    d
                    for d in wk
                    if not bool(d.get("is_rest_day"))
                    and isinstance(d.get("duration_minutes"), (int, float))
                ]
                if non_rest:
                    longest = max(non_rest, key=lambda d: float(d.get("duration_minutes", 0)))
                    min_i = _min_minutes_for_day(longest)
                    longest["duration_minutes"] = int(min_i)
                    leftover = 0

            _set_weekly_hours(plan_obj)

            wt = (plan_obj.get("plan") or {}).get("weekly_totals") or {}
            if isinstance(wt, dict):
                old_hours = cur_min / 60.0 if cur_min > 0 else None
                new_hours = _planned_hours_from_obj(plan_obj)
                if old_hours and new_hours is not None and old_hours > 0:
                    ratio = max(0.0, min(1.0, new_hours / old_hours))
                    for k in ("planned_distance_km", "planned_elevation_m"):
                        v = wt.get(k)
                        if isinstance(v, (int, float)):
                            wt[k] = round(float(v) * ratio, 1)

    notes = plan_obj.get("data_notes")
    if isinstance(notes, list):
        if changed_hard_week:
            notes.append(
                f"Guardrails: set is_hard_day=false on {changed_hard_week} to satisfy max_hard_per_7d={cfg.max_hard_per_7d}."
            )
        if changed_hard_consec:
            notes.append(
                f"Guardrails: adjusted hard-day streak on {changed_hard_consec} to satisfy max_consecutive_hard={cfg.max_consecutive_hard}."
            )

        if isinstance(last7, (int, float)) and last7 > 0:
            allowed = float(last7) * (1.0 + cfg.max_ramp_pct / 100.0)
            planned = _planned_hours_from_obj(plan_obj)
            if planned is not None:
                notes.append(
                    f"Guardrails: enforced ramp rate (max_ramp_pct={cfg.max_ramp_pct:.1f}%). "
                    f"last7_hours={float(last7):.3f}, allowed_hours={allowed:.3f}, planned_hours={planned:.3f}."
                )


def _planned_hours_from_obj(plan_obj: dict[str, Any]) -> Optional[float]:
    wt = (plan_obj.get("plan") or {}).get("weekly_totals") or {}
    v = wt.get("planned_moving_time_hours")
    return float(v) if isinstance(v, (int, float)) else None
