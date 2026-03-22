from __future__ import annotations

from typing import Any, Optional

from trailtraining.llm.constraints import (
    ConstraintConfig,
    EffectiveConstraintContext,
    _rest_text_conflicts,
    constraint_config_from_env,
    derive_effective_constraints,
)
from trailtraining.llm.shared import recompute_weekly_totals
from trailtraining.llm.windowing import extract_last7_hours, normalize_plan_days, rolling_windows

_HARD_TYPES = {"tempo", "intervals", "hills"}
_TITLE_BY_SESSION_TYPE = {
    "rest": "Rest day",
    "easy": "Easy run",
    "aerobic": "Aerobic run",
    "long": "Long run",
    "tempo": "Tempo session",
    "intervals": "Intervals session",
    "hills": "Hill session",
    "strength": "Strength session",
    "cross": "Cross-training",
}


def _get_cfg() -> ConstraintConfig:
    return constraint_config_from_env()


def _sum_minutes(days: list[dict[str, Any]]) -> int:
    total = 0
    for d in days:
        m = d.get("duration_minutes")
        if isinstance(m, (int, float)):
            total += int(round(float(m)))
    return total


def _canonicalize_rest_day(
    d: dict[str, Any],
    *,
    zero_duration: bool,
    rest_day_max_minutes: int,
) -> None:
    d["is_rest_day"] = True
    d["session_type"] = "rest"
    d["is_hard_day"] = False
    d["title"] = "Rest day"
    d["target_intensity"] = "rest"
    d["terrain"] = "n/a"
    d["workout"] = "Rest day. No structured training."

    current = d.get("duration_minutes")
    if zero_duration:
        d["duration_minutes"] = 0
    elif isinstance(current, (int, float)) and float(current) > rest_day_max_minutes:
        d["duration_minutes"] = int(rest_day_max_minutes)


def _set_weekly_hours(plan_obj: dict[str, Any]) -> float:
    recompute_weekly_totals(plan_obj)
    wt = (plan_obj.get("plan") or {}).get("weekly_totals") or {}
    hours = wt.get("planned_moving_time_hours")
    return float(hours) if isinstance(hours, (int, float)) else 0.0


def _normalize_note_text(note: str) -> str:
    return " ".join(str(note or "").split()).strip()


def _append_note_once(notes: list[str], note: str, *, replace_prefix: str | None = None) -> None:
    clean_note = _normalize_note_text(note)
    if not clean_note:
        return

    if replace_prefix:
        clean_prefix = _normalize_note_text(replace_prefix)
        notes[:] = [
            existing
            for existing in notes
            if not _normalize_note_text(existing).startswith(clean_prefix)
            and _normalize_note_text(existing) != clean_note
        ]
    elif any(_normalize_note_text(existing) == clean_note for existing in notes):
        return

    notes.append(clean_note)


def _dedupe_data_notes(plan_obj: dict[str, Any]) -> None:
    raw = plan_obj.get("data_notes")
    if not isinstance(raw, list):
        return

    deduped: list[str] = []
    seen: set[str] = set()
    for item in raw:
        clean = _normalize_note_text(str(item or ""))
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    plan_obj["data_notes"] = deduped


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
        return 45
    if st == "long":
        return 60
    return 30


def _canonical_non_rest_workout(day: dict[str, Any]) -> str:
    st = str(day.get("session_type") or "")
    mins = int(round(float(day.get("duration_minutes", 0) or 0)))
    terrain = str(day.get("terrain") or "").strip() or "mixed terrain"
    intensity = str(day.get("target_intensity") or "").strip() or st

    if st == "easy":
        return f"{mins} min easy run on {terrain}."
    if st == "aerobic":
        return f"{mins} min steady aerobic run on {terrain}."
    if st == "long":
        return f"{mins} min aerobic long run on {terrain}."
    if st == "tempo":
        return f"{mins} min tempo session on {terrain} at {intensity} effort."
    if st == "intervals":
        return f"{mins} min intervals session on {terrain} at {intensity} effort."
    if st == "hills":
        return f"{mins} min hills session on {terrain} at {intensity} effort."
    if st == "strength":
        return f"{mins} min strength session."
    if st == "cross":
        return f"{mins} min cross-training on {terrain} at {intensity} effort."
    return f"{mins} min session on {terrain}."


def _canonicalize_day(day: dict[str, Any]) -> None:
    st = str(day.get("session_type") or "").strip() or "easy"
    is_rest = bool(day.get("is_rest_day")) or st == "rest"

    if is_rest:
        day["is_rest_day"] = True
        day["session_type"] = "rest"
        day["is_hard_day"] = False
        day["duration_minutes"] = 0
        day["title"] = "Rest day"
        day["target_intensity"] = "rest"
        day["terrain"] = "n/a"
        day["workout"] = "Rest day. No structured training."
        purpose = str(day.get("purpose") or "").strip()
        if not purpose or "run" in purpose.lower() or "workout" in purpose.lower():
            day["purpose"] = "Absorb training and reduce fatigue."
        return

    day["is_rest_day"] = False
    day["session_type"] = st
    day["is_hard_day"] = st in _HARD_TYPES
    mins = day.get("duration_minutes")
    if not isinstance(mins, (int, float)) or float(mins) <= 0:
        day["duration_minutes"] = _min_minutes_for_day(day)
    day["title"] = _TITLE_BY_SESSION_TYPE.get(st, "Training session")
    day["target_intensity"] = str(day.get("target_intensity") or "").strip() or st
    day["terrain"] = str(day.get("terrain") or "").strip() or ("road" if st != "long" else "trail")
    day["workout"] = _canonical_non_rest_workout(day)


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
        d["duration_minutes"] = int(cur_i - cut)
        remaining -= cut
        _canonicalize_day(d)

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

    windows = rolling_windows(days, size=7)
    for _ in range(len(days)):
        violating: list[dict[str, Any]] | None = None
        hard: list[dict[str, Any]] = []

        for wk in windows:
            hard = [d for d in wk if bool(d.get("is_hard_day")) and not bool(d.get("is_rest_day"))]
            if len(hard) > max_hard:
                violating = wk
                break

        if violating is None:
            break

        cand = max(hard, key=_hard_downgrade_score)
        cand["is_hard_day"] = False
        if str(cand.get("session_type") or "") in _HARD_TYPES:
            cand["session_type"] = "aerobic"
        _canonicalize_day(cand)
        changed.append(str(cand.get("date") or ""))

    return changed


def _rest_convert_score(d: dict[str, Any]) -> int:
    st = str(d.get("session_type") or "")
    if st in ("easy", "aerobic"):
        return 1
    if st == "strength":
        return 2
    if st == "cross":
        return 3
    if st == "long":
        return 4
    return 5


def _enforce_min_rest_per_rolling7d(days: list[dict[str, Any]], min_rest: int) -> list[str]:
    changed: list[str] = []
    if min_rest <= 0:
        return changed

    windows = rolling_windows(days, size=7)

    for _ in range(len(days)):
        violating: list[dict[str, Any]] | None = None
        for wk in windows:
            if sum(1 for d in wk if bool(d.get("is_rest_day"))) < min_rest:
                violating = wk
                break
        if violating is None:
            break

        candidates = [d for d in violating if not bool(d.get("is_rest_day"))]
        if not candidates:
            break

        best = min(candidates, key=_rest_convert_score)
        best["is_rest_day"] = True
        best["session_type"] = "rest"
        best["is_hard_day"] = False
        best["duration_minutes"] = 0
        _canonicalize_day(best)
        changed.append(str(best.get("date") or ""))

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
                if str(cand.get("session_type") or "") in _HARD_TYPES:
                    cand["session_type"] = "aerobic"
                _canonicalize_day(cand)
                changed.append(str(cand.get("date") or ""))
                streak = [
                    x
                    for x in streak
                    if bool(x.get("is_hard_day")) and not bool(x.get("is_rest_day"))
                ]
        else:
            streak = []

    return changed


def build_eval_constraints_block(
    rollups: Optional[dict[str, Any]],
    effective: Optional[EffectiveConstraintContext] = None,
) -> str:
    cfg = _get_cfg()
    ctx = effective or derive_effective_constraints(
        det_forecast=None,
        rollups=rollups,
        cfg=cfg,
        lifestyle_notes="",
    )

    lines: list[str] = []
    if ctx.allowed_week1_hours is not None:
        lines.append(
            f"- MAX_RAMP_PCT: planned_moving_time_hours MUST be <= {ctx.allowed_week1_hours:.2f}h "
            f"(effective_max_ramp_pct={ctx.effective_max_ramp_pct:.1f}%)."
        )
    else:
        lines.append(
            f"- MAX_RAMP_PCT: effective_max_ramp_pct={ctx.effective_max_ramp_pct:.1f}% "
            "(rollups last7 hours unavailable; be conservative)."
        )

    lines.append(
        f"- TOO_MANY_CONSEC_HARD: NEVER exceed {ctx.effective_max_consecutive_hard} consecutive hard days."
    )
    lines.append(
        f"- TOO_MANY_HARD_PER_WEEK: hard days in any 7-day chunk MUST be <= {ctx.effective_max_hard_per_7d}."
    )
    lines.append(
        f"- NOT_ENOUGH_REST: rest days in any 7-day chunk MUST be >= {ctx.min_rest_per_7d}."
    )
    lines.append(
        "- Structured fields are authoritative: title/workout/intensity text must agree with session_type, is_rest_day, is_hard_day, and duration_minutes."
    )
    if ctx.reasons:
        lines.append(f"- These stricter limits apply because: {'; '.join(ctx.reasons)}.")
    return "\n".join(lines)


def apply_eval_coach_guardrails(
    plan_obj: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    *,
    effective: Optional[EffectiveConstraintContext] = None,
) -> None:
    if not isinstance(plan_obj, dict):
        return

    cfg = _get_cfg()
    ctx = effective or derive_effective_constraints(
        det_forecast=None,
        rollups=rollups,
        cfg=cfg,
        lifestyle_notes=str(((plan_obj.get("meta") or {}).get("lifestyle_notes")) or ""),
    )
    days = normalize_plan_days(plan_obj)
    strict_zero_rest = rollups is not None or ctx.allowed_week1_hours is not None

    wt = (plan_obj.get("plan") or {}).get("weekly_totals") or {}
    original_hours = _planned_hours_from_obj(plan_obj)
    distance_value = wt.get("planned_distance_km") if isinstance(wt, dict) else None
    original_distance = float(distance_value) if isinstance(distance_value, (int, float)) else None

    elevation_value = wt.get("planned_elevation_m") if isinstance(wt, dict) else None
    original_elevation = (
        float(elevation_value) if isinstance(elevation_value, (int, float)) else None
    )
    # Normalize all rest-day objects to canonical rest days.
    for d in days:
        if bool(d.get("is_rest_day")) or str(d.get("session_type") or "") == "rest":
            strict_for_day = strict_zero_rest or _rest_text_conflicts(d)
            _canonicalize_rest_day(
                d,
                zero_duration=strict_for_day,
                rest_day_max_minutes=cfg.rest_day_max_minutes,
            )

    changed_hard_week = _enforce_max_hard_per_7d(days, ctx.effective_max_hard_per_7d)
    changed_hard_consec = _enforce_max_consecutive_hard(days, ctx.effective_max_consecutive_hard)
    changed_rest = _enforce_min_rest_per_rolling7d(days, ctx.min_rest_per_7d)

    # Re-canonicalize any days converted to rest by rolling-window enforcement.
    for d in days:
        if bool(d.get("is_rest_day")) or str(d.get("session_type") or "") == "rest":
            strict_for_day = strict_zero_rest or _rest_text_conflicts(d)
            _canonicalize_rest_day(
                d,
                zero_duration=strict_for_day,
                rest_day_max_minutes=cfg.rest_day_max_minutes,
            )

    if ctx.allowed_week1_hours is not None and ctx.allowed_week1_hours > 0:
        wk = days[: min(7, len(days))]
        cur_min = _sum_minutes(wk)
        allowed_min = int(round(ctx.allowed_week1_hours * 60.0))
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

    new_hours = _set_weekly_hours(plan_obj)

    # If week 1 is now entirely rest / zero training, weekly distance and elevation
    # should be unknown/null rather than 0.0.
    first7 = days[: min(7, len(days))]
    has_any_non_rest_training = any(
        not bool(d.get("is_rest_day")) and str(d.get("session_type") or "") != "rest"
        for d in first7
    )

    if isinstance(wt, dict):
        if not has_any_non_rest_training or new_hours <= 0:
            wt["planned_distance_km"] = None
            wt["planned_elevation_m"] = None
        elif original_hours and original_hours > 0:
            ratio = max(0.0, min(1.0, new_hours / float(original_hours)))

            if original_distance is not None:
                wt["planned_distance_km"] = round(original_distance * ratio, 1)

            if original_elevation is not None:
                wt["planned_elevation_m"] = round(original_elevation * ratio, 1)

    last7 = extract_last7_hours(rollups)

    notes = plan_obj.get("data_notes")
    if isinstance(notes, list):
        # Remove prior guardrail notes so we do not stack conflicting summaries.
        notes[:] = [
            note
            for note in notes
            if not (
                isinstance(note, str)
                and (
                    note.startswith("Guardrails: set is_hard_day=false")
                    or note.startswith("Guardrails: adjusted hard-day streak")
                    or note.startswith("Guardrails: converted ")
                    or note.startswith("Guardrails: enforced ramp rate")
                )
            )
        ]

        if changed_hard_week:
            notes.append(
                f"Guardrails: set is_hard_day=false on {changed_hard_week} to satisfy max_hard_per_7d={ctx.effective_max_hard_per_7d}."
            )
        if changed_hard_consec:
            notes.append(
                f"Guardrails: adjusted hard-day streak on {changed_hard_consec} to satisfy max_consecutive_hard={ctx.effective_max_consecutive_hard}."
            )
        if changed_rest:
            notes.append(
                f"Guardrails: converted {changed_rest} to rest days to satisfy min_rest_per_7d={ctx.min_rest_per_7d} (rolling window)."
            )

        if ctx.allowed_week1_hours is not None:
            if isinstance(last7, (int, float)) and float(last7) > 0:
                notes.append(
                    f"Guardrails: enforced ramp rate (max_ramp_pct={ctx.effective_max_ramp_pct:.1f}%). "
                    f"last7_hours={float(last7):.3f}, allowed_hours={ctx.allowed_week1_hours:.3f}, planned_hours={new_hours:.3f}."
                )
            else:
                notes.append(
                    f"Guardrails: enforced ramp rate (max_ramp_pct={ctx.effective_max_ramp_pct:.1f}%). "
                    f"allowed_hours={ctx.allowed_week1_hours:.3f}, planned_hours={new_hours:.3f}."
                )


def _planned_hours_from_obj(plan_obj: dict[str, Any]) -> Optional[float]:
    wt = (plan_obj.get("plan") or {}).get("weekly_totals") or {}
    v = wt.get("planned_moving_time_hours")
    return float(v) if isinstance(v, (int, float)) else None
