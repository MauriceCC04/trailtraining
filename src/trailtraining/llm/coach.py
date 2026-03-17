from __future__ import annotations

import calendar
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from trailtraining import config
from trailtraining.llm.guardrails import apply_eval_coach_guardrails, build_eval_constraints_block
from trailtraining.llm.presets import get_system_prompt, get_task_prompt
from trailtraining.llm.rubrics import default_primary_goal_for_style
from trailtraining.llm.schemas import (
    TRAINING_PLAN_SCHEMA,
    ensure_training_plan_shape,
    training_plan_output_contract_text,
)
from trailtraining.llm.signals import build_retrieval_context
from trailtraining.util.dates import _as_date
from trailtraining.util.errors import LLMUnsupportedParameterError
from trailtraining.util.llm_helpers import _classify_and_raise
from trailtraining.util.state import load_json, save_json
from trailtraining.util.text import _safe_json_snippet

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Race-goal parsing
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_MONTH_PATTERN = (
    r"january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)


def _parse_race_context(goal_text: str, today: Optional[date] = None) -> dict[str, Any]:
    """Extract race date from a goal string and return planning context.

    Supports:
    - ISO dates:              "2026-07-30"
    - Full dates:             "July 30 2026", "July 30, 2026"
    - Month + year:           "July 2026"  (approximate -> last day of month)
    - Month only:             "in April"  (next occurrence, approximate)

    Returns an empty dict if no recognisable date is found or the date has passed.
    """
    today = today or date.today()
    low = goal_text.strip().lower()
    race_date: Optional[date] = None
    is_approximate = False

    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", goal_text)
    if m:
        with contextlib.suppress(ValueError):
            race_date = date.fromisoformat(m.group(1))

    if not race_date:
        m = re.search(
            rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
            low,
        )
        if m:
            month = _MONTH_MAP.get(m.group(1))
            if month:
                with contextlib.suppress(ValueError):
                    race_date = date(int(m.group(3)), month, int(m.group(2)))

    if not race_date:
        m = re.search(rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b", low)
        if m:
            month = _MONTH_MAP.get(m.group(1))
            if month:
                with contextlib.suppress(ValueError):
                    year = int(m.group(2))
                    last_day = calendar.monthrange(year, month)[1]
                    race_date = date(year, month, last_day)
                    is_approximate = True

    if not race_date:
        m = re.search(rf"\b(?:in\s+)?({_MONTH_PATTERN})\b", low)
        if m:
            month = _MONTH_MAP.get(m.group(1))
            if month:
                with contextlib.suppress(ValueError):
                    year = today.year if month >= today.month else today.year + 1
                    last_day = calendar.monthrange(year, month)[1]
                    race_date = date(year, month, last_day)
                    is_approximate = True

    if not race_date:
        return {}

    days_to_race = (race_date - today).days
    if days_to_race < 0:
        return {}

    return {
        "race_date": race_date.isoformat(),
        "weeks_to_race": days_to_race // 7,
        "days_to_race": days_to_race,
        "is_approximate": is_approximate,
    }


def _race_context_section(primary_goal: str) -> list[str]:
    """Build prompt lines describing the race goal, or empty list if no date found."""
    ctx = _parse_race_context(primary_goal)
    if not ctx:
        return []
    approx = " (approximate -> use last day of the stated month)" if ctx["is_approximate"] else ""
    weeks = ctx["weeks_to_race"]
    if weeks <= 2:
        phase = "TAPER / RACE-READY -> reduce volume ~20-30 %, sharpen with short quality only."
    elif weeks <= 6:
        phase = "PEAK -> maintain quality, cap volume, practice race-specific efforts."
    elif weeks <= 12:
        phase = "BUILD -> progressive loading, race-specific workouts increasing in frequency."
    else:
        phase = "BASE / DEVELOPMENT -> aerobic foundation; race-specific intensity starts later."

    return [
        "## Race Goal Context (use to calibrate periodization)",
        f"Race date: {ctx['race_date']}{approx}",
        f"Days to race: {ctx['days_to_race']}  |  Weeks to race: {weeks}",
        f"Recommended training phase: {phase}",
        "Anchor the plan to this phase. If plan_days covers multiple phases, transition between them.",
        "",
    ]


# ---------------------------------------------------------------------------
# Type coercions
# ---------------------------------------------------------------------------


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _as_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _as_float(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------


def _make_openrouter_client() -> OpenAI:
    api_key = (
        os.getenv("OPENROUTER_API_KEY") or os.getenv("TRAILTRAINING_OPENROUTER_API_KEY") or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY "
            "(or TRAILTRAINING_OPENROUTER_API_KEY).\n"
            "Example:\n"
            "  export OPENROUTER_API_KEY='sk-or-v1-...'\n"
            "Then rerun: trailtraining coach --prompt training-plan"
        )
    headers: dict[str, str] = {}
    if site_url := (os.getenv("TRAILTRAINING_OPENROUTER_SITE_URL") or "").strip():
        headers["HTTP-Referer"] = site_url
    if app_name := (os.getenv("TRAILTRAINING_OPENROUTER_APP_NAME") or "trailtraining").strip():
        headers["X-OpenRouter-Title"] = app_name
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, default_headers=headers)


def _call_with_param_fallback(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    """Call the responses API, stripping unsupported params on failure.

    Tries four progressively-stripped variants so the same call site works
    across models that reject reasoning or verbosity parameters:

    1. Full kwargs
    2. Without ``text.verbosity`` only
    3. Without ``reasoning`` only
    4. Without either (bare minimum)
    """

    def _strip_verbosity(kw: dict[str, Any]) -> dict[str, Any]:
        text = {k: v for k, v in kw.get("text", {}).items() if k != "verbosity"}
        return {**kw, "text": text} if text else {k: v for k, v in kw.items() if k != "text"}

    def _strip_reasoning(kw: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in kw.items() if k != "reasoning"}

    attempts = [
        ("full", kwargs),
        ("no_text_verbosity", _strip_verbosity(kwargs)),
        ("no_reasoning", _strip_reasoning(kwargs)),
        ("bare_minimum", _strip_reasoning(_strip_verbosity(kwargs))),
    ]
    last_exc: Optional[Exception] = None
    for label, kw in attempts:
        try:
            return client.responses.create(**kw)
        except Exception as exc:
            try:
                _classify_and_raise(exc)
            except LLMUnsupportedParameterError as unsupported:
                log.warning(
                    "LLM call rejected %s attempt due to unsupported parameters: %s",
                    label,
                    unsupported,
                )
                last_exc = unsupported
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _call_with_schema(client: OpenAI, kwargs: dict[str, Any], schema: dict[str, Any]) -> Any:
    """Try structured JSON output, fall back to plain call if the model doesn't support it."""
    name, body = schema.get("name"), schema.get("schema")

    structured_attempts = [
        (
            "text.format",
            {
                **kwargs,
                "text": {
                    **kwargs.get("text", {}),
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "schema": body,
                        "strict": True,
                    },
                },
            },
        ),
        (
            "response_format",
            {
                **kwargs,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": name, "strict": True, "schema": body},
                },
            },
        ),
    ]

    for label, kw in structured_attempts:
        try:
            return _call_with_param_fallback(client, kw)
        except LLMUnsupportedParameterError as exc:
            log.warning("Structured output mode %s unavailable; falling back: %s", label, exc)

    return _call_with_param_fallback(client, kwargs)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def _resolve_input_paths(
    input_path: Optional[str],
    personal_path: Optional[str],
    summary_path: Optional[str],
) -> tuple[Path, Path, Optional[Path]]:
    base = (
        Path(input_path).expanduser().resolve()
        if input_path
        else Path(config.PROMPTING_DIRECTORY).expanduser().resolve()
    )
    personal = (
        Path(personal_path).expanduser().resolve()
        if personal_path
        else base / "formatted_personal_data.json"
    )
    summary = (
        Path(summary_path).expanduser().resolve()
        if summary_path
        else base / "combined_summary.json"
    )
    rollups = base / "combined_rollups.json"
    return personal, summary, (rollups if rollups.exists() else None)


# ---------------------------------------------------------------------------
# Combined-summary helpers
# ---------------------------------------------------------------------------


def _dedup_activities_in_place(combined: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for day in combined:
        acts = day.get("activities")
        if not isinstance(acts, list):
            day["activities"] = []
            continue
        unique = []
        for a in acts:
            if not isinstance(a, dict):
                continue
            aid = a.get("id")
            if aid is None:
                unique.append(a)
            elif (key := str(aid)) not in seen:
                seen.add(key)
                unique.append(a)
        day["activities"] = unique


def _filter_last_days(combined: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    if not combined:
        return combined
    last = _as_date(combined[-1].get("date", ""))
    if not last:
        return combined
    cutoff = last - timedelta(days=days - 1)
    return [d for d in combined if (dd := _as_date(d.get("date", ""))) and dd >= cutoff]


def _summarize_activity(a: dict[str, Any]) -> str:
    parts = [str(a.get("sport_type") or a.get("type") or "unknown")]
    if isinstance(dist := a.get("distance"), (int, float)):
        parts.append(f"{dist / 1000:.2f} km")
    if isinstance(elev := a.get("total_elevation_gain"), (int, float)):
        parts.append(f"{elev:.0f} m+")
    if isinstance(mv := a.get("moving_time"), (int, float)):
        parts.append(f"{mv / 60:.0f} min")
    if isinstance(hr := a.get("average_heartrate"), (int, float)):
        parts.append(f"avgHR {hr:.0f}")
    if isinstance(name := a.get("name"), str) and name.strip():
        parts.append(f"({name.strip()})")
    return " • " + " | ".join(parts)


def _summarize_day(day: dict[str, Any]) -> str:
    lines = [f"## {day.get('date', 'unknown-date')}"]
    sleep = day.get("sleep")
    if isinstance(sleep, dict):
        wanted = (
            "sleep_score",
            "score",
            "duration",
            "total_sleep",
            "resting_hr",
            "rhr",
            "readiness",
            "stress",
        )
        picked = {k: sleep[k] for k in wanted if k in sleep}
        lines.append(f"Sleep: {picked}" if picked else "Sleep: (data present)")
    else:
        lines.append("Sleep: (none)")
    acts = day.get("activities") or []
    if isinstance(acts, list) and acts:
        lines.append(f"Activities ({len(acts)}):")
        lines.extend(_summarize_activity(a) for a in acts[:50] if isinstance(a, dict))
    else:
        lines.append("Activities: (none)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Forecast helpers
# ---------------------------------------------------------------------------


def _load_or_compute_deterministic_forecast(
    base_dir: Path,
    combined: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    forecast_p = base_dir / "readiness_and_risk_forecast.json"
    if forecast_p.exists():
        obj = load_json(forecast_p, default=None)
        if isinstance(obj, dict):
            return obj
        log.warning("Ignoring malformed deterministic forecast artifact at %s", forecast_p)
        return None
    try:
        from trailtraining.forecast.forecast import compute_readiness_and_risk

        fr = compute_readiness_and_risk(combined)
        payload: dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "result": {
                "date": fr.date,
                "readiness": {"score": fr.readiness_score, "status": fr.readiness_status},
                "overreach_risk": {
                    "score": fr.overreach_risk_score,
                    "level": fr.overreach_risk_level,
                },
                "inputs": fr.inputs,
                "drivers": fr.drivers,
            },
        }
        try:
            save_json(forecast_p, payload, compact=False)
        except Exception as exc:
            log.warning("Failed to persist deterministic forecast to %s: %s", forecast_p, exc)
        return payload
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        log.warning("Deterministic forecast unavailable: %s", exc)
        return None


def _forecast_capability_block(det_forecast: dict[str, Any]) -> list[str]:
    inputs = _as_dict(_as_dict(_as_dict(det_forecast.get("result")).get("inputs")))
    label = _as_str(inputs.get("recovery_capability_label"))
    if not label:
        return []
    return [
        "## Available recovery telemetry (authoritative)",
        label,
        (
            f"Recent 7d usable days: sleep={inputs.get('sleep_days_7d')}, "
            f"resting_hr={inputs.get('resting_hr_days_7d')}, hrv={inputs.get('hrv_days_7d')}"
        ),
        "Do not assume unavailable recovery signals exist.",
        "",
    ]


def _forecast_signal_rows(det_forecast: dict[str, Any]) -> list[dict[str, Any]]:
    res = _as_dict(det_forecast.get("result"))
    d = res.get("date")
    dr = f"{d}..{d}" if isinstance(d, str) and d else ""
    inputs = _as_dict(res.get("inputs"))
    src = "readiness_and_risk_forecast.json:result"

    def _row(signal_id: str, value: Any, path: str, unit: str = "") -> dict[str, Any]:
        return {
            "signal_id": signal_id,
            "value": value,
            "unit": unit,
            "source": f"{src}.{path}",
            "date_range": dr,
        }

    rows: list[dict[str, Any]] = []
    if label := _as_str(inputs.get("recovery_capability_label")):
        rows.append(
            _row("forecast.recovery_capability.label", label, "inputs.recovery_capability_label")
        )
    if key := _as_str(inputs.get("recovery_capability_key")):
        rows.append(_row("forecast.recovery_capability.key", key, "inputs.recovery_capability_key"))

    readiness = _as_dict(res.get("readiness"))
    rows += [
        _row("forecast.readiness.status", readiness.get("status"), "readiness.status"),
        _row("forecast.readiness.score", readiness.get("score"), "readiness.score"),
    ]
    for key, sid in [
        ("sleep_days_7d", "forecast.recovery_capability.sleep_days_7d"),
        ("resting_hr_days_7d", "forecast.recovery_capability.resting_hr_days_7d"),
        ("hrv_days_7d", "forecast.recovery_capability.hrv_days_7d"),
    ]:
        rows.append(_row(sid, inputs.get(key), f"inputs.{key}", "days"))

    risk = _as_dict(res.get("overreach_risk"))
    rows += [
        _row("forecast.overreach_risk.level", risk.get("level"), "overreach_risk.level"),
        _row("forecast.overreach_risk.score", risk.get("score"), "overreach_risk.score"),
    ]
    return rows


# ---------------------------------------------------------------------------
# Plan mutation helpers
# ---------------------------------------------------------------------------


def _apply_deterministic_readiness(
    plan_obj: dict[str, Any], det_forecast: Optional[dict[str, Any]]
) -> None:
    if not isinstance(det_forecast, dict):
        return
    readiness = _as_dict(_as_dict(det_forecast.get("result")).get("readiness"))
    status = readiness.get("status")
    if status not in ("primed", "steady", "fatigued"):
        return
    plan_readiness = _as_dict(plan_obj.get("readiness"))
    if not plan_readiness:
        return

    score = readiness.get("score")
    prefix = (
        f"Deterministic readiness: {status} (score {score})."
        if isinstance(score, (int, float))
        else f"Deterministic readiness: {status}."
    )
    old = _as_str(plan_readiness.get("rationale"))
    plan_readiness["status"] = status
    plan_readiness["rationale"] = (
        f"{prefix} {old}" if old and not old.lower().startswith("deterministic") else prefix
    )
    plan_obj["readiness"] = plan_readiness

    notes = _as_list(plan_obj.get("data_notes"))
    note = "Readiness status was set from deterministic readiness_and_risk_forecast.json."
    if note not in notes:
        notes.append(note)
    plan_obj["data_notes"] = notes


def _apply_primary_goal(plan_obj: dict[str, Any], primary_goal: Optional[str]) -> None:
    goal = _as_str(primary_goal)
    meta = _as_dict(plan_obj.get("meta"))
    if goal and meta:
        meta["primary_goal"] = goal
        plan_obj["meta"] = meta


def _recompute_planned_hours(plan_obj: dict[str, Any]) -> None:
    """Recompute weekly_totals.planned_moving_time_hours from week 1 only.

    For multi-week plans, weekly_totals always reflects week 1 (days 1-7).
    """
    all_days = _as_list(_as_dict(plan_obj.get("plan")).get("days"))
    week1 = all_days[: min(7, len(all_days))]
    total_min = sum(
        float(d["duration_minutes"])
        for d in week1
        if isinstance(d, dict) and isinstance(d.get("duration_minutes"), (int, float))
    )
    wt = _as_dict(_as_dict(plan_obj.get("plan")).get("weekly_totals"))
    if wt:
        wt["planned_moving_time_hours"] = round(total_min / 60.0, 1)


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def training_plan_to_text(obj: dict[str, Any]) -> str:
    meta = _as_dict(obj.get("meta"))
    readiness = _as_dict(obj.get("readiness"))
    plan = _as_dict(obj.get("plan"))
    weekly = _as_dict(plan.get("weekly_totals"))
    day_objs = sorted(
        [d for d in _as_list(plan.get("days")) if isinstance(d, dict)],
        key=lambda d: d.get("date") or "9999-99-99",
    )

    lines: list[str] = ["TrailTraining - Training Plan", ""]

    if today := _as_str(meta.get("today")):
        lines.append(f"Generated: {today}")
    plan_start, plan_days = _as_str(meta.get("plan_start")), meta.get("plan_days")
    if plan_start or plan_days is not None:
        lines.append(
            f"Plan start: {plan_start or '(unknown)'}   Days: {plan_days if plan_days is not None else '(unknown)'}"
        )
    if style := _as_str(meta.get("style")):
        lines.append(f"Style: {style}")
    if goal := _as_str(meta.get("primary_goal")):
        lines.append(f"Primary goal: {goal}")

    status = _as_str(readiness.get("status"))
    rationale = _as_str(readiness.get("rationale"))
    if status or rationale:
        lines.append("")
        if status:
            lines.append(f"Readiness: {status}")
        if rationale:
            lines.append(f"Why: {rationale}")

    totals: list[str] = []
    if (h := _as_float(weekly.get("planned_moving_time_hours"))) is not None:
        totals.append(f"{h:.1f} h")
    if (km := _as_float(weekly.get("planned_distance_km"))) and km > 0:
        totals.append(f"{km:.0f} km")
    if (elev := _as_float(weekly.get("planned_elevation_m"))) and elev > 0:
        totals.append(f"{elev:.0f} m+")
    if totals:
        lines += ["", "Weekly totals: " + " • ".join(totals)]

    lines += ["", "Day-by-day", "-" * 10]
    for day_obj in day_objs:
        ds = _as_str(day_obj.get("date"))
        try:
            wd = date.fromisoformat(ds).strftime("%a") if ds else ""
        except Exception:
            wd = ""
        title = _as_str(day_obj.get("title")) or "(no title)"
        st = _as_str(day_obj.get("session_type"))
        is_rest = bool(day_obj.get("is_rest_day"))
        is_hard = bool(day_obj.get("is_hard_day"))
        mins = day_obj.get("duration_minutes")
        dur = f"{mins} min" if isinstance(mins, (int, float)) else "?"
        tags = (["REST"] if is_rest else ([st.upper()] if st else [])) + (
            ["HARD"] if is_hard and not is_rest else []
        )
        date_label = f"{wd} {ds}".strip() if wd or ds else "(unknown date)"
        lines.append(f"{date_label}: {title} ({', '.join(tags) or 'SESSION'}, {dur})")
        for field, label in [
            ("target_intensity", "Intensity"),
            ("terrain", "Terrain"),
            ("workout", "Workout"),
            ("purpose", "Purpose"),
        ]:
            if val := _as_str(day_obj.get(field)):
                lines.append(f"  {label}: {val}")
        lines.append("")

    recovery = _as_dict(obj.get("recovery"))
    actions = [a for a in _as_list(recovery.get("actions")) if isinstance(a, str) and a.strip()]
    if actions:
        lines += ["Recovery focus", "-" * 14]
        lines.extend(f"- {a.strip()}" for a in actions)
        lines.append("")

    risks = [r for r in _as_list(obj.get("risks")) if isinstance(r, dict)]
    if risks:
        lines += ["Risks / cautions", "-" * 15]
        for risk in risks:
            lines.append(
                f"- [{_as_str(risk.get('severity')) or 'unknown'}] {_as_str(risk.get('message')) or '(no message)'}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt_text(
    prompt_name: str,
    personal: Any,
    rollups: Optional[Any],
    combined: list[dict[str, Any]],
    deterministic_forecast: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
    max_chars: int,
    detail_days: int,
    plan_days: int = 7,
) -> str:
    retrieval_weeks = int(os.getenv("TRAILTRAINING_COACH_RETRIEVAL_WEEKS", "8"))

    sections: list[str] = [
        f"# TrailTraining Coach Brief: {prompt_name}",
        "",
        "## Evaluation context",
        f"Style: {style}",
        f"Primary goal (authoritative): {primary_goal}",
        f"Plan duration: {plan_days} days",
        "",
        "The output plan MUST target the primary goal above and copy it exactly into meta.primary_goal.",
        f"The output plan MUST contain exactly {plan_days} days in plan.days.",
        "",
        *_race_context_section(primary_goal),
        "## Personal profile (raw JSON)",
        _safe_json_snippet(personal, max_chars=50_000),
        "",
    ]

    if rollups is not None:
        sections += [
            "## Recent rollups (7d/28d)",
            _safe_json_snippet(rollups, max_chars=80_000),
            "",
        ]

    sections += [
        "## Eval-coach constraints (MUST satisfy)",
        build_eval_constraints_block(rollups if isinstance(rollups, dict) else None),
        "",
    ]

    if deterministic_forecast is not None:
        sections += _forecast_capability_block(deterministic_forecast)
        sections += [
            "## Deterministic readiness & overreach risk (authoritative)",
            _safe_json_snippet(deterministic_forecast, max_chars=40_000),
            "",
            "Use the deterministic readiness.status for readiness.status in your output.",
            "",
        ]

    ctx = build_retrieval_context(
        combined, rollups if isinstance(rollups, dict) else None, retrieval_weeks=retrieval_weeks
    )
    signal_registry = list(ctx.get("signal_registry") or [])
    if isinstance(deterministic_forecast, dict):
        signal_registry += _forecast_signal_rows(deterministic_forecast)

    sections += [
        f"## Retrieved history (weekly summaries; last {retrieval_weeks} weeks)",
        _safe_json_snippet(ctx.get("weekly_history"), max_chars=50_000),
        "",
        "## Signal registry (you MUST cite signal_ids from here)",
        _safe_json_snippet(signal_registry, max_chars=80_000),
        "",
    ]

    budget = max_chars if max_chars > 0 else 200_000
    base = "\n".join(sections)
    parts = [base]
    used = len(base)

    combined_detail = (
        combined[-detail_days:] if detail_days > 0 and len(combined) > detail_days else combined
    )
    if older_count := len(combined) - len(combined_detail):
        note = f"## Older days in window: {older_count} (details omitted; rely on rollups + recent detail)\n"
        parts.append(note)
        used += len(note)

    for day in reversed(combined_detail):
        block = _summarize_day(day)
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)

    tail = "\n## Task\n" + get_task_prompt(prompt_name, style=style, plan_days=plan_days) + "\n"
    if used + len(tail) <= budget:
        parts.append(tail)
        used += len(tail)

    if prompt_name == "training-plan":
        contract = "\n## Output Contract (STRICT)\n" + training_plan_output_contract_text() + "\n"
        if used + len(contract) <= budget:
            parts.append(contract)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoachConfig:
    model: str = "openai/gpt-4o"
    reasoning_effort: str = "medium"
    verbosity: str = "medium"
    days: int = 60
    max_chars: int = 200_000
    temperature: Optional[float] = None
    style: str = "trailrunning"
    primary_goal: str = ""
    plan_days: int = 7

    @classmethod
    def from_env(cls) -> CoachConfig:
        def _env_int(name: str, default: int) -> int:
            v = os.getenv(name)
            try:
                return int(v) if v and v.strip() else default
            except ValueError:
                return default

        style = os.getenv("TRAILTRAINING_COACH_STYLE", cls.style)
        primary_goal = os.getenv("TRAILTRAINING_PRIMARY_GOAL") or default_primary_goal_for_style(
            style
        )
        return cls(
            model=os.getenv("TRAILTRAINING_LLM_MODEL", cls.model),
            reasoning_effort=os.getenv("TRAILTRAINING_REASONING_EFFORT", cls.reasoning_effort),
            verbosity=os.getenv("TRAILTRAINING_VERBOSITY", cls.verbosity),
            days=_env_int("TRAILTRAINING_COACH_DAYS", cls.days),
            max_chars=_env_int("TRAILTRAINING_COACH_MAX_CHARS", cls.max_chars),
            style=style,
            primary_goal=primary_goal,
            plan_days=_env_int("TRAILTRAINING_PLAN_DAYS", cls.plan_days),
        )


# ---------------------------------------------------------------------------
# Training-plan pipeline
# ---------------------------------------------------------------------------


def _parse_training_plan(
    out_text: str, client: OpenAI, cfg: CoachConfig, system_instructions: str
) -> dict[str, Any]:
    try:
        obj = ensure_training_plan_shape(json.loads(_extract_json_object(out_text)))
        _recompute_planned_hours(obj)
        return obj
    except Exception as exc:
        log.warning("Training-plan JSON parse/shape failed; attempting repair: %s", exc)

    repair_resp = _call_with_param_fallback(
        client,
        {
            "model": cfg.model,
            "instructions": system_instructions,
            "input": (
                f"Return ONLY valid JSON matching this schema:\n{TRAINING_PLAN_SCHEMA.get('schema')}\n\n"
                f"Your previous output was invalid. Fix it:\n{out_text}\n"
            ),
            "reasoning": {"effort": "none"},
            "text": {"verbosity": "low"},
        },
    )
    repaired = getattr(repair_resp, "output_text", None) or str(repair_resp)
    obj = ensure_training_plan_shape(json.loads(_extract_json_object(repaired)))
    _recompute_planned_hours(obj)
    return obj


def _run_training_plan(
    client: OpenAI,
    api_kwargs: dict[str, Any],
    cfg: CoachConfig,
    resolved_goal: str,
    deterministic_forecast: Optional[dict[str, Any]],
    rollups: Optional[Any],
    output_path: Optional[str],
) -> tuple[str, str]:
    system_instructions = api_kwargs.get("instructions", "")
    resp = _call_with_schema(client, api_kwargs, TRAINING_PLAN_SCHEMA)
    out_text = getattr(resp, "output_text", None) or str(resp)

    obj = _parse_training_plan(out_text, client, cfg, system_instructions)
    _apply_primary_goal(obj, resolved_goal)
    _apply_deterministic_readiness(obj, deterministic_forecast)
    apply_eval_coach_guardrails(obj, rollups if isinstance(rollups, dict) else None)

    out_p = (
        Path(output_path).expanduser().resolve()
        if output_path
        else Path(config.PROMPTING_DIRECTORY) / "coach_brief_training-plan.json"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_p, obj, compact=False)

    try:
        (out_p.parent / f"{out_p.stem}.txt").write_text(
            training_plan_to_text(obj), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("Failed to write training-plan text: %s", exc)

    return json.dumps(obj, indent=2, ensure_ascii=False), str(out_p)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_coach_brief(
    *,
    prompt: str,
    cfg: CoachConfig,
    input_path: Optional[str] = None,
    personal_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    config.ensure_directories()

    personal_p, summary_p, rollups_p = _resolve_input_paths(input_path, personal_path, summary_path)
    personal = load_json(personal_p, default={})
    combined = load_json(summary_p, default=[])
    rollups = load_json(rollups_p, default=None) if rollups_p else None

    if not isinstance(combined, list):
        raise RuntimeError("combined_summary.json must be a list of day objects")

    _dedup_activities_in_place(combined)
    combined = _filter_last_days(combined, cfg.days)

    deterministic_forecast = _load_or_compute_deterministic_forecast(summary_p.parent, combined)
    detail_days = (
        max(1, min(int(os.getenv("TRAILTRAINING_COACH_DETAIL_DAYS", "14")), len(combined)))
        if combined
        else 0
    )
    resolved_goal = (cfg.primary_goal or "").strip() or default_primary_goal_for_style(cfg.style)

    prompt_text = _build_prompt_text(
        prompt_name=prompt,
        personal=personal,
        rollups=rollups,
        combined=combined,
        deterministic_forecast=deterministic_forecast,
        style=cfg.style,
        primary_goal=resolved_goal,
        max_chars=cfg.max_chars,
        detail_days=detail_days,
        plan_days=cfg.plan_days,
    )

    client = _make_openrouter_client()
    api_kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": get_system_prompt(cfg.style),
        "input": prompt_text,
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {"verbosity": cfg.verbosity},
    }
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        api_kwargs["temperature"] = cfg.temperature

    if prompt == "training-plan":
        return _run_training_plan(
            client, api_kwargs, cfg, resolved_goal, deterministic_forecast, rollups, output_path
        )

    resp = _call_with_param_fallback(client, api_kwargs)
    out_text = getattr(resp, "output_text", None) or str(resp)
    out_p = (
        Path(output_path).expanduser().resolve()
        if output_path
        else Path(config.PROMPTING_DIRECTORY) / f"coach_brief_{prompt}.md"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(out_text, encoding="utf-8")
    return out_text, str(out_p)
