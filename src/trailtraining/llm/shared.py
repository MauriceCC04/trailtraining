from __future__ import annotations

import calendar
import contextlib
import logging
import os
import re
from datetime import date
from typing import Any, Optional

from openai import OpenAI

from trailtraining.util.errors import LLMUnsupportedParameterError
from trailtraining.util.llm_helpers import _classify_and_raise

log = logging.getLogger(__name__)

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


def parse_race_context(goal_text: str, today: Optional[date] = None) -> dict[str, Any]:
    today = today or date.today()
    low = goal_text.strip().lower()
    race_date: Optional[date] = None
    is_approximate = False

    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", goal_text)
    if match:
        with contextlib.suppress(ValueError):
            race_date = date.fromisoformat(match.group(1))

    if not race_date:
        match = re.search(rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", low)
        if match:
            month = _MONTH_MAP.get(match.group(1))
            if month:
                with contextlib.suppress(ValueError):
                    race_date = date(int(match.group(3)), month, int(match.group(2)))

    if not race_date:
        match = re.search(rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b", low)
        if match:
            month = _MONTH_MAP.get(match.group(1))
            if month:
                with contextlib.suppress(ValueError):
                    year = int(match.group(2))
                    last_day = calendar.monthrange(year, month)[1]
                    race_date = date(year, month, last_day)
                    is_approximate = True

    if not race_date:
        match = re.search(rf"\b(?:in\s+)?({_MONTH_PATTERN})\b", low)
        if match:
            month = _MONTH_MAP.get(match.group(1))
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


def race_context_section(primary_goal: str) -> list[str]:
    ctx = parse_race_context(primary_goal)
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


def make_openrouter_client() -> OpenAI:
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

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers=headers,
    )


def call_with_param_fallback(client: OpenAI, kwargs: dict[str, Any]) -> Any:
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


def call_with_schema(client: OpenAI, kwargs: dict[str, Any], schema: dict[str, Any]) -> Any:
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
            return call_with_param_fallback(client, kw)
        except LLMUnsupportedParameterError as exc:
            log.warning("Structured output mode %s unavailable; falling back: %s", label, exc)

    return call_with_param_fallback(client, kwargs)


def extract_json_object(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _as_str(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, date):
        return v.isoformat()
    return ""


def _as_float(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


def apply_primary_goal(plan_obj: dict[str, Any], primary_goal: Optional[str]) -> None:
    goal = _as_str(primary_goal)
    meta = _as_dict(plan_obj.get("meta"))
    if goal and meta:
        meta["primary_goal"] = goal
        plan_obj["meta"] = meta


def recompute_planned_hours(plan_obj: dict[str, Any]) -> None:
    all_days = _as_list(_as_dict(plan_obj.get("plan")).get("days"))
    week1 = all_days[: min(7, len(all_days))]
    total_min = sum(
        float(day["duration_minutes"])
        for day in week1
        if isinstance(day, dict) and isinstance(day.get("duration_minutes"), (int, float))
    )
    weekly_totals = _as_dict(_as_dict(plan_obj.get("plan")).get("weekly_totals"))
    if weekly_totals:
        weekly_totals["planned_moving_time_hours"] = round(total_min / 60.0, 1)


def training_plan_to_text(obj: dict[str, Any]) -> str:
    meta = _as_dict(obj.get("meta"))
    readiness = _as_dict(obj.get("readiness"))
    plan = _as_dict(obj.get("plan"))
    weekly = _as_dict(plan.get("weekly_totals"))
    day_objs = sorted(
        [day for day in _as_list(plan.get("days")) if isinstance(day, dict)],
        key=lambda day: day.get("date") or "9999-99-99",
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
    if lifestyle := _as_str(meta.get("lifestyle_notes")):
        lines.append(f"Lifestyle constraints: {lifestyle}")

    status = _as_str(readiness.get("status"))
    rationale = _as_str(readiness.get("rationale"))
    if status or rationale:
        lines.append("")
        if status:
            lines.append(f"Readiness: {status}")
        if rationale:
            lines.append(f"Why: {rationale}")

    totals: list[str] = []
    if (hours := _as_float(weekly.get("planned_moving_time_hours"))) is not None:
        totals.append(f"{hours:.1f} h")
    if (km := _as_float(weekly.get("planned_distance_km"))) and km > 0:
        totals.append(f"{km:.0f} km")
    if (elev := _as_float(weekly.get("planned_elevation_m"))) and elev > 0:
        totals.append(f"{elev:.0f} m+")
    if totals:
        lines += ["", "Weekly totals: " + " • ".join(totals)]

    lines += ["", "Day-by-day", "-" * 10]
    for day_obj in day_objs:
        day_str = _as_str(day_obj.get("date"))
        try:
            weekday = date.fromisoformat(day_str).strftime("%a") if day_str else ""
        except Exception:
            weekday = ""

        title = _as_str(day_obj.get("title")) or "(no title)"
        session_type = _as_str(day_obj.get("session_type"))
        is_rest = bool(day_obj.get("is_rest_day"))
        is_hard = bool(day_obj.get("is_hard_day"))
        mins = day_obj.get("duration_minutes")
        duration = f"{mins} min" if isinstance(mins, (int, float)) else "?"
        tags = (["REST"] if is_rest else ([session_type.upper()] if session_type else [])) + (
            ["HARD"] if is_hard and not is_rest else []
        )
        date_label = f"{weekday} {day_str}".strip() if weekday or day_str else "(unknown date)"
        lines.append(f"{date_label}: {title} ({', '.join(tags) or 'SESSION'}, {duration})")

        for field, label in [
            ("target_intensity", "Intensity"),
            ("terrain", "Terrain"),
            ("workout", "Workout"),
            ("purpose", "Purpose"),
        ]:
            if value := _as_str(day_obj.get(field)):
                lines.append(f"  {label}: {value}")
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


__all__ = [
    "apply_primary_goal",
    "call_with_param_fallback",
    "call_with_schema",
    "extract_json_object",
    "make_openrouter_client",
    "parse_race_context",
    "race_context_section",
    "recompute_planned_hours",
    "training_plan_to_text",
]
