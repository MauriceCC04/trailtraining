from __future__ import annotations

import os
from typing import Any, Optional

from trailtraining.llm.constraints import (
    EffectiveConstraintContext,
    constraint_config_from_env,
    derive_effective_constraints,
)
from trailtraining.llm.guardrails import build_eval_constraints_block
from trailtraining.llm.presets import get_task_prompt
from trailtraining.llm.schemas import (
    machine_plan_output_contract_text,
    plan_explanation_stage_output_contract_text,
    training_plan_output_contract_text,
)
from trailtraining.llm.shared import race_context_section as _race_context_section
from trailtraining.llm.signals import build_retrieval_context
from trailtraining.util.text import _safe_json_snippet


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _summarize_activity(activity: dict[str, Any]) -> str:
    parts = [str(activity.get("sport_type") or activity.get("type") or "unknown")]
    distance = activity.get("distance")
    elevation = activity.get("total_elevation_gain")
    moving_time = activity.get("moving_time")
    avg_hr = activity.get("average_heartrate")
    name = activity.get("name")
    if isinstance(distance, (int, float)):
        parts.append(f"{distance / 1000:.2f} km")
    if isinstance(elevation, (int, float)):
        parts.append(f"{elevation:.0f} m+")
    if isinstance(moving_time, (int, float)):
        parts.append(f"{moving_time / 60:.0f} min")
    if isinstance(avg_hr, (int, float)):
        parts.append(f"avgHR {avg_hr:.0f}")
    if isinstance(name, str) and name.strip():
        parts.append(f"({name.strip()})")
    return " - " + " | ".join(parts)


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
    result = _as_dict(det_forecast.get("result"))
    day_value = result.get("date")
    date_range = f"{day_value}..{day_value}" if isinstance(day_value, str) and day_value else ""
    inputs = _as_dict(result.get("inputs"))
    source_root = "readiness_and_risk_forecast.json:result"

    def _row(signal_id: str, value: Any, path: str, unit: str = "") -> dict[str, Any]:
        return {
            "signal_id": signal_id,
            "value": value,
            "unit": unit,
            "source": f"{source_root}.{path}",
            "date_range": date_range,
        }

    rows: list[dict[str, Any]] = []
    label = _as_str(inputs.get("recovery_capability_label"))
    if label:
        rows.append(
            _row("forecast.recovery_capability.label", label, "inputs.recovery_capability_label")
        )
    key = _as_str(inputs.get("recovery_capability_key"))
    if key:
        rows.append(_row("forecast.recovery_capability.key", key, "inputs.recovery_capability_key"))

    readiness = _as_dict(result.get("readiness"))
    rows.extend(
        [
            _row("forecast.readiness.status", readiness.get("status"), "readiness.status"),
            _row("forecast.readiness.score", readiness.get("score"), "readiness.score"),
        ]
    )

    for input_key, signal_id in [
        ("sleep_days_7d", "forecast.recovery_capability.sleep_days_7d"),
        ("resting_hr_days_7d", "forecast.recovery_capability.resting_hr_days_7d"),
        ("hrv_days_7d", "forecast.recovery_capability.hrv_days_7d"),
    ]:
        rows.append(_row(signal_id, inputs.get(input_key), f"inputs.{input_key}", "days"))

    risk = _as_dict(result.get("overreach_risk"))
    rows.extend(
        [
            _row("forecast.overreach_risk.level", risk.get("level"), "overreach_risk.level"),
            _row("forecast.overreach_risk.score", risk.get("score"), "overreach_risk.score"),
        ]
    )
    return rows


def _lifestyle_notes_section(lifestyle_notes: str) -> list[str]:
    notes = lifestyle_notes.strip() if isinstance(lifestyle_notes, str) else ""
    if not notes:
        return []
    return [
        "## Lifestyle constraints (authoritative - the athlete's real-world schedule)",
        notes,
        "",
        "These constraints reflect the athlete's actual availability, not a training preference.",
        "The plan MUST respect them. For example, if the athlete can only run on roads during",
        "the week, weekday road runs are correct - not a failure of trail specificity.",
        "Copy these constraints into meta.lifestyle_notes in your output.",
        "",
    ]


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
        picked = {key: sleep[key] for key in wanted if key in sleep}
        lines.append(f"Sleep: {picked}" if picked else "Sleep: (data present)")
    else:
        lines.append("Sleep: (none)")

    activities = day.get("activities") or []
    if isinstance(activities, list) and activities:
        lines.append(f"Activities ({len(activities)}):")
        for activity in activities[:50]:
            if isinstance(activity, dict):
                lines.append(_summarize_activity(activity))
    else:
        lines.append("Activities: (none)")

    return "\n".join(lines) + "\n"


def _build_common_sections(
    *,
    prompt_name: str,
    personal: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    combined: list[dict[str, Any]],
    deterministic_forecast: Optional[dict[str, Any]],
    style: str,
    primary_goal: str,
    lifestyle_notes: str,
    effective_constraints: Optional[EffectiveConstraintContext],
) -> list[str]:
    cfg = constraint_config_from_env()
    effective = effective_constraints or derive_effective_constraints(
        det_forecast=deterministic_forecast if isinstance(deterministic_forecast, dict) else None,
        rollups=rollups,
        cfg=cfg,
        lifestyle_notes=lifestyle_notes,
    )

    retrieval_weeks = int(os.getenv("TRAILTRAINING_COACH_RETRIEVAL_WEEKS", "8"))

    sections: list[str] = [
        f"# TrailTraining Coach Brief: {prompt_name}",
        "",
        "## Evaluation context",
        f"Style: {style}",
        f"Primary goal (authoritative): {primary_goal}",
        "",
        *_lifestyle_notes_section(lifestyle_notes),
        *_race_context_section(primary_goal),
        "## Eval-coach constraints (MUST satisfy)",
        build_eval_constraints_block(rollups, effective),
        "",
        "## Personal profile (raw JSON)",
        _safe_json_snippet(personal, max_chars=50_000),
        "",
    ]

    if rollups is not None:
        sections.extend(
            [
                "## Recent rollups (7d/28d)",
                _safe_json_snippet(rollups, max_chars=80_000),
                "",
            ]
        )

    if deterministic_forecast is not None:
        sections.extend(_forecast_capability_block(deterministic_forecast))
        sections.extend(
            [
                "## Deterministic readiness and overreach risk (authoritative)",
                _safe_json_snippet(deterministic_forecast, max_chars=40_000),
                "",
                "Use the deterministic readiness.status for readiness.status in your output.",
                "",
            ]
        )

    ctx = build_retrieval_context(combined, rollups, retrieval_weeks=retrieval_weeks)
    signal_registry_raw = ctx.get("signal_registry")
    signal_registry = list(signal_registry_raw) if isinstance(signal_registry_raw, list) else []
    if isinstance(deterministic_forecast, dict):
        signal_registry.extend(_forecast_signal_rows(deterministic_forecast))

    sections.extend(
        [
            f"## Retrieved history (weekly summaries; last {retrieval_weeks} weeks)",
            _safe_json_snippet(ctx.get("weekly_history"), max_chars=50_000),
            "",
            "## Signal registry (you MUST cite signal_ids from here)",
            _safe_json_snippet(signal_registry, max_chars=80_000),
            "",
        ]
    )
    return sections


def build_prompt_text(
    prompt_name: str,
    personal: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    combined: list[dict[str, Any]],
    deterministic_forecast: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
    lifestyle_notes: str,
    max_chars: int,
    detail_days: int,
    plan_days: int = 7,
    effective_constraints: Optional[EffectiveConstraintContext] = None,
) -> str:
    sections = _build_common_sections(
        prompt_name=prompt_name,
        personal=personal,
        rollups=rollups,
        combined=combined,
        deterministic_forecast=deterministic_forecast,
        style=style,
        primary_goal=primary_goal,
        lifestyle_notes=lifestyle_notes,
        effective_constraints=effective_constraints,
    )
    sections[
        sections.index("## Evaluation context") + 3 : sections.index("## Evaluation context") + 3
    ] = [
        f"Plan duration: {plan_days} days",
        "",
        "The output plan MUST target the primary goal above and copy it exactly into meta.primary_goal.",
        f"The output plan MUST contain exactly {plan_days} days in plan.days.",
        "",
    ]

    budget = max_chars if max_chars > 0 else 200_000
    base = "\n".join(sections)
    parts = [base]
    used = len(base)

    combined_detail = (
        combined[-detail_days:] if detail_days > 0 and len(combined) > detail_days else combined
    )
    older_count = len(combined) - len(combined_detail)
    if older_count:
        note = (
            f"## Older days in window: {older_count} "
            "(details omitted; rely on rollups + recent detail)\n"
        )
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


def build_machine_plan_prompt_text(
    personal: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    combined: list[dict[str, Any]],
    deterministic_forecast: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
    lifestyle_notes: str,
    max_chars: int,
    detail_days: int,
    plan_days: int = 7,
    effective_constraints: Optional[EffectiveConstraintContext] = None,
) -> str:
    prompt = build_prompt_text(
        prompt_name="training-plan",
        personal=personal,
        rollups=rollups,
        combined=combined,
        deterministic_forecast=deterministic_forecast,
        style=style,
        primary_goal=primary_goal,
        lifestyle_notes=lifestyle_notes,
        max_chars=max_chars,
        detail_days=detail_days,
        plan_days=plan_days,
        effective_constraints=effective_constraints,
    )
    return (
        prompt
        + "\n## Stage\n"
        + "This is STAGE A: produce ONLY the compact machine plan.\n"
        + "\n## Output Contract (STRICT)\n"
        + machine_plan_output_contract_text()
        + "\n"
    )


def build_explainer_prompt_text(
    machine_plan: dict[str, Any],
    personal: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    combined: list[dict[str, Any]],
    deterministic_forecast: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
    lifestyle_notes: str,
    max_chars: int,
    detail_days: int,
    effective_constraints: Optional[EffectiveConstraintContext] = None,
) -> str:
    sections = _build_common_sections(
        prompt_name="training-plan-explainer",
        personal=personal,
        rollups=rollups,
        combined=combined,
        deterministic_forecast=deterministic_forecast,
        style=style,
        primary_goal=primary_goal,
        lifestyle_notes=lifestyle_notes,
        effective_constraints=effective_constraints,
    )

    sections.extend(
        [
            "## Locked machine plan (DO NOT CHANGE)",
            _safe_json_snippet(machine_plan, max_chars=30_000),
            "",
            "## Task",
            "Write explanations, purposes, risks, recovery actions, and data notes against the locked plan.",
            "Use signal_ids to justify the explanation fields you write.",
            "Do not emit citations or claim_attributions in this stage; those are derived deterministically downstream.",
            "Do not change dates, durations, session_type, is_rest_day, is_hard_day, target_intensity, terrain, workout, or weekly_totals.",
            "",
            "## Output Contract (STRICT)",
            plan_explanation_stage_output_contract_text(),
            "",
        ]
    )

    budget = max_chars if max_chars > 0 else 200_000
    base = "\n".join(sections)
    parts = [base]
    used = len(base)

    combined_detail = (
        combined[-detail_days:] if detail_days > 0 and len(combined) > detail_days else combined
    )
    older_count = len(combined) - len(combined_detail)
    if older_count:
        note = (
            f"## Older days in window: {older_count} "
            "(details omitted; rely on rollups + recent detail)\n"
        )
        parts.append(note)
        used += len(note)

    for day in reversed(combined_detail):
        block = _summarize_day(day)
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)

    return "\n".join(parts)
