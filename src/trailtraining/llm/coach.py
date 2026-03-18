from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
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
from trailtraining.llm.shared import (
    apply_primary_goal as _apply_primary_goal,
)
from trailtraining.llm.shared import (
    call_with_param_fallback as _call_with_param_fallback,
)
from trailtraining.llm.shared import (
    call_with_schema as _call_with_schema,
)
from trailtraining.llm.shared import (
    extract_json_object as _extract_json_object,
)
from trailtraining.llm.shared import (
    make_openrouter_client as _make_openrouter_client,
)
from trailtraining.llm.shared import (
    race_context_section as _race_context_section,
)
from trailtraining.llm.shared import (
    recompute_planned_hours as _recompute_planned_hours,
)
from trailtraining.llm.shared import (
    training_plan_to_text,
)
from trailtraining.llm.signals import build_retrieval_context
from trailtraining.util.dates import _as_date
from trailtraining.util.errors import ArtifactError, MissingArtifactError
from trailtraining.util.state import _json_default, load_json, save_json
from trailtraining.util.text import _safe_json_snippet

log = logging.getLogger(__name__)


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _as_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _resolve_input_paths(
    input_path: Optional[str],
    personal_path: Optional[str],
    summary_path: Optional[str],
    *,
    prompting_dir: Path,
) -> tuple[Path, Path, Optional[Path]]:
    base = Path(input_path).expanduser().resolve() if input_path else prompting_dir
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


def _load_required_object_artifact(path: Path, *, producer_hint: str) -> dict[str, Any]:
    raw = load_json(path, default=None)
    if raw is None:
        raise MissingArtifactError(
            message=f"Missing required artifact: {path}",
            hint=producer_hint,
        )
    if not isinstance(raw, dict):
        raise ArtifactError(
            message=f"{path.name} must be a JSON object.",
            hint=f"Got {type(raw).__name__} in {path}.",
        )
    if not raw:
        raise MissingArtifactError(
            message=f"Required artifact is empty: {path}",
            hint=producer_hint,
        )
    return raw


def _load_required_list_artifact(path: Path, *, producer_hint: str) -> list[dict[str, Any]]:
    raw = load_json(path, default=None)
    if raw is None:
        raise MissingArtifactError(
            message=f"Missing required artifact: {path}",
            hint=producer_hint,
        )
    if not isinstance(raw, list):
        raise ArtifactError(
            message=f"{path.name} must be a list of day objects.",
            hint=f"Got {type(raw).__name__} in {path}.",
        )
    if not raw:
        raise MissingArtifactError(
            message=f"Required artifact is empty: {path}",
            hint=producer_hint,
        )
    return raw


def _dedup_activities_in_place(combined: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for day in combined:
        acts = day.get("activities")
        if not isinstance(acts, list):
            day["activities"] = []
            continue
        unique = []
        for activity in acts:
            if not isinstance(activity, dict):
                continue
            aid = activity.get("id")
            if aid is None:
                unique.append(activity)
            elif (key := str(aid)) not in seen:
                seen.add(key)
                unique.append(activity)
        day["activities"] = unique


def _filter_last_days(combined: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    if not combined:
        return combined
    last = _as_date(combined[-1].get("date", ""))
    if not last:
        return combined
    cutoff = last - timedelta(days=days - 1)
    return [
        day
        for day in combined
        if (day_date := _as_date(day.get("date", ""))) and day_date >= cutoff
    ]


def _summarize_activity(activity: dict[str, Any]) -> str:
    parts = [str(activity.get("sport_type") or activity.get("type") or "unknown")]
    if isinstance(dist := activity.get("distance"), (int, float)):
        parts.append(f"{dist / 1000:.2f} km")
    if isinstance(elev := activity.get("total_elevation_gain"), (int, float)):
        parts.append(f"{elev:.0f} m+")
    if isinstance(moving_time := activity.get("moving_time"), (int, float)):
        parts.append(f"{moving_time / 60:.0f} min")
    if isinstance(hr := activity.get("average_heartrate"), (int, float)):
        parts.append(f"avgHR {hr:.0f}")
    if isinstance(name := activity.get("name"), str) and name.strip():
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

    activities = day.get("activities") or []
    if isinstance(activities, list) and activities:
        lines.append(f"Activities ({len(activities)}):")
        lines.extend(
            _summarize_activity(activity)
            for activity in activities[:50]
            if isinstance(activity, dict)
        )
    else:
        lines.append("Activities: (none)")

    return "\n".join(lines) + "\n"


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
    day_value = res.get("date")
    date_range = f"{day_value}..{day_value}" if isinstance(day_value, str) and day_value else ""
    inputs = _as_dict(res.get("inputs"))
    src = "readiness_and_risk_forecast.json:result"

    def _row(signal_id: str, value: Any, path: str, unit: str = "") -> dict[str, Any]:
        return {
            "signal_id": signal_id,
            "value": value,
            "unit": unit,
            "source": f"{src}.{path}",
            "date_range": date_range,
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

    for key, signal_id in [
        ("sleep_days_7d", "forecast.recovery_capability.sleep_days_7d"),
        ("resting_hr_days_7d", "forecast.recovery_capability.resting_hr_days_7d"),
        ("hrv_days_7d", "forecast.recovery_capability.hrv_days_7d"),
    ]:
        rows.append(_row(signal_id, inputs.get(key), f"inputs.{key}", "days"))

    risk = _as_dict(res.get("overreach_risk"))
    rows += [
        _row("forecast.overreach_risk.level", risk.get("level"), "overreach_risk.level"),
        _row("forecast.overreach_risk.score", risk.get("score"), "overreach_risk.score"),
    ]
    return rows


def _apply_deterministic_readiness(
    plan_obj: dict[str, Any],
    det_forecast: Optional[dict[str, Any]],
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
        combined,
        rollups if isinstance(rollups, dict) else None,
        retrieval_weeks=retrieval_weeks,
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
            value = os.getenv(name)
            try:
                return int(value) if value and value.strip() else default
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


def _parse_training_plan(
    out_text: str,
    client: OpenAI,
    cfg: CoachConfig,
    system_instructions: str,
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
    *,
    prompting_dir: Path,
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
        else prompting_dir / "coach_brief_training-plan.json"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_p, obj, compact=False)

    try:
        (out_p.parent / f"{out_p.stem}.txt").write_text(
            training_plan_to_text(obj), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("Failed to write training-plan text: %s", exc)

    return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), str(out_p)


def run_coach_brief(
    *,
    prompt: str,
    cfg: CoachConfig,
    input_path: Optional[str] = None,
    personal_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    output_path: Optional[str] = None,
    runtime: Optional[config.RuntimeConfig] = None,
) -> tuple[str, Optional[str]]:
    runtime = runtime or config.current()
    config.ensure_directories(runtime)
    paths = runtime.paths

    personal_p, summary_p, rollups_p = _resolve_input_paths(
        input_path,
        personal_path,
        summary_path,
        prompting_dir=paths.prompting_directory,
    )

    producer_hint = (
        "Run `trailtraining combine` (or `trailtraining run-all`) to generate the required inputs."
    )
    personal = _load_required_object_artifact(personal_p, producer_hint=producer_hint)
    combined = _load_required_list_artifact(summary_p, producer_hint=producer_hint)
    rollups = load_json(rollups_p, default=None) if rollups_p else None

    _dedup_activities_in_place(combined)
    combined = _filter_last_days(combined, cfg.days)
    if not combined:
        raise MissingArtifactError(
            message=(
                f"{summary_p.name} contains no usable day objects in the last {cfg.days} days."
            ),
            hint="Fetch fresh source data and rerun `trailtraining combine`.",
        )

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
            client,
            api_kwargs,
            cfg,
            resolved_goal,
            deterministic_forecast,
            rollups,
            output_path,
            prompting_dir=paths.prompting_directory,
        )

    resp = _call_with_param_fallback(client, api_kwargs)
    out_text = getattr(resp, "output_text", None) or str(resp)
    out_p = (
        Path(output_path).expanduser().resolve()
        if output_path
        else paths.prompting_directory / f"coach_brief_{prompt}.md"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(out_text, encoding="utf-8")
    return out_text, str(out_p)
