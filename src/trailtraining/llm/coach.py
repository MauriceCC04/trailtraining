# src/trailtraining/llm/coach.py

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from trailtraining import config
from trailtraining.llm.guardrails import apply_eval_coach_guardrails, build_eval_constraints_block
from trailtraining.llm.presets import get_system_prompt, get_task_prompt
from trailtraining.llm.schemas import (
    TRAINING_PLAN_SCHEMA,
    ensure_training_plan_shape,
    training_plan_output_contract_text,
)
from trailtraining.llm.signals import build_retrieval_context
from trailtraining.util.state import load_json, save_json

log = logging.getLogger(__name__)


def _as_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _coerce_path(p: Optional[str]) -> Optional[Path]:
    return Path(p).expanduser().resolve() if p else None


def _resolve_input_paths(
    input_path: Optional[str],
    personal_path: Optional[str],
    summary_path: Optional[str],
) -> tuple[Path, Path, Optional[Path]]:
    """
    Mirrors your CLI behavior:
      - if --personal/--summary provided, use them
      - else use --input (dir) or prompting directory
    """
    base: Path
    if input_path:
        base = Path(input_path).expanduser().resolve()
    else:
        base = Path(config.PROMPTING_DIRECTORY).expanduser().resolve()

    if base.is_file() and base.suffix.lower() == ".zip":
        raise RuntimeError(
            "Zip input not supported in this optimized version. Use a directory path."
        )

    personal = _coerce_path(personal_path) or (base / "formatted_personal_data.json")
    summary = _coerce_path(summary_path) or (base / "combined_summary.json")
    rollups = base / "combined_rollups.json"
    return personal, summary, (rollups if rollups.exists() else None)


def _dedup_activities_in_place(combined: list[dict[str, Any]]) -> None:
    seen = set()
    for day in combined:
        acts = day.get("activities")
        if not isinstance(acts, list):
            day["activities"] = []
            continue
        new_acts = []
        for a in acts:
            if not isinstance(a, dict):
                continue
            aid = a.get("id")
            if aid is None:
                new_acts.append(a)
                continue
            key = str(aid)
            if key in seen:
                continue
            seen.add(key)
            new_acts.append(a)
        day["activities"] = new_acts


def _filter_last_days(combined: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    if days <= 0:
        return combined
    if not combined:
        return combined

    # assume combined sorted by date asc (as produced by combine.py)
    last = _as_date(combined[-1].get("date", ""))
    if not last:
        return combined

    cutoff = last - timedelta(days=days - 1)
    out = []
    for d in combined:
        ds = d.get("date")
        if not isinstance(ds, str):
            continue
        dd = _as_date(ds)
        if dd and dd >= cutoff:
            out.append(d)
    return out


def _summarize_activity(a: dict[str, Any]) -> str:
    sport = a.get("sport_type") or a.get("type") or "unknown"
    dist_m = a.get("distance")
    elev_m = a.get("total_elevation_gain")
    mv_s = a.get("moving_time")
    hr = a.get("average_heartrate")

    parts = [str(sport)]
    if isinstance(dist_m, (int, float)):
        parts.append(f"{dist_m / 1000.0:.2f} km")
    if isinstance(elev_m, (int, float)):
        parts.append(f"{elev_m:.0f} m+")
    if isinstance(mv_s, (int, float)):
        parts.append(f"{mv_s / 60.0:.0f} min")
    if isinstance(hr, (int, float)):
        parts.append(f"avgHR {hr:.0f}")

    name = a.get("name")
    if isinstance(name, str) and name.strip():
        parts.append(f"({name.strip()})")

    return " • " + " | ".join(parts)


def _summarize_day(day: dict[str, Any]) -> str:
    d = day.get("date", "unknown-date")
    lines = [f"## {d}"]

    sleep = day.get("sleep")
    if isinstance(sleep, dict):
        # keep it lightweight: only include a few small fields if present
        keys = [
            "sleep_score",
            "score",
            "duration",
            "total_sleep",
            "resting_hr",
            "rhr",
            "readiness",
            "stress",
        ]
        picked = {k: sleep.get(k) for k in keys if k in sleep}
        if picked:
            lines.append(f"Sleep: {picked}")
        else:
            # fallback: avoid dumping huge dicts
            lines.append("Sleep: (data present)")
    elif sleep is None:
        lines.append("Sleep: (none)")

    acts = day.get("activities") or []
    if isinstance(acts, list) and acts:
        lines.append(f"Activities ({len(acts)}):")
        for a in acts[:50]:  # hard cap per day
            if isinstance(a, dict):
                lines.append(_summarize_activity(a))
    else:
        lines.append("Activities: (none)")

    return "\n".join(lines) + "\n"


def _safe_json_snippet(obj: Any, *, max_chars: int) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + "…"
    return s


def _extract_json_object(text: str) -> str:
    """
    Robust parsing helper when the model accidentally adds extra text.
    Grabs the outermost {...} region.
    """
    if not isinstance(text, str):
        return str(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _load_or_compute_deterministic_forecast(
    base_dir: Path,
    combined: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Best-effort:
      1) Load base_dir/readiness_and_risk_forecast.json if it exists
      2) Else, try to compute via trailtraining.forecast.forecast.compute_readiness_and_risk(combined)
         and (best-effort) write it back to the same path for next time.
    """
    forecast_p = base_dir / "readiness_and_risk_forecast.json"
    if forecast_p.exists():
        obj = load_json(forecast_p, default=None)
        return obj if isinstance(obj, dict) else None

    # Best-effort compute (won't crash coach if module isn't installed yet)
    try:
        from trailtraining.forecast.forecast import compute_readiness_and_risk  # type: ignore
    except Exception:
        return None

    try:
        fr = compute_readiness_and_risk(combined)
        payload: dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "result": {
                "date": getattr(fr, "date", None),
                "readiness": {
                    "score": getattr(fr, "readiness_score", None),
                    "status": getattr(fr, "readiness_status", None),
                },
                "overreach_risk": {
                    "score": getattr(fr, "overreach_risk_score", None),
                    "level": getattr(fr, "overreach_risk_level", None),
                },
                "inputs": getattr(fr, "inputs", None),
                "drivers": getattr(fr, "drivers", None),
            },
        }
        with contextlib.suppress(Exception):
            save_json(forecast_p, payload, compact=False)
        return payload
    except Exception:
        return None


def _forecast_signal_rows(det_forecast: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Converts readiness_and_risk_forecast.json into signal_registry rows so the model can cite them.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(det_forecast, dict):
        return out
    res = det_forecast.get("result")
    if not isinstance(res, dict):
        return out

    d = res.get("date")
    date_range = f"{d}..{d}" if isinstance(d, str) and d else ""

    readiness = res.get("readiness")
    if isinstance(readiness, dict):
        st = readiness.get("status")
        sc = readiness.get("score")
        out.append(
            {
                "signal_id": "forecast.readiness.status",
                "value": st,
                "unit": "",
                "source": "readiness_and_risk_forecast.json:result.readiness.status",
                "date_range": date_range,
            }
        )
        out.append(
            {
                "signal_id": "forecast.readiness.score",
                "value": sc,
                "unit": "0-100",
                "source": "readiness_and_risk_forecast.json:result.readiness.score",
                "date_range": date_range,
            }
        )

    risk = res.get("overreach_risk")
    if isinstance(risk, dict):
        lv = risk.get("level")
        sc2 = risk.get("score")
        out.append(
            {
                "signal_id": "forecast.overreach_risk.level",
                "value": lv,
                "unit": "",
                "source": "readiness_and_risk_forecast.json:result.overreach_risk.level",
                "date_range": date_range,
            }
        )
        out.append(
            {
                "signal_id": "forecast.overreach_risk.score",
                "value": sc2,
                "unit": "0-100",
                "source": "readiness_and_risk_forecast.json:result.overreach_risk.score",
                "date_range": date_range,
            }
        )

    return out


def _apply_deterministic_readiness_to_plan(
    plan_obj: dict[str, Any],
    det_forecast: Optional[dict[str, Any]],
) -> None:
    """
    Ensures training-plan output uses the deterministic readiness status.
    Schema only allows readiness.status + rationale + signal_ids; we update status and annotate rationale.
    """
    if not isinstance(det_forecast, dict):
        return
    res = det_forecast.get("result")
    if not isinstance(res, dict):
        return
    readiness = res.get("readiness")
    if not isinstance(readiness, dict):
        return

    status = readiness.get("status")
    score = readiness.get("score")
    if status not in ("primed", "steady", "fatigued"):
        return

    r = plan_obj.get("readiness")
    if not isinstance(r, dict):
        return

    r["status"] = status

    prefix = f"Deterministic readiness: {status}"
    if isinstance(score, (int, float)):
        prefix += f" (score {score})."
    else:
        prefix += "."

    old = r.get("rationale")
    if isinstance(old, str) and old.strip():
        if not old.strip().lower().startswith("deterministic readiness"):
            r["rationale"] = prefix + " " + old.strip()
    else:
        r["rationale"] = prefix

    dn = plan_obj.get("data_notes")
    if isinstance(dn, list):
        note = "Readiness status was set from deterministic readiness_and_risk_forecast.json."
        if note not in dn:
            dn.append(note)


def training_plan_to_text(obj: dict[str, Any]) -> str:
    """
    Convert a training-plan JSON object into a simple, human-readable text plan.
    Output is intended to be saved as a .txt file alongside the JSON.
    """
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    readiness = obj.get("readiness") if isinstance(obj.get("readiness"), dict) else {}
    plan = obj.get("plan") if isinstance(obj.get("plan"), dict) else {}
    weekly = plan.get("weekly_totals") if isinstance(plan.get("weekly_totals"), dict) else {}
    days = plan.get("days") if isinstance(plan.get("days"), list) else []

    def _fmt_weekday(ds: str) -> str:
        if not isinstance(ds, str):
            return ""
        try:
            return date.fromisoformat(ds).strftime("%a")
        except Exception:
            return ""

    def _safe_str(x: Any) -> str:
        return x.strip() if isinstance(x, str) else ""

    def _safe_num(x: Any) -> Optional[float]:
        return float(x) if isinstance(x, (int, float)) else None

    # Sort days by ISO date if possible
    def _day_key(d: dict[str, Any]) -> str:
        ds = d.get("date")
        return ds if isinstance(ds, str) else "9999-99-99"

    day_objs = [d for d in days if isinstance(d, dict)]
    day_objs.sort(key=_day_key)

    lines: list[str] = []
    lines.append("TrailTraining - Training Plan")
    lines.append("")

    # Meta
    plan_start = _safe_str(meta.get("plan_start"))
    plan_days = meta.get("plan_days")
    style = _safe_str(meta.get("style"))
    today = _safe_str(meta.get("today"))

    if today:
        lines.append(f"Generated: {today}")
    if plan_start or plan_days:
        lines.append(
            f"Plan start: {plan_start or '(unknown)'}   Days: {plan_days if plan_days is not None else '(unknown)'}"
        )
    if style:
        lines.append(f"Style: {style}")

    # Readiness
    status = _safe_str(readiness.get("status"))
    rationale = _safe_str(readiness.get("rationale"))
    if status or rationale:
        lines.append("")
        if status:
            lines.append(f"Readiness: {status}")
        if rationale:
            lines.append(f"Why: {rationale}")

    # Weekly totals (best-effort)
    dist_km = _safe_num(weekly.get("planned_distance_km"))
    hours = _safe_num(weekly.get("planned_moving_time_hours"))
    elev_m = _safe_num(weekly.get("planned_elevation_m"))
    if dist_km is not None or hours is not None or elev_m is not None:
        parts: list[str] = []
        if hours is not None:
            parts.append(f"{hours:.1f} h")
        if dist_km is not None and dist_km > 0:
            parts.append(f"{dist_km:.0f} km")
        if elev_m is not None and elev_m > 0:
            parts.append(f"{elev_m:.0f} m+")
        if parts:
            lines.append("")
            lines.append("Weekly totals: " + " • ".join(parts))

    # Day-by-day
    lines.append("")
    lines.append("Day-by-day")
    lines.append("-" * 10)

    for d in day_objs:
        ds = _safe_str(d.get("date"))
        wd = _fmt_weekday(ds)
        title = _safe_str(d.get("title")) or "(no title)"
        session_type = _safe_str(d.get("session_type"))
        is_rest = bool(d.get("is_rest_day"))
        is_hard = bool(d.get("is_hard_day"))
        mins = d.get("duration_minutes")
        dur = f"{mins} min" if isinstance(mins, (int, float)) else "?"

        tag_parts: list[str] = []
        if is_rest:
            tag_parts.append("REST")
        elif session_type:
            tag_parts.append(session_type.upper())
        if is_hard and not is_rest:
            tag_parts.append("HARD")
        tag = ", ".join(tag_parts) if tag_parts else "SESSION"

        date_label = f"{wd} {ds}".strip() if wd or ds else "(unknown date)"
        lines.append(f"{date_label}: {title} ({tag}, {dur})")

        target_intensity = _safe_str(d.get("target_intensity"))
        terrain = _safe_str(d.get("terrain"))
        workout = _safe_str(d.get("workout"))
        purpose = _safe_str(d.get("purpose"))

        if target_intensity:
            lines.append(f"  Intensity: {target_intensity}")
        if terrain:
            lines.append(f"  Terrain: {terrain}")
        if workout:
            lines.append(f"  Workout: {workout}")
        if purpose:
            lines.append(f"  Purpose: {purpose}")

        lines.append("")

    # Recovery actions
    rec = obj.get("recovery") if isinstance(obj.get("recovery"), dict) else {}
    actions = rec.get("actions") if isinstance(rec.get("actions"), list) else []
    actions = [a for a in actions if isinstance(a, str) and a.strip()]
    if actions:
        lines.append("Recovery focus")
        lines.append("-" * 14)
        for a in actions:
            lines.append(f"- {a.strip()}")
        lines.append("")

    # Risks
    risks = obj.get("risks") if isinstance(obj.get("risks"), list) else []
    risks = [r for r in risks if isinstance(r, dict)]
    if risks:
        lines.append("Risks / cautions")
        lines.append("-" * 15)
        for r in risks:
            sev = _safe_str(r.get("severity")) or "unknown"
            msg = _safe_str(r.get("message")) or "(no message)"
            lines.append(f"- [{sev}] {msg}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _call_responses_best_effort_schema(
    client: OpenAI, kwargs: dict[str, Any], schema: dict[str, Any]
) -> Any:
    """
    Best-effort "structured output" call:
    - Try text.format=json_schema
    - Try response_format=json_schema
    - Fallback: plain call
    We only swallow TypeError (unsupported kwarg).
    """
    # Attempt A: text.format
    try:
        kw = dict(kwargs)
        text_cfg = dict(kw.get("text") or {})
        text_cfg["format"] = {
            "type": "json_schema",
            "name": schema.get("name"),
            "schema": schema.get("schema"),
        }
        kw["text"] = text_cfg
        return client.responses.create(**kw)
    except TypeError:
        pass

    # Attempt B: response_format
    try:
        kw = dict(kwargs)
        kw["response_format"] = {
            "type": "json_schema",
            "name": schema.get("name"),
            "schema": schema.get("schema"),
        }
        return client.responses.create(**kw)
    except TypeError:
        pass

    return client.responses.create(**kwargs)


def _prompt_instruction(prompt_name: str, *, style: str) -> str:
    """
    Only style-switches:
      - system instructions (handled elsewhere)
      - training-plan task prompt (handled here via presets)
    Other prompts remain unchanged from prompts.py.
    """
    try:
        return get_task_prompt(prompt_name, style=style)
    except Exception:
        # Fallbacks (should rarely hit)
        if prompt_name == "training-plan":
            return "Generate a trail-running training plan for the next 7-14 days based on fatigue, recent volume, and sleep."
        if prompt_name == "recovery-status":
            return "Assess recovery status for the last 7 days and give actionable guidance for today and tomorrow."
        if prompt_name == "meal-plan":
            return "Suggest a practical meal plan for the next 3 days aligned with training load and recovery."
        return "Provide coaching guidance based on the provided data."


def _build_prompt_text(
    prompt_name: str,
    personal: Any,
    rollups: Optional[Any],
    combined: list[dict[str, Any]],
    deterministic_forecast: Optional[dict[str, Any]],
    *,
    style: str,
    max_chars: int,
    detail_days: int,
) -> str:
    """
    Budgeted assembly.
    We build from newest → oldest until we hit max_chars, then stop.
    """
    retrieval_weeks = int(os.getenv("TRAILTRAINING_COACH_RETRIEVAL_WEEKS", "8"))

    header = [
        f"# TrailTraining Coach Brief: {prompt_name}",
        "",
        "## Personal profile (raw JSON)",
        _safe_json_snippet(personal, max_chars=50_000),
        "",
    ]

    if rollups is not None:
        header += [
            "## Recent rollups (7d/28d)",
            _safe_json_snippet(rollups, max_chars=80_000),
            "",
        ]
    header += [
        "## Eval-coach constraints (MUST satisfy)",
        build_eval_constraints_block(rollups if isinstance(rollups, dict) else None),
        "",
    ]
    if deterministic_forecast is not None:
        header += [
            "## Deterministic readiness & overreach risk (authoritative)",
            _safe_json_snippet(deterministic_forecast, max_chars=40_000),
            "",
            "Guidance: Use the deterministic readiness.status for readiness.status in your output. "
            "You may cite forecast.* signals from the Signal registry, and/or the underlying load/recovery signals.",
            "",
        ]

    # Retrieval: last N weeks + citeable signal registry
    ctx = build_retrieval_context(
        combined,
        rollups if isinstance(rollups, dict) else None,
        retrieval_weeks=retrieval_weeks,
    )

    weekly_history = ctx.get("weekly_history")
    signal_registry = ctx.get("signal_registry")

    # Append forecast into signal registry so the model can cite it (forecast.*)
    if isinstance(signal_registry, list) and isinstance(deterministic_forecast, dict):
        signal_registry = list(signal_registry) + _forecast_signal_rows(deterministic_forecast)

    header += [
        f"## Retrieved history (weekly summaries; last {retrieval_weeks} weeks)",
        _safe_json_snippet(weekly_history, max_chars=50_000),
        "",
        "## Signal registry (you MUST cite signal_ids from here)",
        _safe_json_snippet(signal_registry, max_chars=80_000),
        "",
    ]

    # Limit daily detail further even if combined is large
    if detail_days > 0 and len(combined) > detail_days:
        combined_detail = combined[-detail_days:]
        combined_older = combined[:-detail_days]
    else:
        combined_detail = combined
        combined_older = []

    # We include only minimal reference to older days (optional)
    if combined_older:
        header += [
            f"## Older days included in window: {len(combined_older)} (details omitted; rely on rollups + recent detail)",
            "",
        ]

    base = "\n".join(header)
    budget = max_chars if max_chars > 0 else 200_000

    # Start with base; then add day blocks newest→oldest until budget is exhausted
    text_parts: list[str] = [base]
    used = len(base)

    # Add detailed days newest→oldest
    for day in reversed(combined_detail):
        block = _summarize_day(day)
        if used + len(block) > budget:
            break
        text_parts.append(block)
        used += len(block)

    # Add tail task instruction
    tail = "\n## Task\n" + _prompt_instruction(prompt_name, style=style) + "\n"
    if used + len(tail) <= budget:
        text_parts.append(tail)
        used += len(tail)

    # For training-plan, append strict output contract (works even without schema-mode)
    if prompt_name == "training-plan":
        contract = "\n## Output Contract (STRICT)\n" + training_plan_output_contract_text() + "\n"
        if used + len(contract) <= budget:
            text_parts.append(contract)

    return "\n".join(text_parts)


@dataclass(frozen=True)
class CoachConfig:
    # Safe constants as dataclass defaults (no function calls)
    model: str = "gpt-5.2"
    reasoning_effort: str = "medium"  # none|low|medium|high|xhigh
    verbosity: str = "medium"  # low|medium|high
    days: int = 60
    max_chars: int = 200_000
    temperature: Optional[float] = None
    style: str = "trailrunning"

    @classmethod
    def from_env(cls) -> CoachConfig:
        def _env_int(name: str, default: int) -> int:
            v = os.getenv(name)
            if v is None or not v.strip():
                return default
            try:
                return int(v)
            except ValueError:
                return default

        return cls(
            model=os.getenv("TRAILTRAINING_LLM_MODEL", cls.model),
            reasoning_effort=os.getenv("TRAILTRAINING_REASONING_EFFORT", cls.reasoning_effort),
            verbosity=os.getenv("TRAILTRAINING_VERBOSITY", cls.verbosity),
            days=_env_int("TRAILTRAINING_COACH_DAYS", cls.days),
            max_chars=_env_int("TRAILTRAINING_COACH_MAX_CHARS", cls.max_chars),
            style=os.getenv("TRAILTRAINING_COACH_STYLE", cls.style),
        )


def _recompute_planned_hours_from_days(obj: dict[str, Any]) -> None:
    """
    Make weekly_totals.planned_moving_time_hours consistent with sum(plan.days[].duration_minutes).
    This prevents false MAX_RAMP_PCT violations caused by rounding / inconsistent totals.
    """
    plan = obj.get("plan")
    if not isinstance(plan, dict):
        return
    days = plan.get("days")
    if not isinstance(days, list):
        return

    total_min = 0.0
    for d in days:
        if not isinstance(d, dict):
            continue
        m = d.get("duration_minutes")
        if isinstance(m, (int, float)):
            total_min += float(m)

    wt = plan.get("weekly_totals")
    if not isinstance(wt, dict):
        return

    wt["planned_moving_time_hours"] = round(total_min / 60.0, 1)


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

    # early pruning
    _dedup_activities_in_place(combined)
    combined = _filter_last_days(combined, cfg.days)

    base_dir = summary_p.parent
    deterministic_forecast = _load_or_compute_deterministic_forecast(base_dir, combined)

    detail_days = int(os.getenv("TRAILTRAINING_COACH_DETAIL_DAYS", "14"))
    detail_days = max(1, min(detail_days, len(combined))) if combined else 0

    prompt_text = _build_prompt_text(
        prompt_name=prompt,
        personal=personal,
        rollups=rollups,
        combined=combined,
        deterministic_forecast=deterministic_forecast,
        style=cfg.style,
        max_chars=cfg.max_chars,
        detail_days=detail_days,
    )

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("TRAILTRAINING_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI API key. Set OPENAI_API_KEY (recommended) or TRAILTRAINING_OPENAI_API_KEY.\n"
            "Example:\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "Then rerun: trailtraining coach --prompt training-plan"
        )

    client = OpenAI(api_key=api_key)

    # Use style-specific system instructions
    system_instructions = get_system_prompt(cfg.style)

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": system_instructions,
        "input": prompt_text,
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {"verbosity": cfg.verbosity},
    }
    # API restriction: temperature typically only allowed when reasoning.effort == "none"
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature

    # Training-plan: request structured JSON output (best-effort)
    if prompt == "training-plan":
        resp = _call_responses_best_effort_schema(client, kwargs, TRAINING_PLAN_SCHEMA)
    else:
        resp = client.responses.create(**kwargs)

    out_text = getattr(resp, "output_text", None) or str(resp)

    # If training-plan: parse + validate + save pretty JSON
    if prompt == "training-plan":
        raw = _extract_json_object(out_text)
        try:
            obj = json.loads(raw)
            obj = ensure_training_plan_shape(obj)
            _recompute_planned_hours_from_days(obj)
        except Exception as e:
            log.warning("Training-plan JSON parse/shape failed; attempting one repair pass: %s", e)
            repair_prompt = (
                "Return ONLY valid JSON (no markdown, no backticks) matching this schema:\n"
                f"{TRAINING_PLAN_SCHEMA.get('schema')}\n\n"
                "Your previous output was invalid. Fix it. Here is the invalid output:\n"
                f"{out_text}\n"
            )
            repair_kwargs: dict[str, Any] = {
                "model": cfg.model,
                "instructions": system_instructions,
                "input": repair_prompt,
                "reasoning": {"effort": "none"},
                "text": {"verbosity": "low"},
            }
            repair_resp = client.responses.create(**repair_kwargs)
            repaired = getattr(repair_resp, "output_text", None) or str(repair_resp)
            raw2 = _extract_json_object(repaired)
            obj = ensure_training_plan_shape(json.loads(raw2))
            _recompute_planned_hours_from_days(obj)

        # Force deterministic readiness into the final plan object
        _apply_deterministic_readiness_to_plan(obj, deterministic_forecast)
        apply_eval_coach_guardrails(obj, rollups if isinstance(rollups, dict) else None)

        # Save JSON output
        if output_path:
            out_p = Path(output_path).expanduser().resolve()
        else:
            out_p = Path(config.PROMPTING_DIRECTORY) / f"coach_brief_{prompt}.json"

        out_p.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_p, obj, compact=False)

        # Also write a human-readable interpretation (.txt) next to the JSON
        try:
            txt_p = out_p.parent / f"{out_p.stem}.txt"
            txt_p.write_text(training_plan_to_text(obj), encoding="utf-8")
        except Exception as e:
            log.warning("Failed to write training-plan text interpretation: %s", e)

        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        return pretty, str(out_p)

    # Non-training-plan: save markdown text
    if output_path:
        out_p = Path(output_path).expanduser().resolve()
    else:
        out_p = Path(config.PROMPTING_DIRECTORY) / f"coach_brief_{prompt}.md"

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(out_text, encoding="utf-8")

    return out_text, str(out_p)
