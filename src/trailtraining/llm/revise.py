from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from trailtraining import config
from trailtraining.contracts import EvaluationReportArtifact, TrainingPlanArtifact
from trailtraining.llm.eval import _load_rollups_near
from trailtraining.llm.guardrails import apply_eval_coach_guardrails
from trailtraining.llm.presets import _multiweek_addendum
from trailtraining.llm.rubrics import _normalize_style, default_primary_goal_for_style
from trailtraining.llm.schemas import TRAINING_PLAN_SCHEMA, ensure_training_plan_shape
from trailtraining.llm.shared import (
    apply_primary_goal,
    call_with_schema,
    extract_json_object,
    make_openrouter_client,
    race_context_section,
    recompute_planned_hours,
    training_plan_to_text,
)
from trailtraining.util.state import load_json, save_json
from trailtraining.util.text import _safe_json_snippet

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RevisePlanConfig:
    model: str = "openai/gpt-4o"
    reasoning_effort: str = "medium"
    verbosity: str = "medium"
    temperature: Optional[float] = None
    primary_goal: Optional[str] = None

    @classmethod
    def from_env(cls) -> RevisePlanConfig:
        return cls(
            model=os.getenv("TRAILTRAINING_LLM_MODEL", cls.model),
            reasoning_effort=os.getenv("TRAILTRAINING_REASONING_EFFORT", cls.reasoning_effort),
            verbosity=os.getenv("TRAILTRAINING_VERBOSITY", cls.verbosity),
            primary_goal=os.getenv("TRAILTRAINING_PRIMARY_GOAL") or None,
        )


def _summarize_eval_targets(report_obj: dict[str, Any]) -> list[str]:
    violations = report_obj.get("violations") or []
    soft = report_obj.get("soft_assessment") or {}

    lines: list[str] = []

    if isinstance(violations, list) and violations:
        lines.append("## Deterministic issues to fix")
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            severity = str(violation.get("severity", "") or "").strip() or "unknown"
            code = str(violation.get("code", "") or "").strip() or "UNKNOWN"
            msg = str(violation.get("message", "") or "").strip()
            lines.append(f"- [{severity}] {code}: {msg}")
        lines.append("")

    if isinstance(soft, dict) and soft:
        summary = str(soft.get("summary", "") or "").strip()
        if summary:
            lines.append("## Soft assessment summary")
            lines.append(summary)
            lines.append("")

        strengths = soft.get("strengths") or []
        if isinstance(strengths, list) and strengths:
            lines.append("## Strengths to preserve")
            for item in strengths:
                s = str(item or "").strip()
                if s:
                    lines.append(f"- {s}")
            lines.append("")

        concerns = soft.get("concerns") or []
        if isinstance(concerns, list) and concerns:
            lines.append("## Concerns to address")
            for item in concerns:
                s = str(item or "").strip()
                if s:
                    lines.append(f"- {s}")
            lines.append("")

        improvements = soft.get("suggested_improvements") or []
        if isinstance(improvements, list) and improvements:
            lines.append("## Suggested improvements to implement")
            for item in improvements:
                s = str(item or "").strip()
                if s:
                    lines.append(f"- {s}")
            lines.append("")

    if not lines:
        lines.extend(
            [
                "## Revision target",
                "- Improve clarity, specificity, coherence, and actionability.",
                "- Preserve the best parts of the existing plan.",
                "",
            ]
        )

    return lines


def _build_revise_prompt(
    plan_obj: dict[str, Any],
    report_obj: dict[str, Any],
    *,
    style: str,
    primary_goal: str,
) -> str:
    plan_days: int = int((plan_obj.get("meta") or {}).get("plan_days") or 7)

    multiweek_note: list[str] = []
    if plan_days > 7:
        multiweek_note = [
            "## Multi-week plan rules (plan_days > 7)",
            f"- The revised plan MUST contain exactly {plan_days} days in plan.days.",
            "- Preserve the phased structure (build -> build -> peak -> recovery) across the weeks.",
            "- weekly_totals MUST reflect WEEK 1 values only, not the full-period total.",
            *_multiweek_addendum(plan_days).strip().splitlines(),
            "",
        ]
        multiweek_note += race_context_section(primary_goal)

    return "\n".join(
        [
            "You are revising an existing endurance training plan using evaluator feedback.",
            "Return a FULL revised training-plan JSON artifact, not a diff.",
            "Fix clear problems, preserve strong parts, and avoid unnecessary rewrites.",
            "Do not invent new telemetry, sources, or citations.",
            f"Style: {style}",
            f"Primary goal: {primary_goal}",
            f"Plan duration: {plan_days} days",
            "",
            "## Non-negotiable rules",
            "- Return JSON only.",
            "- The output must match the training-plan schema exactly.",
            f"- The revised plan MUST contain exactly {plan_days} days in plan.days — do NOT reduce the day count.",
            "- Preserve meta.plan_days and the overall date range unless evaluator feedback clearly requires a change.",
            "- Keep the plan grounded in the original artifact's signals and citations.",
            "- Do not invent new signal_ids unless they already exist in the original plan/citations.",
            "- weekly_totals must match week 1 of the revised day list.",
            "- If the evaluator identified concerns or improvements, address them concretely.",
            "- Preserve strong sessions and useful explanations when they are already good.",
            "",
            *multiweek_note,
            *_summarize_eval_targets(report_obj),
            "## Original training plan JSON",
            _safe_json_snippet(plan_obj, max_chars=80_000),
            "",
            "## Eval report JSON",
            _safe_json_snippet(report_obj, max_chars=80_000),
            "",
            "## Output requirement",
            f"Return the complete revised training-plan JSON artifact with all {plan_days} days.",
        ]
    )


def run_revise_plan(
    *,
    cfg: RevisePlanConfig,
    input_plan_path: str,
    eval_report_path: str,
    output_path: Optional[str] = None,
    rollups_path: Optional[str] = None,
    runtime: Optional[config.RuntimeConfig] = None,
) -> tuple[str, str]:
    runtime = runtime or config.current()
    config.ensure_directories(runtime)
    paths = runtime.paths

    plan_p = Path(input_plan_path).expanduser().resolve()
    report_p = Path(eval_report_path).expanduser().resolve()

    raw_plan = load_json(plan_p, default=None)
    raw_report = load_json(report_p, default=None)

    plan_obj = TrainingPlanArtifact.model_validate(raw_plan).model_dump(mode="python")
    report_obj = EvaluationReportArtifact.model_validate(raw_report).model_dump(mode="python")

    style = _normalize_style((plan_obj.get("meta") or {}).get("style"))
    primary_goal = (
        str(cfg.primary_goal or "").strip()
        or str((plan_obj.get("meta") or {}).get("primary_goal") or "").strip()
        or default_primary_goal_for_style(style)
    )

    client = make_openrouter_client()
    prompt_text = _build_revise_prompt(
        plan_obj,
        report_obj,
        style=style,
        primary_goal=primary_goal,
    )

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": (
            "You are a strict but constructive training-plan reviser. "
            "Revise the plan using evaluator feedback while preserving what already works."
        ),
        "input": prompt_text,
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {"verbosity": cfg.verbosity},
    }
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature

    resp = call_with_schema(client, kwargs, TRAINING_PLAN_SCHEMA)
    out_text = getattr(resp, "output_text", None) or str(resp)

    try:
        obj = ensure_training_plan_shape(json.loads(extract_json_object(out_text)))
        recompute_planned_hours(obj)
    except Exception as exc:
        log.warning("Revised-plan JSON parse/shape failed; attempting one repair pass: %s", exc)
        repair_prompt = (
            "Return ONLY valid JSON (no markdown, no backticks) matching this schema:\n"
            f"{TRAINING_PLAN_SCHEMA.get('schema')}\n\n"
            "Your previous output was invalid. Fix it.\n\n"
            f"Previous output:\n{out_text}\n"
        )
        repair_kwargs: dict[str, Any] = {
            "model": cfg.model,
            "instructions": "Return only valid JSON. No markdown.",
            "input": repair_prompt,
            "reasoning": {"effort": "none"},
            "text": {"verbosity": "low"},
        }
        repair_resp = call_with_schema(client, repair_kwargs, TRAINING_PLAN_SCHEMA)
        repaired = getattr(repair_resp, "output_text", None) or str(repair_resp)
        obj = ensure_training_plan_shape(json.loads(extract_json_object(repaired)))
        recompute_planned_hours(obj)

    apply_primary_goal(obj, primary_goal)

    rollups = _load_rollups_near(plan_p, rollups_path)
    if isinstance(rollups, dict):
        apply_eval_coach_guardrails(obj, rollups)

    final_obj = TrainingPlanArtifact.model_validate(obj).model_dump(mode="json")

    out_p = (
        Path(output_path).expanduser().resolve()
        if output_path
        else paths.prompting_directory / "revised-plan.json"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_p, final_obj, compact=False)

    txt_p = out_p.parent / f"{out_p.stem}.txt"
    txt_p.write_text(training_plan_to_text(final_obj), encoding="utf-8")

    pretty = json.dumps(final_obj, indent=2, ensure_ascii=False)
    return pretty, str(out_p)
