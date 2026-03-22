from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from trailtraining import config
from trailtraining.contracts import EvaluationReportArtifact, TrainingPlanArtifact
from trailtraining.llm.constraints import _extract_effective_constraints
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
    recompute_weekly_totals,
    training_plan_to_text,
)
from trailtraining.llm.soft_eval import SoftEvalConfig, compare_plans
from trailtraining.util.state import load_json, save_json
from trailtraining.util.text import _safe_json_snippet

_make_openrouter_client = make_openrouter_client
_call_with_schema = call_with_schema

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RevisePlanConfig:
    model: str = "openai/gpt-4o"
    reasoning_effort: str = "medium"
    verbosity: str = "medium"
    temperature: Optional[float] = None
    primary_goal: Optional[str] = None
    lifestyle_notes: str = ""

    @classmethod
    def from_env(cls) -> RevisePlanConfig:
        return cls(
            model=os.getenv("TRAILTRAINING_LLM_MODEL", cls.model),
            reasoning_effort=os.getenv("TRAILTRAINING_REASONING_EFFORT", cls.reasoning_effort),
            verbosity=os.getenv("TRAILTRAINING_VERBOSITY", cls.verbosity),
            primary_goal=os.getenv("TRAILTRAINING_PRIMARY_GOAL") or None,
            lifestyle_notes=os.getenv("TRAILTRAINING_LIFESTYLE_NOTES", "").strip(),
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


def _lifestyle_notes_for_revise(lifestyle_notes: str) -> list[str]:
    notes = lifestyle_notes.strip() if isinstance(lifestyle_notes, str) else ""
    if not notes:
        return []
    return [
        "## Lifestyle constraints (MUST respect in revised plan)",
        f"The athlete has these schedule constraints: {notes}",
        "The revised plan must continue to respect these constraints.",
        "Do not move sessions to days that violate these constraints.",
        "",
    ]


def _build_revise_prompt(
    plan_obj: dict[str, Any],
    report_obj: dict[str, Any],
    *,
    style: str,
    primary_goal: str,
    lifestyle_notes: str,
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
            "Treat structured fields as authoritative: title/workout text MUST agree with session_type, is_rest_day, is_hard_day, duration_minutes, and terrain.",
            "If the original plan was stronger overall, keep its strong ideas but still return a normalized revised artifact that fixes schema and consistency errors.",
            "Do not invent new telemetry, sources, or citations.",
            f"Style: {style}",
            f"Primary goal: {primary_goal}",
            f"Plan duration: {plan_days} days",
            "",
            "## Non-negotiable rules",
            "- Return JSON only.",
            "- The output must match the training-plan schema exactly.",
            f"- The revised plan MUST contain exactly {plan_days} days in plan.days.",
            "- Preserve meta.plan_days and the overall date range unless evaluator feedback clearly requires a change.",
            "- Keep the plan grounded in the original artifact's signals and citations.",
            "- Do not invent new signal_ids unless they already exist in the original plan/citations.",
            "- weekly_totals must be derived from WEEK 1 day objects only.",
            "- planned_distance_km and planned_elevation_m must be null unless supported by complete per-day estimates.",
            "- If the evaluator identified concerns or improvements, address them concretely.",
            "- Preserve strong sessions and useful explanations when they are already good.",
            "- Preserve meta.lifestyle_notes from the original plan.",
            "- Do not simply re-emit the original unchanged when the report asks for fixes.",
            "",
            *_lifestyle_notes_for_revise(lifestyle_notes),
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


def _apply_lifestyle_notes(plan_obj: dict[str, Any], lifestyle_notes: str) -> None:
    notes = lifestyle_notes.strip() if isinstance(lifestyle_notes, str) else ""
    meta = plan_obj.get("meta")
    if isinstance(meta, dict):
        meta["lifestyle_notes"] = notes
        plan_obj["meta"] = meta


def _apply_guardrails_compat(
    plan_obj: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    effective: Any = None,
) -> None:
    try:
        apply_eval_coach_guardrails(plan_obj, rollups, effective=effective)
    except TypeError as exc:
        if "unexpected keyword argument 'effective'" not in str(exc):
            raise
        apply_eval_coach_guardrails(plan_obj, rollups)


def _pairwise_cfg_for_revision(
    report_obj: dict[str, Any],
    cfg: RevisePlanConfig,
    *,
    primary_goal: str,
    lifestyle_notes: str,
) -> SoftEvalConfig:
    soft = report_obj.get("soft_assessment") or {}
    judge_model = (
        str((soft or {}).get("model", "") or "").strip()
        or os.getenv("TRAILTRAINING_SOFT_EVAL_MODEL", "").strip()
        or cfg.model
    )
    judge_reasoning = (
        os.getenv("TRAILTRAINING_SOFT_EVAL_REASONING_EFFORT", "").strip() or cfg.reasoning_effort
    )
    judge_verbosity = os.getenv("TRAILTRAINING_SOFT_EVAL_VERBOSITY", "").strip() or cfg.verbosity
    return SoftEvalConfig(
        enabled=True,
        model=judge_model,
        reasoning_effort=judge_reasoning,
        verbosity=judge_verbosity,
        primary_goal=primary_goal,
        lifestyle_notes=lifestyle_notes,
    )


def _report_requests_change(report_obj: dict[str, Any]) -> bool:
    violations = report_obj.get("violations") or []
    if isinstance(violations, list) and violations:
        return True
    soft = report_obj.get("soft_assessment") or {}
    if isinstance(soft, dict):
        for key in ("concerns", "suggested_improvements"):
            value = soft.get(key) or []
            if isinstance(value, list) and any(str(item or "").strip() for item in value):
                return True
    return False


def _compare_revised_candidate(
    original_plan: dict[str, Any],
    revised_candidate: dict[str, Any],
    *,
    rollups: Optional[dict[str, Any]],
    report_obj: dict[str, Any],
    cfg: RevisePlanConfig,
    primary_goal: str,
    lifestyle_notes: str,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    original_json = json.dumps(original_plan, sort_keys=True, default=str)
    candidate_json = json.dumps(revised_candidate, sort_keys=True, default=str)
    if original_json == candidate_json:
        return revised_candidate, {
            "preferred": "tie",
            "reasoning": "Original plan and revised candidate were identical after normalization.",
            "plan_a_advantages": [],
            "plan_b_advantages": [],
            "selected_plan": "revised_candidate",
        }

    pairwise_cfg = _pairwise_cfg_for_revision(
        report_obj,
        cfg,
        primary_goal=primary_goal,
        lifestyle_notes=lifestyle_notes,
    )

    try:
        comparison = compare_plans(
            original_plan,
            revised_candidate,
            rollups=rollups,
            cfg=pairwise_cfg,
        )
    except Exception as exc:
        log.warning("Pairwise comparison failed; keeping revised candidate only: %s", exc)
        return None, None

    preferred = str(comparison.get("preferred", "tie") or "tie").strip().lower()
    selected_plan = revised_candidate
    selected_label = "revised_candidate"
    if preferred == "plan_a":
        selected_plan = original_plan
        selected_label = "original"

    comparison_payload = {
        "judge_model": pairwise_cfg.model,
        "preferred": preferred,
        "reasoning": str(comparison.get("reasoning", "") or "").strip(),
        "plan_a_advantages": comparison.get("plan_a_advantages", []),
        "plan_b_advantages": comparison.get("plan_b_advantages", []),
        "selected_plan": selected_label,
        "revised_artifact_contract": "revised-plan.json is always the normalized revised artifact; selected-plan.json stores the pairwise winner.",
    }
    return selected_plan, comparison_payload


def _write_selected_plan_artifacts(out_p: Path, selected_plan: dict[str, Any]) -> None:
    selected_json = out_p.parent / "selected-plan.json"
    selected_txt = out_p.parent / "selected-plan.txt"
    save_json(selected_json, selected_plan, compact=False)
    selected_txt.write_text(training_plan_to_text(selected_plan), encoding="utf-8")


def run_revise_plan(
    *,
    cfg: RevisePlanConfig,
    input_plan_path: str,
    eval_report_path: str,
    output_path: Optional[str] = None,
    rollups_path: Optional[str] = None,
    runtime: Optional[config.RuntimeConfig] = None,
    auto_reeval: bool = False,
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
    lifestyle_notes = (
        str(cfg.lifestyle_notes or "").strip()
        or str((plan_obj.get("meta") or {}).get("lifestyle_notes") or "").strip()
    )

    client = _make_openrouter_client()
    prompt_text = _build_revise_prompt(
        plan_obj,
        report_obj,
        style=style,
        primary_goal=primary_goal,
        lifestyle_notes=lifestyle_notes,
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

    resp = _call_with_schema(client, kwargs, TRAINING_PLAN_SCHEMA)
    out_text = getattr(resp, "output_text", None) or str(resp)

    used_repair_fallback = False
    try:
        raw_candidate = json.loads(extract_json_object(out_text))
        obj = ensure_training_plan_shape(raw_candidate)
    except Exception as exc:
        used_repair_fallback = True
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
        repair_resp = _call_with_schema(client, repair_kwargs, TRAINING_PLAN_SCHEMA)
        repaired = getattr(repair_resp, "output_text", None) or str(repair_resp)
        raw_candidate = json.loads(extract_json_object(repaired))
        obj = ensure_training_plan_shape(raw_candidate)

    # Check for a true authored revision BEFORE deterministic guardrails/normalization.
    # But do not fail the repair-fallback path if the repaired artifact normalizes back
    # to the original; that path is meant to recover invalid JSON, not prove a semantic rewrite.
    original_basis_json = json.dumps(plan_obj, sort_keys=True, default=str)
    candidate_basis_json = json.dumps(obj, sort_keys=True, default=str)
    if (
        not used_repair_fallback
        and original_basis_json == candidate_basis_json
        and _report_requests_change(report_obj)
    ):
        raise RuntimeError(
            "Revision produced no material change despite requested fixes. "
            "Refusing to save an unchanged revised-plan artifact."
        )

    apply_primary_goal(obj, primary_goal)
    _apply_lifestyle_notes(obj, lifestyle_notes)

    rollups = _load_rollups_near(plan_p, rollups_path)
    effective = _extract_effective_constraints(plan_obj) or _extract_effective_constraints(obj)
    _apply_guardrails_compat(obj, rollups if isinstance(rollups, dict) else None, effective)
    recompute_weekly_totals(obj)

    revised_candidate = TrainingPlanArtifact.model_validate(obj).model_dump(mode="json")

    selected_plan, comparison_payload = _compare_revised_candidate(
        plan_obj,
        revised_candidate,
        rollups=rollups,
        report_obj=report_obj,
        cfg=cfg,
        primary_goal=primary_goal,
        lifestyle_notes=lifestyle_notes,
    )

    final_obj = TrainingPlanArtifact.model_validate(revised_candidate).model_dump(mode="json")
    out_p = (
        Path(output_path).expanduser().resolve()
        if output_path
        else paths.prompting_directory / "revised-plan.json"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_p, final_obj, compact=False)

    txt_p = out_p.parent / f"{out_p.stem}.txt"
    txt_p.write_text(training_plan_to_text(final_obj), encoding="utf-8")

    if selected_plan is not None:
        _write_selected_plan_artifacts(out_p, selected_plan)

    if comparison_payload:
        comparison_path = out_p.parent / f"{out_p.stem}-comparison.json"
        try:
            save_json(comparison_path, comparison_payload, compact=False)
        except Exception as exc:
            log.warning("Could not write pairwise comparison file: %s", exc)

    if auto_reeval:
        _run_auto_reeval(
            revised_plan_path=out_p,
            original_report_obj=report_obj,
            rollups_path=rollups_path,
        )

    pretty = json.dumps(final_obj, indent=2, ensure_ascii=False)
    return pretty, str(out_p)


def _run_auto_reeval(
    revised_plan_path: Path,
    original_report_obj: dict[str, Any],
    rollups_path: Optional[str],
) -> None:
    from trailtraining.llm.eval import evaluate_training_plan_quality_file

    original_score = float(original_report_obj.get("score", 0))

    try:
        revised_report, _ = evaluate_training_plan_quality_file(
            str(revised_plan_path),
            rollups_path=rollups_path,
        )
    except Exception as exc:
        log.warning("Auto-reeval failed: %s", exc)
        return

    revised_score = float(revised_report.get("score", 0))
    delta = revised_score - original_score

    reeval_data: dict[str, Any] = {
        "original_score": original_score,
        "revised_score": revised_score,
        "delta_score": round(delta, 1),
        "violations": revised_report.get("violations", []),
        "grade": revised_report.get("grade", "?"),
        "blocking_issues": revised_report.get("blocking_issues", []),
        "score_components": revised_report.get("score_components", {}),
    }

    reeval_path = revised_plan_path.parent / f"{revised_plan_path.stem}-reeval.json"
    try:
        save_json(reeval_path, reeval_data, compact=False)
        print(f"[Saved] {reeval_path}")
    except Exception as exc:
        log.warning("Could not write reeval file: %s", exc)

    if delta < 0:
        msg = (
            f"Revision degraded deterministic score: "
            f"{original_score:.0f} -> {revised_score:.0f} ({delta:+.1f}). "
            f"See {reeval_path.name} for violations."
        )
        log.warning(msg)
        print(msg)
    else:
        print(
            f"Auto-reeval: deterministic score "
            f"{original_score:.0f} -> {revised_score:.0f} ({delta:+.1f})."
        )
