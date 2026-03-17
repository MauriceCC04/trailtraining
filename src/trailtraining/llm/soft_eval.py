from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from trailtraining.contracts import SoftAssessmentArtifact
from trailtraining.llm.rubrics import (
    DEFAULT_PRIMARY_GOAL,
    _normalize_style,
    default_primary_goal_for_style,
    get_default_rubrics,
    grade_from_score,
    render_rubrics_for_prompt,
    weighted_score_from_rubric_scores,
)
from trailtraining.llm.shared import call_with_schema, extract_json_object, make_openrouter_client
from trailtraining.util.text import _safe_json_snippet

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoftEvalConfig:
    enabled: bool = False
    model: str = "anthropic/claude-sonnet-4"
    reasoning_effort: str = "medium"
    verbosity: str = "medium"
    temperature: Optional[float] = None
    primary_goal: Optional[str] = None

    @classmethod
    def from_env(cls) -> SoftEvalConfig:
        return cls(
            enabled=False,
            model=os.getenv("TRAILTRAINING_SOFT_EVAL_MODEL", cls.model),
            reasoning_effort=os.getenv(
                "TRAILTRAINING_SOFT_EVAL_REASONING_EFFORT",
                cls.reasoning_effort,
            ),
            verbosity=os.getenv(
                "TRAILTRAINING_SOFT_EVAL_VERBOSITY",
                cls.verbosity,
            ),
            primary_goal=os.getenv("TRAILTRAINING_PRIMARY_GOAL") or None,
        )


def _rubric_ids() -> list[str]:
    return [r.rubric_id for r in get_default_rubrics("trailrunning")]


SOFT_EVAL_SCHEMA: dict[str, Any] = {
    "name": "trailtraining_soft_quality_assessment_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary",
            "confidence",
            "rubric_scores",
            "marker_results",
            "strengths",
            "concerns",
            "suggested_improvements",
        ],
        "properties": {
            "summary": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "rubric_scores": {
                "type": "object",
                "additionalProperties": False,
                "required": _rubric_ids(),
                "properties": {
                    rubric_id: {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["score", "reasoning"],
                        "properties": {
                            "score": {"type": "number", "minimum": 0, "maximum": 100},
                            "reasoning": {"type": "string"},
                        },
                    }
                    for rubric_id in _rubric_ids()
                },
            },
            "marker_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "rubric",
                        "marker_id",
                        "marker",
                        "verdict",
                        "score",
                        "evidence",
                        "improvement_hint",
                    ],
                    "properties": {
                        "rubric": {"type": "string"},
                        "marker_id": {"type": "string"},
                        "marker": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["pass", "partial", "fail"],
                        },
                        "score": {"type": "number", "minimum": 0, "maximum": 5},
                        "evidence": {"type": "string"},
                        "improvement_hint": {"type": "string"},
                    },
                },
            },
            "strengths": {"type": "array", "items": {"type": "string"}},
            "concerns": {"type": "array", "items": {"type": "string"}},
            "suggested_improvements": {"type": "array", "items": {"type": "string"}},
        },
    },
}


def _marker_only_schema() -> dict[str, Any]:
    return {
        "name": "trailtraining_soft_quality_markers_v1",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["marker_results"],
            "properties": {
                "marker_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "rubric",
                            "marker_id",
                            "marker",
                            "verdict",
                            "score",
                            "evidence",
                            "improvement_hint",
                        ],
                        "properties": {
                            "rubric": {"type": "string"},
                            "marker_id": {"type": "string"},
                            "marker": {"type": "string"},
                            "verdict": {
                                "type": "string",
                                "enum": ["pass", "partial", "fail"],
                            },
                            "score": {"type": "number", "minimum": 0, "maximum": 5},
                            "evidence": {"type": "string"},
                            "improvement_hint": {"type": "string"},
                        },
                    },
                }
            },
        },
    }


def _normalize_confidence(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in {"low", "medium", "high"}:
        return s
    return "medium"


def _normalize_verdict(value: Any, score: float) -> str:
    s = str(value or "").strip().lower()
    if s in {"pass", "partial", "fail"}:
        return s
    if score >= 4.0:
        return "pass"
    if score >= 2.0:
        return "partial"
    return "fail"


def _clean_string_list(raw: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for item in raw or []:
        s = " ".join(str(item).split()).strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)

    return out


def _build_feedback_lists(
    raw: dict[str, Any],
    rubric_scores: dict[str, dict[str, Any]],
    marker_results: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    strengths = _clean_string_list(raw.get("strengths"))
    concerns = _clean_string_list(raw.get("concerns"))
    improvements = _clean_string_list(raw.get("suggested_improvements"))
    derived_fields: list[str] = []

    if len(strengths) < 2:
        derived_fields.append("strengths")
        for _, rubric_score in sorted(
            rubric_scores.items(),
            key=lambda kv: float((kv[1] or {}).get("score", 0)),
            reverse=True,
        ):
            reasoning = str((rubric_score or {}).get("reasoning", "")).strip()
            if reasoning and reasoning not in strengths:
                strengths.append(reasoning)
            if len(strengths) >= 3:
                break

    weaker_markers = sorted(
        marker_results,
        key=lambda marker: float(marker.get("score", 0)),
    )

    if len(concerns) < 1:
        derived_fields.append("concerns")
        for marker in weaker_markers:
            evidence = str(marker.get("evidence", "")).strip()
            label = str(marker.get("marker", "")).strip()
            if evidence:
                text = f"{label}: {evidence}" if label else evidence
                if text not in concerns:
                    concerns.append(text)
            if len(concerns) >= 2:
                break

    if len(improvements) < 2:
        derived_fields.append("suggested_improvements")
        for marker in weaker_markers:
            hint = str(marker.get("improvement_hint", "")).strip()
            if hint and hint not in improvements:
                improvements.append(hint)
            if len(improvements) >= 3:
                break

    if len(strengths) < 2:
        for fallback_strength in [
            "The plan contains concrete and executable session structure.",
            "The week shows a clear training purpose across sessions.",
        ]:
            if fallback_strength not in strengths:
                strengths.append(fallback_strength)
            if len(strengths) >= 2:
                break

    if len(concerns) < 1:
        concerns = ["Some parts of the plan could be more specific or better justified."]

    if len(improvements) < 2:
        for fallback_improvement in [
            "Add more specific execution guidance for key sessions.",
            "Clarify progression and recovery logic where useful.",
        ]:
            if fallback_improvement not in improvements:
                improvements.append(fallback_improvement)
            if len(improvements) >= 2:
                break

    return strengths[:4], concerns[:3], improvements[:4], sorted(set(derived_fields))


def _expected_markers(style: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for rubric in get_default_rubrics(style):
        for marker in rubric.markers:
            out.append(
                {
                    "rubric": rubric.rubric_id,
                    "marker_id": marker.marker_id,
                    "marker": marker.label,
                }
            )
    return out


def _normalize_marker_results(raw: list[Any], *, style: str) -> list[dict[str, Any]]:
    rubrics = get_default_rubrics(style)
    by_marker: dict[str, dict[str, Any]] = {}
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("marker_id"), str):
            by_marker[item["marker_id"]] = item

    out: list[dict[str, Any]] = []
    for rubric in rubrics:
        for marker in rubric.markers:
            item = by_marker.get(marker.marker_id, {})
            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(5.0, score))
            verdict = _normalize_verdict(item.get("verdict"), score)
            out.append(
                {
                    "rubric": rubric.rubric_id,
                    "marker_id": marker.marker_id,
                    "marker": marker.label,
                    "verdict": verdict,
                    "score": round(score, 1),
                    "evidence": str(item.get("evidence", "") or "").strip(),
                    "improvement_hint": str(item.get("improvement_hint", "") or "").strip(),
                }
            )
    return out


def _normalize_rubric_scores(raw: Any, *, style: str) -> dict[str, dict[str, Any]]:
    rubrics = get_default_rubrics(style)
    raw_dict = raw if isinstance(raw, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for rubric in rubrics:
        item = raw_dict.get(rubric.rubric_id, {})
        if not isinstance(item, dict):
            item = {}
        try:
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(100.0, score))
        out[rubric.rubric_id] = {
            "score": round(score, 1),
            "reasoning": str(item.get("reasoning", "") or "").strip(),
        }
    return out


def _resolve_style_and_goal(
    plan_obj: dict[str, Any],
    cfg: SoftEvalConfig,
) -> tuple[str, str]:
    meta = plan_obj.get("meta") or {}
    style = _normalize_style(meta.get("style"))
    primary_goal = (
        str(cfg.primary_goal or "").strip()
        or str(meta.get("primary_goal") or "").strip()
        or default_primary_goal_for_style(style)
        or DEFAULT_PRIMARY_GOAL
    )
    return style, primary_goal


def _build_soft_eval_prompt(
    plan_obj: dict[str, Any],
    deterministic_report: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
) -> str:
    plan_days: int = int((plan_obj.get("meta") or {}).get("plan_days") or 7)
    plan_weeks = plan_days // 7
    duration_note = (
        f"{plan_days}-day ({plan_weeks}-week) multi-week plan"
        if plan_days > 7
        else f"{plan_days}-day plan"
    )
    return "\n".join(
        [
            "You are the second-stage quality assessor for a generated endurance training plan.",
            "Judge plan quality using the provided rubrics and markers.",
            "Do not rewrite the plan. Do not invent missing data. Be concrete and non-generic.",
            f"Evaluate this as a {style} plan and apply sport-specific standards for that style.",
            "",
            "## Evaluation context",
            f"Style: {style}",
            f"Primary goal: {primary_goal}",
            f"Plan duration: {duration_note}",
            *(
                [
                    f"This is a {plan_weeks}-week plan — evaluate periodization structure and weekly progression across all {plan_weeks} weeks, not just the first.",
                    "Hard-day and rest-day constraints apply per rolling 7-day window throughout all weeks.",
                ]
                if plan_days > 7
                else []
            ),
            "",
            "## Rubrics and markers",
            render_rubrics_for_prompt(style=style, primary_goal=primary_goal),
            "",
            "## Deterministic evaluation report",
            _safe_json_snippet(deterministic_report, max_chars=25_000),
            "",
            "## Rollups context",
            _safe_json_snippet(rollups or {}, max_chars=15_000),
            "",
            "## Training plan JSON",
            _safe_json_snippet(plan_obj, max_chars=50_000),
            "",
            "## Output rules",
            "- Return JSON only.",
            "- Fill every rubric in rubric_scores.",
            "- Include marker_results for every supplied marker.",
            "- Do not leave summary blank.",
            "- Use concrete evidence from the plan.",
            "- strengths: provide 2 to 4 concrete strengths grounded in the plan.",
            "- concerns: provide at least 1 concrete concern, even if minor.",
            "- suggested_improvements: provide 2 to 4 specific improvements tied to concerns or lower-scored markers.",
            "- Avoid empty arrays unless absolutely unavoidable.",
        ]
    )


def _build_marker_only_prompt(
    plan_obj: dict[str, Any],
    deterministic_report: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
) -> str:
    plan_days: int = int((plan_obj.get("meta") or {}).get("plan_days") or 7)
    return "\n".join(
        [
            "Return JSON only.",
            "Your only job is to produce marker_results for every expected marker.",
            "Do not omit markers.",
            "",
            f"Style: {style}",
            f"Primary goal: {primary_goal}",
            f"Plan duration: {plan_days} days",
            "",
            "## Expected markers",
            _safe_json_snippet(_expected_markers(style), max_chars=20_000),
            "",
            "## Rubrics and markers",
            render_rubrics_for_prompt(style=style, primary_goal=primary_goal),
            "",
            "## Deterministic evaluation report",
            _safe_json_snippet(deterministic_report, max_chars=20_000),
            "",
            "## Rollups context",
            _safe_json_snippet(rollups or {}, max_chars=10_000),
            "",
            "## Training plan JSON",
            _safe_json_snippet(plan_obj, max_chars=40_000),
            "",
            "## Output rules",
            "- Return one marker_results item for every expected marker_id.",
            "- Use rubric, marker_id, and marker exactly as provided.",
            "- verdict must be pass, partial, or fail.",
            "- score must be 0 to 5.",
            "- evidence must be concrete.",
            "- improvement_hint must be specific.",
        ]
    )


def _looks_internally_broken_soft_eval(
    summary: str,
    rubric_scores: dict[str, dict[str, Any]],
    marker_results: list[dict[str, Any]],
) -> bool:
    rubric_vals = [float((item or {}).get("score", 0)) for item in rubric_scores.values()]
    marker_vals = [float(item.get("score", 0)) for item in marker_results]
    return (
        bool(str(summary).strip())
        and bool(rubric_vals)
        and bool(marker_vals)
        and max(rubric_vals) == 0.0
        and max(marker_vals) == 0.0
    )


def _parse_soft_eval_json(out_text: str) -> dict[str, Any]:
    raw = json.loads(extract_json_object(out_text))
    if not isinstance(raw, dict):
        raise ValueError("Soft evaluator did not return a JSON object.")
    return raw


def _validate_soft_eval_completeness(raw: dict[str, Any]) -> None:
    rubric_scores_raw = raw.get("rubric_scores")
    summary_raw = str(raw.get("summary", "") or "").strip()

    if not isinstance(rubric_scores_raw, dict) or not rubric_scores_raw:
        raise ValueError("Soft evaluator returned missing or empty rubric_scores.")
    if not summary_raw:
        raise ValueError("Soft evaluator returned an empty summary.")

    expected = set(_rubric_ids())
    actual = {str(k) for k in rubric_scores_raw}
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"Soft evaluator omitted rubric ids: {missing}")


def _rubric_scores_look_usable(raw: Any, *, style: str) -> bool:
    if not isinstance(raw, dict):
        return False

    for rubric in get_default_rubrics(style):
        item = raw.get(rubric.rubric_id)
        if not isinstance(item, dict):
            return False

        raw_score = item.get("score")
        if raw_score is None:
            return False

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return False

        if not 0.0 <= score <= 100.0:
            return False

    return True


def _derive_rubric_scores_from_markers(
    marker_results: list[dict[str, Any]],
    *,
    style: str,
) -> dict[str, dict[str, Any]]:
    rubrics = get_default_rubrics(style)
    by_rubric: dict[str, list[float]] = {rubric.rubric_id: [] for rubric in rubrics}

    for item in marker_results:
        rubric_id = str(item.get("rubric", "") or "").strip()
        if rubric_id not in by_rubric:
            continue
        try:
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(5.0, score))
        by_rubric[rubric_id].append(score)

    out: dict[str, dict[str, Any]] = {}
    for rubric in rubrics:
        vals = by_rubric.get(rubric.rubric_id, [])
        if vals:
            rubric_score = round((sum(vals) / len(vals)) * 20.0, 1)
            reasoning = (
                "Derived from marker scores because the model omitted or malformed rubric_scores."
            )
        else:
            rubric_score = 0.0
            reasoning = "No marker evidence available to derive a rubric score."
        out[rubric.rubric_id] = {"score": rubric_score, "reasoning": reasoning}

    return out


def _parse_and_validate_soft_eval_output(out_text: str) -> dict[str, Any]:
    raw = _parse_soft_eval_json(out_text)
    _validate_soft_eval_completeness(raw)
    return raw


def _repair_soft_eval_output(
    client: Any,
    cfg: SoftEvalConfig,
    previous_output: str,
) -> dict[str, Any]:
    repair_prompt = (
        "Return ONLY valid JSON matching the required schema. "
        "Your previous output was malformed or incomplete.\n\n"
        "Common failure cases to avoid:\n"
        "- empty summary\n"
        "- missing rubric ids\n"
        "- markdown or commentary outside JSON\n\n"
        f"Previous output:\n{previous_output}\n"
    )
    repair_kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": "Return only valid JSON. No markdown.",
        "input": repair_prompt,
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
        "max_output_tokens": int(os.getenv("TRAILTRAINING_SOFT_EVAL_MAX_OUTPUT_TOKENS", "6000")),
    }
    repair_resp = call_with_schema(client, repair_kwargs, SOFT_EVAL_SCHEMA)
    repair_text = getattr(repair_resp, "output_text", None) or str(repair_resp)
    return _parse_and_validate_soft_eval_output(repair_text)


def _generate_marker_results_only(
    client: Any,
    cfg: SoftEvalConfig,
    plan_obj: dict[str, Any],
    deterministic_report: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": "Return only valid JSON. No markdown.",
        "input": _build_marker_only_prompt(
            plan_obj,
            deterministic_report,
            rollups,
            style=style,
            primary_goal=primary_goal,
        ),
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
        "max_output_tokens": int(os.getenv("TRAILTRAINING_SOFT_EVAL_MAX_OUTPUT_TOKENS", "6000")),
    }
    resp = call_with_schema(client, kwargs, _marker_only_schema())
    out_text = getattr(resp, "output_text", None) or str(resp)
    raw = _parse_soft_eval_json(out_text)
    marker_results = raw.get("marker_results")
    if not isinstance(marker_results, list) or not marker_results:
        raise ValueError("Marker-only repair returned missing or empty marker_results.")
    return marker_results


def _too_much_output_was_locally_derived(derived_fields: list[str]) -> bool:
    derived = set(derived_fields)
    return {
        "rubric_scores",
        "strengths",
        "concerns",
        "suggested_improvements",
    }.issubset(derived)


def evaluate_training_plan_soft(
    plan_obj: dict[str, Any],
    deterministic_report: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    cfg: SoftEvalConfig,
) -> dict[str, Any]:
    if not cfg.enabled:
        raise ValueError("Soft evaluation is disabled.")

    style, primary_goal = _resolve_style_and_goal(plan_obj, cfg)

    client = make_openrouter_client()
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": (
            "You are a strict but constructive endurance training-plan assessor. "
            "Your job is to grade plan quality against supplied rubrics, not to generate a new plan."
        ),
        "input": _build_soft_eval_prompt(
            plan_obj,
            deterministic_report,
            rollups,
            style=style,
            primary_goal=primary_goal,
        ),
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {"verbosity": cfg.verbosity},
        "max_output_tokens": int(os.getenv("TRAILTRAINING_SOFT_EVAL_MAX_OUTPUT_TOKENS", "6000")),
    }
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature

    repaired = False
    derived_fields: list[str] = []

    resp = call_with_schema(client, kwargs, SOFT_EVAL_SCHEMA)
    out_text = getattr(resp, "output_text", None) or str(resp)

    try:
        raw = _parse_and_validate_soft_eval_output(out_text)
    except Exception as exc:
        log.warning("Soft eval primary response invalid or incomplete: %s", exc)
        raw = _repair_soft_eval_output(client, cfg, out_text)
        repaired = True

    marker_results_raw = raw.get("marker_results")
    if not isinstance(marker_results_raw, list) or not marker_results_raw:
        log.warning("Soft eval missing marker_results; running marker-only repair pass.")
        marker_results_raw = _generate_marker_results_only(
            client,
            cfg,
            plan_obj,
            deterministic_report,
            rollups,
            style=style,
            primary_goal=primary_goal,
        )
        repaired = True

    marker_results = _normalize_marker_results(marker_results_raw, style=style)

    if _rubric_scores_look_usable(raw.get("rubric_scores"), style=style):
        rubric_scores = _normalize_rubric_scores(raw.get("rubric_scores"), style=style)
    else:
        log.warning(
            "Soft eval returned missing or malformed rubric_scores; deriving rubric scores from marker_results."
        )
        rubric_scores = _derive_rubric_scores_from_markers(marker_results, style=style)
        derived_fields.append("rubric_scores")

    overall_score = weighted_score_from_rubric_scores(rubric_scores, style=style)

    if _looks_internally_broken_soft_eval(
        str(raw.get("summary", "") or ""),
        rubric_scores,
        marker_results,
    ):
        raise ValueError(
            "Soft evaluator returned an internally inconsistent result "
            "(non-empty narrative with all-zero scores)."
        )

    strengths, concerns, suggested_improvements, feedback_derived = _build_feedback_lists(
        raw,
        rubric_scores,
        marker_results,
    )
    derived_fields.extend(feedback_derived)
    derived_fields = sorted(set(derived_fields))
    repaired = repaired or bool(derived_fields)

    if _too_much_output_was_locally_derived(derived_fields):
        raise ValueError("Soft evaluator output required too much local synthesis after repair.")

    payload = {
        "model": cfg.model,
        "style": style,
        "primary_goal": primary_goal,
        "summary": str(raw.get("summary", "") or "").strip(),
        "overall_score": overall_score,
        "grade": grade_from_score(overall_score),
        "confidence": _normalize_confidence(raw.get("confidence")),
        "rubric_scores": rubric_scores,
        "marker_results": marker_results,
        "strengths": strengths,
        "concerns": concerns,
        "suggested_improvements": suggested_improvements,
        "repaired": repaired,
        "derived_fields": derived_fields,
    }

    return SoftAssessmentArtifact.model_validate(payload).model_dump(mode="json")
