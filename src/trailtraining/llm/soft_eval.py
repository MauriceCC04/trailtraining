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
    render_rubric_batch_for_prompt,
    render_rubrics_for_prompt,
    weighted_score_from_rubric_scores,
)
from trailtraining.llm.shared import call_with_schema, extract_json_object, make_openrouter_client
from trailtraining.util.text import _safe_json_snippet

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rubric batches for decomposed evaluation
# Each tuple is (batch_name, list_of_rubric_ids).
# goal_alignment + plan_coherence are batched together (related load/goal logic).
# caution_proportionality is isolated (hardest to score consistently).
# ---------------------------------------------------------------------------
_RUBRIC_BATCHES: list[tuple[str, list[str]]] = [
    ("goal_coherence", ["goal_alignment", "plan_coherence"]),
    ("explanation", ["explanation_quality"]),
    ("caution", ["caution_proportionality"]),
    ("actionability", ["actionability"]),
]


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


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def _rubric_ids() -> list[str]:
    return [r.rubric_id for r in get_default_rubrics("trailrunning")]


def _build_batch_marker_schema(batch_name: str) -> dict[str, Any]:
    """Per-batch schema requiring observation before score."""
    return {
        "name": f"trailtraining_batch_{batch_name}_v1",
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
                            "observation",
                            "verdict",
                            "score",
                            "evidence",
                            "improvement_hint",
                        ],
                        "properties": {
                            "rubric": {"type": "string"},
                            "marker_id": {"type": "string"},
                            "marker": {"type": "string"},
                            "observation": {
                                "type": "string",
                                "description": (
                                    "What you observe in the plan relevant to this marker. "
                                    "Write this BEFORE deciding the score."
                                ),
                            },
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


_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "name": "trailtraining_soft_eval_synthesis_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "confidence", "strengths", "concerns", "suggested_improvements"],
        "properties": {
            "summary": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "concerns": {"type": "array", "items": {"type": "string"}},
            "suggested_improvements": {"type": "array", "items": {"type": "string"}},
        },
    },
}

_COMPARE_PLANS_SCHEMA: dict[str, Any] = {
    "name": "trailtraining_compare_plans_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["preferred", "reasoning", "plan_a_advantages", "plan_b_advantages"],
        "properties": {
            "preferred": {"type": "string", "enum": ["plan_a", "plan_b", "tie"]},
            "reasoning": {"type": "string"},
            "plan_a_advantages": {"type": "array", "items": {"type": "string"}},
            "plan_b_advantages": {"type": "array", "items": {"type": "string"}},
        },
    },
}

# Keep the full schema for repair-path fallbacks (observation optional here for compat)
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
                        "observation": {"type": "string"},  # optional in repair schema
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
                            "observation": {"type": "string"},
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


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


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


def _expected_markers_for_rubrics(rubric_ids: list[str], style: str) -> list[dict[str, str]]:
    """Return expected marker stubs for the given rubric_ids only."""
    out: list[dict[str, str]] = []
    for rubric in get_default_rubrics(style):
        if rubric.rubric_id not in rubric_ids:
            continue
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
            observation_raw = str(item.get("observation", "") or "").strip()
            out.append(
                {
                    "rubric": rubric.rubric_id,
                    "marker_id": marker.marker_id,
                    "marker": marker.label,
                    "verdict": verdict,
                    "score": round(score, 1),
                    "observation": observation_raw if observation_raw else None,
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
            reasoning = "Derived from marker scores (per-rubric batch evaluation)."
        else:
            rubric_score = 0.0
            reasoning = "No marker evidence available to derive a rubric score."
        out[rubric.rubric_id] = {"score": rubric_score, "reasoning": reasoning}

    return out


def _parse_soft_eval_json(out_text: str) -> dict[str, Any]:
    raw = json.loads(extract_json_object(out_text))
    if not isinstance(raw, dict):
        raise ValueError("Soft evaluator did not return a JSON object.")
    return raw


# ---------------------------------------------------------------------------
# Per-batch prompt and execution
# ---------------------------------------------------------------------------


def _build_batch_prompt(
    rubric_ids: list[str],
    plan_obj: dict[str, Any],
    deterministic_report: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
) -> str:
    """Build the prompt for a single rubric batch call."""
    has_week_coherence = "plan_coherence" in rubric_ids
    rubric_text = render_rubric_batch_for_prompt(rubric_ids, style=style, primary_goal=primary_goal)
    expected = _expected_markers_for_rubrics(rubric_ids, style)

    instructions = [
        "You are evaluating specific rubrics of an endurance training plan.",
        "Score ONLY the rubrics and markers listed below — do not evaluate other rubrics.",
        "",
        "For each marker, you MUST follow this exact order in your JSON output:",
        "  1. observation: Describe what you observe in the plan relevant to this marker.",
        "     This must be written BEFORE you decide the verdict or score.",
        "     It makes your reasoning auditable.",
        "  2. verdict: pass / partial / fail",
        "  3. score: 0-5",
        "  4. evidence: Concrete quote or reference from the plan.",
        "  5. improvement_hint: A specific, actionable suggestion.",
        "",
    ]

    if has_week_coherence:
        instructions += [
            "IMPORTANT: Score the 'week_coherence' marker LAST.",
            "Only score week_coherence after you have scored all other markers in plan_coherence.",
            "week_coherence evaluates the full week as a unit, so per-session markers must come first.",
            "",
        ]

    return "\n".join(
        [
            *instructions,
            "## Rubrics to evaluate in this call",
            rubric_text,
            "",
            "## Expected markers — include exactly these in marker_results",
            _safe_json_snippet(expected, max_chars=8_000),
            "",
            "## Deterministic evaluation report (context)",
            _safe_json_snippet(deterministic_report, max_chars=12_000),
            "",
            "## Rollups context",
            _safe_json_snippet(rollups or {}, max_chars=8_000),
            "",
            "## Training plan JSON",
            _safe_json_snippet(plan_obj, max_chars=40_000),
            "",
            "## Output rules",
            "- Return JSON only — no markdown fences, no commentary.",
            "- Include one marker_results item for every expected marker_id.",
            "- observation MUST describe what you see before you assign a score.",
            "- evidence MUST cite specific content from the plan.",
            "- improvement_hint MUST be specific, not generic.",
        ]
    )


def _build_synthesis_prompt(
    plan_obj: dict[str, Any],
    all_marker_results: list[dict[str, Any]],
    rollups: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
) -> str:
    """Build the synthesis prompt from all marker results."""
    return "\n".join(
        [
            "You have evaluated a training plan across all rubrics.",
            "Based on the marker results below, write a synthesis assessment.",
            f"Style: {style}",
            f"Primary goal: {primary_goal}",
            "",
            "## All marker results (from per-rubric evaluation)",
            _safe_json_snippet(all_marker_results, max_chars=25_000),
            "",
            "## Training plan JSON",
            _safe_json_snippet(plan_obj, max_chars=25_000),
            "",
            "## Output rules",
            "- Return JSON only.",
            "- summary: 2-3 sentences on overall plan quality, grounded in markers.",
            "- confidence: low / medium / high — how confident are you in the assessment?",
            "- strengths: 2-4 concrete strengths from the plan (not generic praise).",
            "- concerns: 1-3 concrete concerns tied to low-scoring markers.",
            "- suggested_improvements: 2-4 specific, actionable improvements.",
        ]
    )


def _run_rubric_batch(
    client: Any,
    cfg: SoftEvalConfig,
    rubric_ids: list[str],
    batch_name: str,
    plan_obj: dict[str, Any],
    deterministic_report: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
) -> list[dict[str, Any]]:
    """Execute one LLM call for a rubric batch; return raw marker_results list."""
    schema = _build_batch_marker_schema(batch_name)
    prompt = _build_batch_prompt(
        rubric_ids,
        plan_obj,
        deterministic_report,
        rollups,
        style=style,
        primary_goal=primary_goal,
    )

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": (
            "You are a strict endurance training-plan assessor. "
            "Score only the markers listed. "
            "Write your observation before assigning a score."
        ),
        "input": prompt,
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {"verbosity": cfg.verbosity},
        "max_output_tokens": int(os.getenv("TRAILTRAINING_SOFT_EVAL_MAX_OUTPUT_TOKENS", "6000")),
    }
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature

    resp = call_with_schema(client, kwargs, schema)
    out_text = getattr(resp, "output_text", None) or str(resp)

    try:
        raw = _parse_soft_eval_json(out_text)
        results = raw.get("marker_results")
        if not isinstance(results, list) or not results:
            raise ValueError("Empty marker_results in batch response.")
        return results
    except Exception as exc:
        log.warning(
            "Batch '%s' returned unusable results (%s); falling back to marker-only repair.",
            batch_name,
            exc,
        )
        return _generate_marker_results_only(
            client,
            cfg,
            plan_obj,
            deterministic_report,
            rollups,
            style=style,
            primary_goal=primary_goal,
            rubric_ids=rubric_ids,
        )


def _run_synthesis_call(
    client: Any,
    cfg: SoftEvalConfig,
    plan_obj: dict[str, Any],
    all_marker_results: list[dict[str, Any]],
    rollups: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
) -> dict[str, Any]:
    """Run a synthesis call to produce summary, strengths, concerns, improvements."""
    prompt = _build_synthesis_prompt(
        plan_obj, all_marker_results, rollups, style=style, primary_goal=primary_goal
    )

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": (
            "Synthesize the marker evaluations into a concise assessment. "
            "Ground every claim in the marker evidence."
        ),
        "input": prompt,
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {"verbosity": cfg.verbosity},
        "max_output_tokens": int(os.getenv("TRAILTRAINING_SOFT_EVAL_MAX_OUTPUT_TOKENS", "4000")),
    }
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature

    try:
        resp = call_with_schema(client, kwargs, _SYNTHESIS_SCHEMA)
        out_text = getattr(resp, "output_text", None) or str(resp)
        raw = _parse_soft_eval_json(out_text)
        return {
            "summary": str(raw.get("summary", "") or "").strip(),
            "confidence": _normalize_confidence(raw.get("confidence")),
            "strengths": _clean_string_list(raw.get("strengths")),
            "concerns": _clean_string_list(raw.get("concerns")),
            "suggested_improvements": _clean_string_list(raw.get("suggested_improvements")),
        }
    except Exception as exc:
        log.warning("Synthesis call failed: %s; using empty synthesis.", exc)
        return {
            "summary": "",
            "confidence": "low",
            "strengths": [],
            "concerns": [],
            "suggested_improvements": [],
        }


def _generate_marker_results_only(
    client: Any,
    cfg: SoftEvalConfig,
    plan_obj: dict[str, Any],
    deterministic_report: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
    rubric_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    expected = (
        _expected_markers_for_rubrics(rubric_ids, style) if rubric_ids else _expected_markers(style)
    )
    rubric_text = (
        render_rubric_batch_for_prompt(rubric_ids, style=style, primary_goal=primary_goal)
        if rubric_ids
        else render_rubrics_for_prompt(style=style, primary_goal=primary_goal)
    )
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": "Return only valid JSON. No markdown.",
        "input": "\n".join(
            [
                "Return JSON only.",
                "Your only job is to produce marker_results for every expected marker.",
                "Do not omit markers.",
                "",
                f"Style: {style}",
                f"Primary goal: {primary_goal}",
                "",
                "## Expected markers",
                _safe_json_snippet(expected, max_chars=15_000),
                "",
                "## Rubrics and markers",
                rubric_text,
                "",
                "## Deterministic evaluation report",
                _safe_json_snippet(deterministic_report, max_chars=15_000),
                "",
                "## Rollups context",
                _safe_json_snippet(rollups or {}, max_chars=8_000),
                "",
                "## Training plan JSON",
                _safe_json_snippet(plan_obj, max_chars=35_000),
                "",
                "## Output rules",
                "- Return one marker_results item for every expected marker_id.",
                "- Use rubric, marker_id, and marker exactly as provided.",
                "- verdict must be pass, partial, or fail.",
                "- score must be 0 to 5.",
                "- observation: what you see in the plan for this marker.",
                "- evidence must be concrete.",
                "- improvement_hint must be specific.",
            ]
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


def _too_much_output_was_locally_derived(derived_fields: list[str]) -> bool:
    derived = set(derived_fields)
    return {
        "strengths",
        "concerns",
        "suggested_improvements",
    }.issubset(derived)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def evaluate_training_plan_soft(
    plan_obj: dict[str, Any],
    deterministic_report: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    cfg: SoftEvalConfig,
) -> dict[str, Any]:
    """
    Evaluate a training plan using a decomposed per-rubric batch architecture.

    Instead of one large prompt with all 18+ markers, we run one LLM call per
    rubric batch to eliminate inter-marker anchoring bias.  Each batch call
    requires `observation` before `score` to make reasoning auditable.  A final
    synthesis call produces the summary, strengths, concerns, and improvements.
    """
    if not cfg.enabled:
        raise ValueError("Soft evaluation is disabled.")

    style, primary_goal = _resolve_style_and_goal(plan_obj, cfg)
    client = make_openrouter_client()

    # --- Per-rubric batch calls ---
    repaired = False
    all_marker_results_raw: list[dict[str, Any]] = []

    for batch_name, rubric_ids in _RUBRIC_BATCHES:
        try:
            batch_results = _run_rubric_batch(
                client,
                cfg,
                rubric_ids,
                batch_name,
                plan_obj,
                deterministic_report,
                rollups,
                style=style,
                primary_goal=primary_goal,
            )
            all_marker_results_raw.extend(batch_results)
        except Exception as exc:
            log.warning(
                "Rubric batch '%s' failed entirely: %s. Results for these rubrics will be empty.",
                batch_name,
                exc,
            )
            repaired = True

    if not all_marker_results_raw:
        log.warning("All batches returned empty results; falling back to single-call evaluation.")
        repaired = True
        all_marker_results_raw = _generate_marker_results_only(
            client,
            cfg,
            plan_obj,
            deterministic_report,
            rollups,
            style=style,
            primary_goal=primary_goal,
        )

    # --- Normalise markers and derive rubric scores ---
    marker_results = _normalize_marker_results(all_marker_results_raw, style=style)
    # rubric_scores are always derived from markers in the batch architecture
    rubric_scores = _derive_rubric_scores_from_markers(marker_results, style=style)

    overall_score = weighted_score_from_rubric_scores(rubric_scores, style=style)

    if _looks_internally_broken_soft_eval("placeholder", rubric_scores, marker_results):
        raise ValueError(
            "Soft evaluator returned an internally inconsistent result "
            "(non-empty narrative with all-zero scores)."
        )

    # --- Synthesis call ---
    synthesis = _run_synthesis_call(
        client,
        cfg,
        plan_obj,
        marker_results,
        rollups,
        style=style,
        primary_goal=primary_goal,
    )

    # --- Apply feedback fallbacks ---
    strengths, concerns, suggested_improvements, feedback_derived = _build_feedback_lists(
        synthesis,
        rubric_scores,
        marker_results,
    )

    if _too_much_output_was_locally_derived(feedback_derived):
        log.warning("Synthesis output required heavy local fallback; results may be lower quality.")
        repaired = True

    derived_fields = sorted(set(["rubric_scores", *feedback_derived]))

    payload = {
        "model": cfg.model,
        "style": style,
        "primary_goal": primary_goal,
        "summary": synthesis["summary"],
        "overall_score": overall_score,
        "grade": grade_from_score(overall_score),
        "confidence": synthesis["confidence"],
        "rubric_scores": rubric_scores,
        "marker_results": marker_results,
        "strengths": strengths,
        "concerns": concerns,
        "suggested_improvements": suggested_improvements,
        "repaired": repaired,
        "derived_fields": derived_fields,
    }

    return SoftAssessmentArtifact.model_validate(payload).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Forced-ranking comparison
# ---------------------------------------------------------------------------


def compare_plans(
    plan_a: dict[str, Any],
    plan_b: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    cfg: SoftEvalConfig,
) -> dict[str, Any]:
    """
    Compare two training plans and return which is better with reasoning.

    Forced comparison is harder to game than absolute scoring and reveals
    preference ordering that absolute scores can obscure.

    Returns a dict with keys: preferred, reasoning, plan_a_advantages, plan_b_advantages.
    """
    style, primary_goal = _resolve_style_and_goal(plan_a, cfg)
    client = make_openrouter_client()

    prompt = "\n".join(
        [
            "Compare these two training plans and determine which is better.",
            "First reason through what you observe in each plan, then give your preference.",
            "Do NOT simply prefer the longer or more complex plan — prefer the one that better",
            "serves the athlete's goal given their current state.",
            "",
            f"Style: {style}",
            f"Primary goal: {primary_goal}",
            "",
            "## Plan A",
            _safe_json_snippet(plan_a, max_chars=25_000),
            "",
            "## Plan B",
            _safe_json_snippet(plan_b, max_chars=25_000),
            "",
            "## Context (rollups)",
            _safe_json_snippet(rollups or {}, max_chars=8_000),
            "",
            "## Output rules",
            "- preferred: 'plan_a', 'plan_b', or 'tie'",
            "- reasoning: 3-5 sentences explaining your choice, citing specific differences.",
            "- plan_a_advantages: 2-4 specific advantages of Plan A over Plan B.",
            "- plan_b_advantages: 2-4 specific advantages of Plan B over Plan A.",
            "- Return JSON only.",
        ]
    )

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "instructions": (
            "Compare two training plans. Reason carefully before concluding. "
            "Be specific about what makes one plan better than the other."
        ),
        "input": prompt,
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {"verbosity": cfg.verbosity},
    }
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature

    resp = call_with_schema(client, kwargs, _COMPARE_PLANS_SCHEMA)
    out_text = getattr(resp, "output_text", None) or str(resp)

    try:
        raw = json.loads(extract_json_object(out_text))
    except Exception as exc:
        log.warning("compare_plans JSON parse failed: %s", exc)
        return {
            "preferred": "tie",
            "reasoning": "Could not parse comparison response.",
            "plan_a_advantages": [],
            "plan_b_advantages": [],
        }

    return {
        "preferred": str(raw.get("preferred", "tie")),
        "reasoning": str(raw.get("reasoning", "")).strip(),
        "plan_a_advantages": _clean_string_list(raw.get("plan_a_advantages", [])),
        "plan_b_advantages": _clean_string_list(raw.get("plan_b_advantages", [])),
    }


# ---------------------------------------------------------------------------
# Legacy soft eval prompt (kept for repair path reference)
# ---------------------------------------------------------------------------


def _build_soft_eval_prompt(
    plan_obj: dict[str, Any],
    deterministic_report: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    *,
    style: str,
    primary_goal: str,
) -> str:
    """Single-call prompt, retained for repair paths and backwards compatibility."""
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
                    f"This is a {plan_weeks}-week plan — evaluate periodization across all weeks.",
                    "Hard-day and rest-day constraints apply per rolling 7-day window.",
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
            "- For each marker: write observation (what you see) before scoring.",
            "- Use concrete evidence from the plan.",
            "- strengths: 2-4 concrete strengths.",
            "- concerns: at least 1 concrete concern.",
            "- suggested_improvements: 2-4 specific improvements.",
        ]
    )
