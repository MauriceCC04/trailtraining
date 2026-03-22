from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Optional

from trailtraining.contracts import EvaluationReportArtifact, TrainingPlanArtifact
from trailtraining.llm.constraints import (
    ConstraintConfig,
    evaluate_training_plan_quality,
    validate_training_plan,
)
from trailtraining.llm.rubrics import (
    get_default_rubrics,
    grade_from_score,
    weighted_score_from_rubric_scores,
)
from trailtraining.llm.soft_eval import (
    SoftEvalConfig,
    _clean_string_list,
    _derive_rubric_scores_from_markers,
    _normalize_confidence,
    _normalize_marker_results,
    _normalize_verdict,
    evaluate_training_plan_soft,
)
from trailtraining.util.state import load_json

log = logging.getLogger(__name__)

_HIGH_VARIANCE_THRESHOLD = 0.5
_INTER_RATER_DEFAULT_TEMPERATURE = 0.3
_CONSENSUS_METHOD = "median-marker-scores + majority-verdict + representative-narrative"

_SCHEMA_BLOCKER_CODES = {
    "BAD_DATE",
    "DUPLICATE_DATE",
    "NON_CONSECUTIVE_DATES",
    "REST_DAY_FLAG_MISMATCH",
    "REST_DAY_NONZERO_DURATION",
    "REST_DAY_TOO_LONG",
    "REST_DAY_MARKED_HARD",
    "REST_DAY_WORKOUT_CONFLICT",
    "HARD_SESSION_NOT_MARKED_HARD",
    "NON_HARD_SESSION_MARKED_HARD",
    "SESSION_ZERO_DURATION_CONFLICT",
}
_ARITHMETIC_BLOCKER_CODES = {
    "WEEKLY_TOTALS_MISMATCH",
    "UNVERIFIABLE_WEEKLY_TOTAL_DISTANCE",
    "UNVERIFIABLE_WEEKLY_TOTAL_ELEVATION",
}


def _load_rollups_near(
    path: Path, explicit_rollups: Optional[str] = None
) -> Optional[dict[str, Any]]:
    if explicit_rollups:
        p = Path(explicit_rollups).expanduser().resolve()
        x = load_json(p, default=None)
        return x if isinstance(x, dict) else None

    guess = path.parent / "combined_rollups.json"
    if guess.exists():
        x = load_json(guess, default=None)
        return x if isinstance(x, dict) else None

    return None


def evaluate_training_plan_file(
    coach_json_path: str,
    *,
    rollups_path: Optional[str] = None,
    cfg: Optional[ConstraintConfig] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p = Path(coach_json_path).expanduser().resolve()
    obj = load_json(p, default=None)
    if not isinstance(obj, dict):
        raise ValueError("Coach JSON must be an object (dict).")

    rollups = _load_rollups_near(p, rollups_path)
    ccfg = cfg or ConstraintConfig()
    violations = validate_training_plan(obj, rollups, ccfg)
    return violations, obj


def _compute_marker_variance(all_runs: list[list[dict[str, Any]]]) -> dict[str, float]:
    by_marker: dict[str, list[float]] = {}
    for run in all_runs:
        for item in run:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("marker_id", "") or "")
            if not mid:
                continue
            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            by_marker.setdefault(mid, []).append(score)

    out: dict[str, float] = {}
    for mid, vals in by_marker.items():
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        out[mid] = round(math.sqrt(variance), 3)
    return out


def _median(values: list[float]) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _select_representative_item(
    items: list[dict[str, Any]],
    *,
    target_score: Optional[float] = None,
) -> dict[str, Any]:
    if not items:
        return {}
    if target_score is None:
        return items[0]

    best_item = items[0]
    best_key: tuple[float, int, int] | None = None
    for idx, item in enumerate(items):
        score = _as_float(item.get("score", 0.0))
        completeness = sum(
            1
            for key in ("observation", "evidence", "improvement_hint")
            if str(item.get(key, "") or "").strip()
        )
        key = (abs(score - target_score), -completeness, idx)
        if best_key is None or key < best_key:
            best_key = key
            best_item = item
    return best_item


def _merge_ranked_string_lists(raw_lists: list[Any], *, limit: int) -> list[str]:
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    canonical: dict[str, str] = {}

    for outer_idx, raw in enumerate(raw_lists):
        for item in _clean_string_list(raw):
            key = item.casefold()
            counts[key] += 1
            canonical.setdefault(key, item)
            first_seen.setdefault(key, outer_idx)

    ranked = sorted(
        counts,
        key=lambda key: (-counts[key], first_seen.get(key, 0), canonical[key]),
    )
    return [canonical[key] for key in ranked[:limit]]


def _select_representative_assessment(all_assessments: list[dict[str, Any]]) -> dict[str, Any]:
    if not all_assessments:
        return {}
    target = _median([_as_float(item.get("overall_score", 0.0)) for item in all_assessments])
    best = all_assessments[0]
    best_key: tuple[float, int] | None = None
    for idx, item in enumerate(all_assessments):
        key = (abs(_as_float(item.get("overall_score", 0.0)) - target), idx)
        if best_key is None or key < best_key:
            best_key = key
            best = item
    return best


def _aggregate_marker_results(
    all_runs: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_marker: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        for item in run:
            if not isinstance(item, dict):
                continue
            marker_id = str(item.get("marker_id", "") or "").strip()
            if not marker_id:
                continue
            by_marker.setdefault(marker_id, []).append(item)

    aggregated_raw: list[dict[str, Any]] = []
    for marker_id, items in by_marker.items():
        scores = [_as_float(item.get("score", 0.0)) for item in items]
        agg_score = round(_median(scores), 1)

        verdict_counts = Counter(
            str(item.get("verdict", "") or "").strip().lower()
            for item in items
            if str(item.get("verdict", "") or "").strip().lower() in {"pass", "partial", "fail"}
        )
        if verdict_counts:
            top_count = max(verdict_counts.values())
            winners = [name for name, count in verdict_counts.items() if count == top_count]
            agg_verdict = winners[0] if len(winners) == 1 else _normalize_verdict(None, agg_score)
        else:
            agg_verdict = _normalize_verdict(None, agg_score)

        representative = _select_representative_item(items, target_score=agg_score)
        aggregated_raw.append(
            {
                "rubric": representative.get("rubric", ""),
                "marker_id": marker_id,
                "marker": representative.get("marker", ""),
                "observation": representative.get("observation"),
                "verdict": agg_verdict,
                "score": agg_score,
                "evidence": representative.get("evidence", ""),
                "improvement_hint": representative.get("improvement_hint", ""),
            }
        )

    return aggregated_raw


def _aggregate_soft_assessments(
    all_assessments: list[dict[str, Any]],
    *,
    style: str,
    variance: dict[str, float],
) -> dict[str, Any]:
    representative = _select_representative_assessment(all_assessments)

    marker_results_raw = _aggregate_marker_results(
        [assessment.get("marker_results") or [] for assessment in all_assessments],
    )

    rubric_scores = _derive_rubric_scores_from_markers(marker_results_raw, style=style)

    present_rubric_ids = {
        str(item.get("rubric", "") or "").strip()
        for item in marker_results_raw
        if str(item.get("rubric", "") or "").strip()
    }
    present_rubrics = [
        rubric for rubric in get_default_rubrics(style) if rubric.rubric_id in present_rubric_ids
    ]
    if not present_rubrics:
        present_rubrics = list(get_default_rubrics(style))

    overall_score = weighted_score_from_rubric_scores(
        rubric_scores,
        rubrics=present_rubrics,
        style=style,
    )

    marker_results = _normalize_marker_results(marker_results_raw, style=style)

    confidence_counts = Counter(
        _normalize_confidence(assessment.get("confidence")) for assessment in all_assessments
    )
    confidence = representative.get("confidence") or "medium"
    if confidence_counts:
        top_count = max(confidence_counts.values())
        winners = [name for name, count in confidence_counts.items() if count == top_count]
        confidence = winners[0] if len(winners) == 1 else _normalize_confidence(confidence)
    confidence = _normalize_confidence(confidence)

    strengths = _merge_ranked_string_lists(
        [assessment.get("strengths", []) for assessment in all_assessments],
        limit=4,
    )
    concerns = _merge_ranked_string_lists(
        [assessment.get("concerns", []) for assessment in all_assessments],
        limit=3,
    )
    suggested_improvements = _merge_ranked_string_lists(
        [assessment.get("suggested_improvements", []) for assessment in all_assessments],
        limit=4,
    )
    derived_fields = sorted(
        {
            field
            for assessment in all_assessments
            for field in (assessment.get("derived_fields") or [])
            if isinstance(field, str) and field.strip()
        }
    )

    return {
        "model": str(representative.get("model", "") or ""),
        "style": representative.get("style") or style,
        "primary_goal": str(representative.get("primary_goal", "") or "").strip(),
        "summary": str(representative.get("summary", "") or "").strip(),
        "overall_score": overall_score,
        "grade": grade_from_score(overall_score),
        "confidence": confidence,
        "rubric_scores": rubric_scores,
        "marker_results": marker_results,
        "strengths": strengths,
        "concerns": concerns,
        "suggested_improvements": suggested_improvements,
        "repaired": any(bool(assessment.get("repaired")) for assessment in all_assessments),
        "derived_fields": derived_fields,
        "inter_rater_runs": len(all_assessments),
        "inter_rater_variance": variance,
    }


def _blocking_issue_labels(violations: list[dict[str, Any]]) -> list[str]:
    codes = {str(v.get("code") or "") for v in violations if isinstance(v, dict)}
    issues: list[str] = []
    if codes & _SCHEMA_BLOCKER_CODES:
        issues.append("schema_validity")
    if codes & _ARITHMETIC_BLOCKER_CODES:
        issues.append("derived_totals")
    return issues


def _apply_blocking_caps(score: float, blocking_issues: list[str]) -> float:
    capped = float(score)
    if len(blocking_issues) >= 2:
        capped = min(capped, 79.0)
    elif blocking_issues:
        capped = min(capped, 89.0)
    return round(capped, 1)


def _finalize_report_scores(report_raw: dict[str, Any]) -> dict[str, Any]:
    deterministic_score = _as_float(report_raw.get("score", 0.0))
    deterministic_grade = str(report_raw.get("grade", "?") or "?")
    violations = report_raw.get("violations") or []
    blocking_issues = _blocking_issue_labels(violations if isinstance(violations, list) else [])

    soft = report_raw.get("soft_assessment")
    score_components: dict[str, float] = {"deterministic": deterministic_score}
    base_score = deterministic_score
    score_basis = "deterministic"

    if isinstance(soft, dict):
        soft_overall = _as_float(
            soft.get("overall_score", deterministic_score), deterministic_score
        )
        score_components["soft_overall"] = soft_overall
        justification = report_raw.get("subscores", {}).get("justification")
        if isinstance(justification, (int, float)):
            score_components["claim_support"] = float(justification)
        base_score = soft_overall
        score_basis = "soft_overall"

    final_score = _apply_blocking_caps(base_score, blocking_issues)
    final_grade = grade_from_score(final_score)

    report_raw["deterministic_score"] = deterministic_score
    report_raw["deterministic_grade"] = deterministic_grade
    report_raw["score_components"] = {k: round(v, 1) for k, v in score_components.items()}
    report_raw["blocking_issues"] = blocking_issues
    report_raw["score"] = final_score
    report_raw["grade"] = final_grade

    stats = report_raw.setdefault("stats", {})
    if isinstance(stats, dict):
        stats["final_score_basis"] = score_basis
        if blocking_issues:
            stats["grade_caps_applied"] = blocking_issues

    return report_raw


def evaluate_training_plan_quality_file(
    coach_json_path: str,
    *,
    rollups_path: str | None = None,
    cfg: ConstraintConfig | None = None,
    soft_eval_cfg: SoftEvalConfig | None = None,
    primary_goal: str | None = None,
    soft_eval_runs: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    p = Path(coach_json_path).expanduser().resolve()
    raw_obj = load_json(p, default=None)
    obj = TrainingPlanArtifact.model_validate(raw_obj)

    if primary_goal and primary_goal.strip():
        obj.meta.primary_goal = primary_goal.strip()

    rollups = _load_rollups_near(p, rollups_path)
    ccfg = cfg or ConstraintConfig()

    report_raw = evaluate_training_plan_quality(obj.model_dump(mode="json"), rollups, ccfg)

    if soft_eval_cfg and soft_eval_cfg.enabled:
        n = max(1, int(soft_eval_runs))

        if n == 1:
            try:
                report_raw["soft_assessment"] = evaluate_training_plan_soft(
                    obj.model_dump(mode="json"),
                    report_raw,
                    rollups,
                    soft_eval_cfg,
                )
            except Exception as exc:
                log.warning("Soft evaluation failed: %s", exc)
                stats = report_raw.setdefault("stats", {})
                if isinstance(stats, dict):
                    stats["soft_eval_error"] = str(exc)

        else:
            run_cfg = soft_eval_cfg
            if soft_eval_cfg.temperature is None:
                run_cfg = _dc_replace(soft_eval_cfg, temperature=_INTER_RATER_DEFAULT_TEMPERATURE)

            all_assessments: list[dict[str, Any]] = []
            all_run_markers: list[list[dict[str, Any]]] = []

            for i in range(n):
                try:
                    assessment = evaluate_training_plan_soft(
                        obj.model_dump(mode="json"),
                        report_raw,
                        rollups,
                        run_cfg,
                    )
                    all_assessments.append(assessment)
                    all_run_markers.append(assessment.get("marker_results") or [])
                except Exception as exc:
                    log.warning("Soft eval run %d/%d failed: %s", i + 1, n, exc)

            if not all_assessments:
                stats = report_raw.setdefault("stats", {})
                if isinstance(stats, dict):
                    stats["soft_eval_error"] = f"All {n} runs failed."
            else:
                successful_runs = len(all_assessments)
                style = str(all_assessments[0].get("style") or (obj.meta.style or "trailrunning"))
                variance = _compute_marker_variance(all_run_markers) if successful_runs > 1 else {}
                high_variance = {
                    mid: std for mid, std in variance.items() if std > _HIGH_VARIANCE_THRESHOLD
                }

                if successful_runs > 1:
                    consensus = _aggregate_soft_assessments(
                        all_assessments,
                        style=style,
                        variance=variance,
                    )
                else:
                    consensus = all_assessments[0]
                    consensus["inter_rater_runs"] = successful_runs
                    consensus["inter_rater_variance"] = variance

                stats = report_raw.setdefault("stats", {})
                if isinstance(stats, dict):
                    stats["inter_rater_runs"] = successful_runs
                    stats["inter_rater_consensus_method"] = _CONSENSUS_METHOD
                    if successful_runs != n:
                        stats["soft_eval_failed_runs"] = n - successful_runs
                    if high_variance:
                        stats["high_variance_markers"] = high_variance

                if high_variance:
                    for mid, std in sorted(high_variance.items()):
                        log.warning(
                            "High inter-rater variance on marker '%s' "
                            "(std=%.2f over %d successful runs) - "
                            "this marker's rubric definition may be ambiguous.",
                            mid,
                            std,
                            successful_runs,
                        )

                report_raw["soft_assessment"] = consensus

    report_raw = _finalize_report_scores(report_raw)
    report = EvaluationReportArtifact.model_validate(report_raw)
    return report.model_dump(mode="json"), obj.model_dump(mode="json")
