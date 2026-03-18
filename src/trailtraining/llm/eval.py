from __future__ import annotations

import logging
import math
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Optional

from trailtraining.contracts import EvaluationReportArtifact, TrainingPlanArtifact
from trailtraining.llm.constraints import (
    ConstraintConfig,
    evaluate_training_plan_quality,
    validate_training_plan,
)
from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft
from trailtraining.util.state import load_json

log = logging.getLogger(__name__)

# Markers with std above this threshold on a 1-5 scale are flagged as ambiguous.
_HIGH_VARIANCE_THRESHOLD = 0.5

# Temperature used for inter-rater runs when no explicit temperature is configured.
_INTER_RATER_DEFAULT_TEMPERATURE = 0.3


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
    """
    Compute per-marker score standard deviation across N evaluation runs.

    Returns a dict mapping marker_id → std (on a 0-5 scale).
    Only markers with ≥2 data points are included.
    """
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


def evaluate_training_plan_quality_file(
    coach_json_path: str,
    *,
    rollups_path: str | None = None,
    cfg: ConstraintConfig | None = None,
    soft_eval_cfg: SoftEvalConfig | None = None,
    primary_goal: str | None = None,
    soft_eval_runs: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Evaluate a training plan file, optionally running the soft evaluator N times.

    When soft_eval_runs > 1, the function runs the soft evaluator N times
    (with temperature > 0 if not explicitly set) and computes per-marker score
    variance.  Markers with std > 0.5 on a 1-5 scale are flagged in stats as
    potentially ambiguous rubric definitions.
    """
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
            # Single run — standard path.
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
            # Multi-run for inter-rater reliability measurement.
            # Use temperature > 0 so runs differ; only takes effect when supported.
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
                # Use first run as the primary assessment.
                primary = all_assessments[0]

                # Compute variance and flag high-variance markers.
                variance = _compute_marker_variance(all_run_markers)
                high_variance = {
                    mid: std for mid, std in variance.items() if std > _HIGH_VARIANCE_THRESHOLD
                }

                # Attach inter-rater metadata to the assessment.
                primary["inter_rater_runs"] = n
                primary["inter_rater_variance"] = variance

                stats = report_raw.setdefault("stats", {})
                if isinstance(stats, dict):
                    stats["inter_rater_runs"] = n
                    if high_variance:
                        stats["high_variance_markers"] = high_variance

                if high_variance:
                    for mid, std in sorted(high_variance.items()):
                        log.warning(
                            "High inter-rater variance on marker '%s' "
                            "(std=%.2f over %d runs) — "
                            "this marker's rubric definition may be ambiguous.",
                            mid,
                            std,
                            n,
                        )

                report_raw["soft_assessment"] = primary

    report = EvaluationReportArtifact.model_validate(report_raw)
    return report.model_dump(mode="json"), obj.model_dump(mode="json")
