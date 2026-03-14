# src/trailtraining/llm/eval.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from trailtraining.contracts import EvaluationReportArtifact, TrainingPlanArtifact
from trailtraining.llm.constraints import (
    ConstraintConfig,
    evaluate_training_plan_quality,
    validate_training_plan,
)
from trailtraining.util.state import load_json


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
    """
    Backwards compatible: returns (violations, obj)
    """
    p = Path(coach_json_path).expanduser().resolve()
    obj = load_json(p, default=None)
    if not isinstance(obj, dict):
        raise ValueError("Coach JSON must be an object (dict).")

    rollups = _load_rollups_near(p, rollups_path)
    ccfg = cfg or ConstraintConfig()
    violations = validate_training_plan(obj, rollups, ccfg)
    return violations, obj


def evaluate_training_plan_quality_file(
    coach_json_path: str,
    *,
    rollups_path: str | None = None,
    cfg: ConstraintConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    p = Path(coach_json_path).expanduser().resolve()
    raw_obj = load_json(p, default=None)
    obj = TrainingPlanArtifact.model_validate(raw_obj)

    rollups = _load_rollups_near(p, rollups_path)
    ccfg = cfg or ConstraintConfig()

    report_raw = evaluate_training_plan_quality(obj.model_dump(mode="python"), rollups, ccfg)
    report = EvaluationReportArtifact.model_validate(report_raw)

    return report.model_dump(mode="json"), obj.model_dump(mode="json")
