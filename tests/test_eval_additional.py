from __future__ import annotations

import json
from pathlib import Path

import pytest
from trailtraining.llm.constraints import ConstraintConfig
from trailtraining.llm.eval import (
    SoftEvalConfig,
    _compute_marker_variance,
    _load_rollups_near,
    evaluate_training_plan_file,
    evaluate_training_plan_quality_file,
)

from tests._artifact_builders import make_training_plan_artifact


def test_load_rollups_near_prefers_explicit_path(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    explicit = tmp_path / "rollups-explicit.json"
    explicit.write_text(json.dumps({"windows": {"7": {}}}), encoding="utf-8")

    loaded = _load_rollups_near(plan_path, str(explicit))

    assert loaded == {"windows": {"7": {}}}


def test_load_rollups_near_uses_sibling_combined_rollups(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    sibling = tmp_path / "combined_rollups.json"
    sibling.write_text(json.dumps({"windows": {"28": {}}}), encoding="utf-8")

    loaded = _load_rollups_near(plan_path)

    assert loaded == {"windows": {"28": {}}}


def test_evaluate_training_plan_file_rejects_non_object(tmp_path: Path) -> None:
    input_path = tmp_path / "plan.json"
    input_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="Coach JSON must be an object"):
        evaluate_training_plan_file(str(input_path))


def test_compute_marker_variance_returns_per_marker_std() -> None:
    runs = [
        [{"marker_id": "m1", "score": 1}, {"marker_id": "m2", "score": 5}],
        [{"marker_id": "m1", "score": 3}, {"marker_id": "m2", "score": 5}],
        [{"marker_id": "m1", "score": 5}, {"marker_id": "m2", "score": 5}],
    ]

    variance = _compute_marker_variance(runs)

    assert variance["m1"] > 0.0
    assert variance["m2"] == 0.0


def test_evaluate_training_plan_quality_file_records_soft_eval_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trailtraining.llm.eval as eval_mod

    plan = make_training_plan_artifact()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def fake_soft_eval(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(eval_mod, "evaluate_training_plan_soft", fake_soft_eval)

    report, _ = evaluate_training_plan_quality_file(
        str(plan_path),
        cfg=ConstraintConfig(min_signal_ids_per_day=0),
        soft_eval_cfg=SoftEvalConfig(enabled=True),
    )

    assert report["stats"]["soft_eval_error"] == "boom"
    assert report.get("soft_assessment") is None


def test_evaluate_training_plan_quality_file_multi_run_tracks_high_variance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trailtraining.llm.eval as eval_mod

    plan = make_training_plan_artifact()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    assessments = [
        {
            "model": "fake-model",
            "style": "trailrunning",
            "primary_goal": "build trail endurance",
            "summary": "run 1",
            "overall_score": 80,
            "grade": "B",
            "confidence": "medium",
            "rubric_scores": {"goal_alignment": {"score": 80, "reasoning": "ok"}},
            "marker_results": [
                {
                    "rubric": "goal_alignment",
                    "marker_id": "m1",
                    "marker": "marker 1",
                    "verdict": "partial",
                    "score": 1,
                    "observation": "obs",
                    "evidence": "ev",
                    "improvement_hint": "hint",
                }
            ],
            "strengths": ["s1"],
            "concerns": ["c1"],
            "suggested_improvements": ["i1"],
            "repaired": False,
            "derived_fields": [],
        },
        {
            "model": "fake-model",
            "style": "trailrunning",
            "primary_goal": "build trail endurance",
            "summary": "run 2",
            "overall_score": 80,
            "grade": "B",
            "confidence": "medium",
            "rubric_scores": {"goal_alignment": {"score": 80, "reasoning": "ok"}},
            "marker_results": [
                {
                    "rubric": "goal_alignment",
                    "marker_id": "m1",
                    "marker": "marker 1",
                    "verdict": "pass",
                    "score": 5,
                    "observation": "obs",
                    "evidence": "ev",
                    "improvement_hint": "hint",
                }
            ],
            "strengths": ["s1"],
            "concerns": ["c1"],
            "suggested_improvements": ["i1"],
            "repaired": False,
            "derived_fields": [],
        },
    ]
    call_count = {"n": 0}

    def fake_soft_eval(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return assessments[idx]

    monkeypatch.setattr(eval_mod, "evaluate_training_plan_soft", fake_soft_eval)

    report, _ = evaluate_training_plan_quality_file(
        str(plan_path),
        cfg=ConstraintConfig(min_signal_ids_per_day=0),
        soft_eval_cfg=SoftEvalConfig(enabled=True),
        soft_eval_runs=2,
    )

    assert report["stats"]["inter_rater_runs"] == 2
    assert "m1" in report["stats"]["high_variance_markers"]
    assert report["soft_assessment"]["inter_rater_runs"] == 2
    assert report["soft_assessment"]["inter_rater_variance"]["m1"] > 0.5
