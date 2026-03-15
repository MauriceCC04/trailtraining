from __future__ import annotations

from pathlib import Path

from trailtraining.contracts import EvaluationReportArtifact
from trailtraining.llm.eval import evaluate_training_plan_quality_file
from trailtraining.llm.rubrics import grade_from_score, weighted_score_from_rubric_scores
from trailtraining.llm.soft_eval import SoftEvalConfig
from trailtraining.util.state import save_json


def _write_rollups(path: Path) -> None:
    payload = {
        "windows": {
            "7": {
                "end_date": "2026-03-02",
                "activities": {
                    "total_moving_time_hours": 2.166,
                    "total_training_load_hours": 2.166,
                },
            }
        }
    }
    save_json(path, payload, compact=False)


def _write_training_plan(path: Path) -> None:
    payload = {
        "meta": {
            "today": "2026-03-02",
            "plan_start": "2026-03-03",
            "plan_days": 7,
            "style": "trailrunning",
            "primary_goal": "to become a faster trail runner",
        },
        "snapshot": {
            "last7": {
                "distance_km": "22",
                "moving_time_hours": "2.2",
                "elevation_m": "300",
                "activity_count": "2",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "46",
            },
            "baseline28": {
                "distance_km": "18",
                "moving_time_hours": "1.9",
                "elevation_m": "220",
                "activity_count": "2",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "45",
            },
            "notes": "",
        },
        "readiness": {
            "status": "steady",
            "rationale": "Stable recent load.",
            "signal_ids": ["forecast.readiness.status"],
        },
        "plan": {
            "weekly_totals": {
                "planned_distance_km": 30.0,
                "planned_moving_time_hours": 4.5,
                "planned_elevation_m": 600.0,
            },
            "days": [
                {
                    "date": "2026-03-03",
                    "title": "Easy run",
                    "session_type": "easy",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 45,
                    "target_intensity": "easy",
                    "terrain": "road",
                    "workout": "45 min easy",
                    "purpose": "aerobic maintenance",
                    "signal_ids": ["forecast.readiness.status"],
                }
            ],
        },
        "recovery": {
            "actions": ["Sleep 8h"],
            "signal_ids": ["forecast.readiness.status"],
        },
        "risks": [],
        "data_notes": [],
        "citations": [
            {
                "signal_id": "forecast.readiness.status",
                "source": "readiness_and_risk_forecast.json:result.readiness.status",
                "date_range": "2026-03-02..2026-03-02",
                "value": "steady",
            }
        ],
    }
    save_json(path, payload, compact=False)


def test_weighted_rubric_grade_is_deterministic() -> None:
    scores = {
        "goal_alignment": {"score": 90},
        "plan_coherence": {"score": 80},
        "explanation_quality": {"score": 70},
        "caution_proportionality": {"score": 60},
        "actionability": {"score": 50},
    }
    overall = weighted_score_from_rubric_scores(scores)
    assert overall == 75.0
    assert grade_from_score(overall) == "C"


def test_quality_eval_can_attach_soft_assessment(tmp_path: Path, monkeypatch) -> None:
    _write_training_plan(tmp_path / "coach_brief_training-plan.json")
    _write_rollups(tmp_path / "combined_rollups.json")

    fake_soft = {
        "model": "anthropic/claude-opus-4.6",
        "primary_goal": "to become a faster trail runner",
        "summary": "Mostly coherent but a little generic.",
        "overall_score": 84.0,
        "grade": "B",
        "confidence": "medium",
        "rubric_scores": {
            "goal_alignment": {"score": 88, "reasoning": "Good alignment."},
            "plan_coherence": {"score": 86, "reasoning": "Coherent."},
            "explanation_quality": {"score": 78, "reasoning": "Somewhat generic."},
            "caution_proportionality": {"score": 84, "reasoning": "Reasonable cautions."},
            "actionability": {"score": 82, "reasoning": "Easy to follow."},
        },
        "marker_results": [],
        "strengths": ["Good alignment"],
        "concerns": ["A little generic"],
        "suggested_improvements": ["Make one session purpose more specific"],
    }

    monkeypatch.setattr(
        "trailtraining.llm.eval.evaluate_training_plan_soft",
        lambda plan_obj, deterministic_report, rollups, cfg: fake_soft,
    )

    report, _obj = evaluate_training_plan_quality_file(
        str(tmp_path / "coach_brief_training-plan.json"),
        rollups_path=str(tmp_path / "combined_rollups.json"),
        soft_eval_cfg=SoftEvalConfig(enabled=True),
    )

    artifact = EvaluationReportArtifact.model_validate(report)
    assert artifact.soft_assessment is not None
    assert artifact.soft_assessment.grade == "B"
    assert artifact.soft_assessment.model == "anthropic/claude-opus-4.6"
