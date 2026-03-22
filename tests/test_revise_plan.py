from __future__ import annotations

import json
from pathlib import Path

from trailtraining.contracts import TrainingPlanArtifact
from trailtraining.llm.revise import RevisePlanConfig, run_revise_plan
from trailtraining.util.state import save_json


class _FakeResp:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


def _write_plan(path: Path) -> None:
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


def _write_report(path: Path) -> None:
    payload = {
        "score": 92,
        "grade": "A",
        "subscores": {},
        "stats": {},
        "violations": [],
        "soft_assessment": {
            "model": "anthropic/claude-sonnet-4",
            "style": "trailrunning",
            "primary_goal": "to become a faster trail runner",
            "summary": "Good plan with room for more specific purpose wording.",
            "overall_score": 84,
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
        },
    }
    save_json(path, payload, compact=False)


def test_run_revise_plan_writes_json_and_txt(tmp_path: Path, monkeypatch) -> None:
    plan_path = tmp_path / "coach_brief_training-plan.json"
    report_path = tmp_path / "eval_report.json"
    out_path = tmp_path / "revised-plan.json"

    _write_plan(plan_path)
    _write_report(report_path)

    revised = json.loads(plan_path.read_text(encoding="utf-8"))
    revised["plan"]["days"][0]["purpose"] = "Aerobic maintenance with slightly clearer wording."

    monkeypatch.setattr("trailtraining.llm.revise._make_openrouter_client", lambda: object())
    monkeypatch.setattr(
        "trailtraining.llm.revise._call_with_schema",
        lambda client, kwargs, schema: _FakeResp(json.dumps(revised)),
    )
    monkeypatch.setattr(
        "trailtraining.llm.revise.compare_plans",
        lambda *args, **kwargs: {
            "preferred": "plan_b",
            "reasoning": "The revised plan is more specific.",
            "plan_a_advantages": [],
            "plan_b_advantages": ["More specific purpose wording."],
        },
    )

    text, saved = run_revise_plan(
        cfg=RevisePlanConfig(),
        input_plan_path=str(plan_path),
        eval_report_path=str(report_path),
        output_path=str(out_path),
    )

    assert saved == str(out_path)
    assert "Aerobic maintenance with slightly clearer wording." in text

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    artifact = TrainingPlanArtifact.model_validate(payload)
    assert artifact.plan.days[0].purpose == "Aerobic maintenance with slightly clearer wording."

    txt_path = tmp_path / "revised-plan.txt"
    assert txt_path.exists()
    assert "Training Plan" in txt_path.read_text(encoding="utf-8")

    comparison_path = tmp_path / "revised-plan-comparison.json"
    assert comparison_path.exists()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["selected_plan"] == "revised_candidate"


def test_run_revise_plan_keeps_original_when_pairwise_prefers_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan_path = tmp_path / "coach_brief_training-plan.json"
    report_path = tmp_path / "eval_report.json"
    out_path = tmp_path / "revised-plan.json"

    _write_plan(plan_path)
    _write_report(report_path)

    revised = json.loads(plan_path.read_text(encoding="utf-8"))
    revised["plan"]["days"][0]["purpose"] = "New but worse wording."

    monkeypatch.setattr("trailtraining.llm.revise._make_openrouter_client", lambda: object())
    monkeypatch.setattr(
        "trailtraining.llm.revise._call_with_schema",
        lambda client, kwargs, schema: _FakeResp(json.dumps(revised)),
    )
    monkeypatch.setattr(
        "trailtraining.llm.revise.compare_plans",
        lambda *args, **kwargs: {
            "preferred": "plan_a",
            "reasoning": "The original plan is clearer and does not introduce regressions.",
            "plan_a_advantages": ["Cleaner wording."],
            "plan_b_advantages": [],
        },
    )

    run_revise_plan(
        cfg=RevisePlanConfig(),
        input_plan_path=str(plan_path),
        eval_report_path=str(report_path),
        output_path=str(out_path),
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["plan"]["days"][0]["purpose"] == revised["plan"]["days"][0]["purpose"]

    comparison_path = tmp_path / "revised-plan-comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["selected_plan"] == "original"
    assert comparison["preferred"] == "plan_a"
