from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest


class _FakeResp:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


@pytest.fixture()
def base_plan() -> dict[str, Any]:
    return {
        "meta": {
            "today": "2026-03-02",
            "plan_start": "2026-03-03",
            "plan_days": 7,
            "style": "trailrunning",
            "primary_goal": "to become a faster trail runner",
            "lifestyle_notes": "weekday road only",
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
                    "estimated_distance_km": 8.0,
                    "estimated_elevation_m": 50.0,
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
                "citation_id": "cit_forecast.readiness.status",
                "signal_id": "forecast.readiness.status",
                "source": "readiness_and_risk_forecast.json:result.readiness.status",
                "date_range": "2026-03-02..2026-03-02",
                "value": "steady",
            }
        ],
        "claim_attributions": [],
        "effective_constraints": {
            "allowed_week1_hours": 5.0,
            "effective_max_ramp_pct": 10.0,
            "effective_max_hard_per_7d": 3,
            "effective_max_consecutive_hard": 2,
            "min_rest_per_7d": 1,
            "readiness_status": "steady",
            "overreach_risk_level": "low",
            "recovery_capability_key": "load_only",
            "lifestyle_notes": "weekday road only",
            "reasons": ["lifestyle constraints apply"],
        },
    }


@pytest.fixture()
def base_report() -> dict[str, Any]:
    return {
        "score": 97,
        "grade": "A",
        "subscores": {"justification": 97},
        "stats": {},
        "violations": [
            {
                "code": "WEEKLY_TOTALS_MISMATCH",
                "severity": "medium",
                "category": "structure",
                "penalty": 10,
                "message": "Fix weekly totals.",
                "details": {},
            }
        ],
        "soft_assessment": {
            "model": "anthropic/claude-sonnet-4",
            "style": "trailrunning",
            "primary_goal": "to become a faster trail runner",
            "summary": "Good plan with structural fixes needed.",
            "overall_score": 82.9,
            "grade": "B",
            "confidence": "medium",
            "rubric_scores": {},
            "marker_results": [],
            "strengths": ["Good alignment"],
            "concerns": ["Weekly totals are wrong"],
            "suggested_improvements": ["Fix weekly totals to match day objects"],
        },
    }


def test_revise_plan_keeps_revised_artifact_even_if_pairwise_prefers_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_plan: dict,
    base_report: dict,
) -> None:
    import trailtraining.llm.revise as revise

    runtime = SimpleNamespace(paths=SimpleNamespace(prompting_directory=tmp_path))
    plan_path = tmp_path / "coach_brief_training-plan.json"
    report_path = tmp_path / "eval_report.json"
    out_path = tmp_path / "revised-plan.json"
    plan_path.write_text(json.dumps(base_plan), encoding="utf-8")
    report_path.write_text(json.dumps(base_report), encoding="utf-8")

    revised = json.loads(plan_path.read_text(encoding="utf-8"))
    revised["plan"]["days"][0]["purpose"] = "Aerobic maintenance with clearer intent."

    monkeypatch.setattr(revise.config, "current", lambda: runtime)
    monkeypatch.setattr(revise.config, "ensure_directories", lambda runtime=None: None)
    monkeypatch.setattr(revise, "_make_openrouter_client", lambda: object())
    monkeypatch.setattr(
        revise,
        "_call_with_schema",
        lambda client, kwargs, schema: _FakeResp(json.dumps(revised)),
    )
    monkeypatch.setattr(
        revise,
        "compare_plans",
        lambda *args, **kwargs: {
            "preferred": "plan_a",
            "reasoning": "Original plan is slightly clearer overall.",
            "plan_a_advantages": ["Cleaner wording."],
            "plan_b_advantages": ["Fixes totals."],
        },
    )

    text, saved = revise.run_revise_plan(
        cfg=revise.RevisePlanConfig(),
        input_plan_path=str(plan_path),
        eval_report_path=str(report_path),
        output_path=str(out_path),
    )

    assert saved == str(out_path)
    assert "Aerobic maintenance with clearer intent." in text

    revised_payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert (
        revised_payload["plan"]["days"][0]["purpose"] == "Aerobic maintenance with clearer intent."
    )

    selected = json.loads((tmp_path / "selected-plan.json").read_text(encoding="utf-8"))
    assert selected["plan"]["days"][0]["purpose"] == base_plan["plan"]["days"][0]["purpose"]

    comparison = json.loads((tmp_path / "revised-plan-comparison.json").read_text(encoding="utf-8"))
    assert comparison["selected_plan"] == "original"


def test_revise_plan_fails_if_no_change_despite_requested_fixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_plan: dict,
    base_report: dict,
) -> None:
    import trailtraining.llm.revise as revise

    runtime = SimpleNamespace(paths=SimpleNamespace(prompting_directory=tmp_path))
    plan_path = tmp_path / "coach_brief_training-plan.json"
    report_path = tmp_path / "eval_report.json"
    plan_path.write_text(json.dumps(base_plan), encoding="utf-8")
    report_path.write_text(json.dumps(base_report), encoding="utf-8")

    monkeypatch.setattr(revise.config, "current", lambda: runtime)
    monkeypatch.setattr(revise.config, "ensure_directories", lambda runtime=None: None)
    monkeypatch.setattr(revise, "_make_openrouter_client", lambda: object())
    monkeypatch.setattr(
        revise,
        "_call_with_schema",
        lambda client, kwargs, schema: _FakeResp(json.dumps(base_plan)),
    )

    with pytest.raises(RuntimeError, match="no material change"):
        revise.run_revise_plan(
            cfg=revise.RevisePlanConfig(),
            input_plan_path=str(plan_path),
            eval_report_path=str(report_path),
        )


def test_guardrails_canonicalize_rest_day_and_derive_totals() -> None:
    from trailtraining.llm.guardrails import apply_eval_coach_guardrails

    plan_obj: dict[str, Any] = {
        "meta": {"lifestyle_notes": "weekday road only"},
        "plan": {
            "weekly_totals": {
                "planned_distance_km": 20.0,
                "planned_moving_time_hours": 3.0,
                "planned_elevation_m": 300.0,
            },
            "days": [
                {
                    "date": "2026-03-01",
                    "title": "Easy road run",
                    "session_type": "easy",
                    "is_rest_day": True,
                    "is_hard_day": False,
                    "duration_minutes": 40,
                    "target_intensity": "easy",
                    "terrain": "road",
                    "workout": "40 min easy road run",
                    "purpose": "Get a light run done",
                    "signal_ids": ["s1"],
                }
            ],
        },
        "data_notes": [],
    }

    apply_eval_coach_guardrails(plan_obj, None)

    plan = cast(dict[str, Any], plan_obj["plan"])
    day = cast(list[dict[str, Any]], plan["days"])[0]
    weekly = cast(dict[str, Any], plan["weekly_totals"])

    assert day["session_type"] == "rest"
    assert day["duration_minutes"] == 0
    assert day["workout"] == "Rest day. No structured training."
    assert weekly["planned_moving_time_hours"] == 0.0
    assert weekly["planned_distance_km"] is None
    assert weekly["planned_elevation_m"] is None
