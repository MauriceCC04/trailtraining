from trailtraining.llm.constraints import (
    ConstraintConfig,
    evaluate_training_plan_quality,
    score_from_violations,
    validate_training_plan,
)


def test_constraints_ramp_and_consecutive_hard():
    # last 7 = 10h, actual first 7 planned = 12h -> +20% (violates max 10%)
    rollups = {"windows": {"7": {"activities": {"total_moving_time_hours": 10.0}}}}

    plan_obj = {
        "plan": {
            "weekly_totals": {
                "planned_distance_km": 60.0,
                "planned_moving_time_hours": 12.0,
                "planned_elevation_m": 2000.0,
            },
            "days": [
                {
                    "date": "2026-03-01",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 240,
                    "session_type": "tempo",
                    "signal_ids": [],
                },
                {
                    "date": "2026-03-02",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 240,
                    "session_type": "intervals",
                    "signal_ids": [],
                },
                {
                    "date": "2026-03-03",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 240,
                    "session_type": "hills",
                    "signal_ids": [],
                },
            ],
        }
    }

    cfg = ConstraintConfig(max_ramp_pct=10.0, max_consecutive_hard=2)
    v = validate_training_plan(plan_obj, rollups, cfg)
    codes = {x.get("code") for x in v}
    assert "MAX_RAMP_PCT" in codes
    assert "TOO_MANY_CONSEC_HARD" in codes


def test_quality_eval_hits_structure_safety_and_justification_checks():
    # Make a plan that intentionally violates many of the quality checks
    rollups = {"windows": {"7": {"activities": {"total_moving_time_hours": 10.0}}}}

    plan_obj = {
        "plan": {
            "weekly_totals": {"planned_moving_time_hours": 20.0},
            "days": [
                # BAD_DATE
                {
                    "date": "bad-date",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 60,
                    "session_type": "tempo",
                    "signal_ids": ["s1"],
                },
                # DUPLICATE_DATE + NON_CONSECUTIVE_DATES (no 2026-03-02)
                {
                    "date": "2026-03-01",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 60,
                    "session_type": "tempo",
                    "signal_ids": ["s1"],
                },
                {
                    "date": "2026-03-01",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 60,
                    "session_type": "intervals",
                    "signal_ids": [],
                },
                {
                    "date": "2026-03-03",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 60,
                    "session_type": "hills",
                    "signal_ids": [],
                },
                {
                    "date": "2026-03-04",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 60,
                    "session_type": "tempo",
                    "signal_ids": [],
                },
                {
                    "date": "2026-03-05",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 60,
                    "session_type": "tempo",
                    "signal_ids": [],
                },
                {
                    "date": "2026-03-06",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 60,
                    "session_type": "tempo",
                    "signal_ids": [],
                },
                # REST_DAY_TOO_LONG + REST_DAY_BAD_SESSION_TYPE
                {
                    "date": "2026-03-07",
                    "is_hard_day": False,
                    "is_rest_day": True,
                    "duration_minutes": 60,
                    "session_type": "easy",
                    "signal_ids": [],
                },
            ],
        }
        # No citations => should trigger MISSING_CITATIONS because we used signal_ids
    }

    cfg = ConstraintConfig(
        max_ramp_pct=10.0,
        max_consecutive_hard=2,
        max_hard_per_7d=3,
        min_rest_per_7d=2,  # force NOT_ENOUGH_REST (we only have 1 rest day in first 7)
        min_signal_ids_per_day=1,
        weekly_time_tolerance_pct=30.0,
        rest_day_max_minutes=30,
        require_rest_session_type=True,
    )

    report = evaluate_training_plan_quality(plan_obj, rollups, cfg)
    codes = {v.get("code") for v in report.get("violations", [])}

    expected = {
        "MAX_RAMP_PCT",
        "TOO_MANY_CONSEC_HARD",
        "BAD_DATE",
        "DUPLICATE_DATE",
        "NON_CONSECUTIVE_DATES",
        "WEEKLY_TOTALS_MISMATCH",
        "TOO_MANY_HARD_PER_WEEK",
        "NOT_ENOUGH_REST",
        "REST_DAY_TOO_LONG",
        "REST_DAY_BAD_SESSION_TYPE",
        "MISSING_SIGNAL_IDS",
        "MISSING_CITATIONS",
    }
    assert expected.issubset(codes)


def test_quality_eval_flags_uncited_signal_ids():
    plan_obj = {
        "plan": {
            "weekly_totals": {"planned_moving_time_hours": 1.0},
            "days": [
                {
                    "date": "2026-03-01",
                    "is_hard_day": False,
                    "is_rest_day": False,
                    "duration_minutes": 30,
                    "session_type": "easy",
                    "signal_ids": ["s1", "s2"],
                },
                {
                    "date": "2026-03-02",
                    "is_hard_day": False,
                    "is_rest_day": True,
                    "duration_minutes": 10,
                    "session_type": "rest",
                    "signal_ids": ["s1"],
                },
            ],
        },
        "citations": [{"signal_id": "s1"}],  # s2 is used but not cited
    }

    cfg = ConstraintConfig(
        max_consecutive_hard=99,
        max_hard_per_7d=99,
        min_rest_per_7d=0,
        min_signal_ids_per_day=1,
        weekly_time_tolerance_pct=999.0,
        rest_day_max_minutes=999,
        require_rest_session_type=False,
    )

    report = evaluate_training_plan_quality(plan_obj, rollups=None, cfg=cfg)
    codes = {v.get("code") for v in report.get("violations", [])}
    assert "UNCITED_SIGNAL_IDS" in codes


def test_score_from_violations_grading_edges():
    assert score_from_violations([])["grade"] == "A"
    assert (
        score_from_violations([{"penalty": 250, "category": "safety", "severity": "high"}])["grade"]
        == "F"
    )
