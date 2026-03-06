# src/trailtraining/tests/test_coach_constraints.py
from trailtraining.llm.constraints import ConstraintConfig, validate_training_plan


def test_constraints_ramp_and_consecutive_hard():
    # last 7 = 10h, planned = 12h -> +20% (violates max 10%)
    rollups = {"windows": {"7": {"activities": {"total_moving_time_hours": 10.0}}}}

    plan_obj = {
        "plan": {
            "weekly_totals": {
                "planned_distance_km": 60.0,
                "planned_moving_time_hours": 12.0,
                "planned_elevation_m": 2000.0,
            },
            "days": [
                {"date": "2026-03-01", "is_hard_day": True, "is_rest_day": False, "duration_minutes": 60, "session_type": "tempo", "signal_ids": []},
                {"date": "2026-03-02", "is_hard_day": True, "is_rest_day": False, "duration_minutes": 60, "session_type": "intervals", "signal_ids": []},
                {"date": "2026-03-03", "is_hard_day": True, "is_rest_day": False, "duration_minutes": 60, "session_type": "hills", "signal_ids": []},
            ],
        }
    }

    cfg = ConstraintConfig(max_ramp_pct=10.0, max_consecutive_hard=2)
    v = validate_training_plan(plan_obj, rollups, cfg)
    codes = {x.get("code") for x in v}
    assert "MAX_RAMP_PCT" in codes
    assert "TOO_MANY_CONSEC_HARD" in codes