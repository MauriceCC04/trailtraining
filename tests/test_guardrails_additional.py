from __future__ import annotations

from typing import Any, cast

from trailtraining.llm.constraints import ConstraintConfig
from trailtraining.llm.guardrails import (
    _enforce_max_consecutive_hard,
    _enforce_max_hard_per_7d,
    _reduce_total_minutes,
    apply_eval_coach_guardrails,
    build_eval_constraints_block,
)


def _plan_days() -> list[dict[str, object]]:
    return [
        {
            "date": "2026-03-01",
            "session_type": "easy",
            "is_rest_day": False,
            "is_hard_day": False,
            "duration_minutes": 60,
        },
        {
            "date": "2026-03-02",
            "session_type": "tempo",
            "is_rest_day": False,
            "is_hard_day": True,
            "duration_minutes": 60,
        },
        {
            "date": "2026-03-03",
            "session_type": "intervals",
            "is_rest_day": False,
            "is_hard_day": True,
            "duration_minutes": 60,
        },
        {
            "date": "2026-03-04",
            "session_type": "hills",
            "is_rest_day": False,
            "is_hard_day": True,
            "duration_minutes": 60,
        },
        {
            "date": "2026-03-05",
            "session_type": "easy",
            "is_rest_day": False,
            "is_hard_day": False,
            "duration_minutes": 60,
        },
        {
            "date": "2026-03-06",
            "session_type": "long",
            "is_rest_day": False,
            "is_hard_day": False,
            "duration_minutes": 120,
        },
        {
            "date": "2026-03-07",
            "session_type": "rest",
            "is_rest_day": True,
            "is_hard_day": False,
            "duration_minutes": 0,
        },
    ]


def test_reduce_total_minutes_prefers_easy_days_before_hard_days() -> None:
    days = _plan_days()
    remaining = _reduce_total_minutes(days, 40)

    assert remaining == 0
    assert days[0]["duration_minutes"] == 30
    assert days[1]["duration_minutes"] == 60
    assert days[2]["duration_minutes"] == 60


def test_enforce_max_hard_per_7d_downgrades_one_hard_day() -> None:
    days = _plan_days()
    changed = _enforce_max_hard_per_7d(days, max_hard=2)

    assert len(changed) == 1
    assert sum(1 for day in days if day["is_hard_day"]) == 2


def test_enforce_max_consecutive_hard_breaks_streak() -> None:
    days = _plan_days()
    changed = _enforce_max_consecutive_hard(days, max_consec=2)

    assert len(changed) == 1
    streaks: list[int] = []
    current = 0
    for day in days:
        if day["is_hard_day"] and not day["is_rest_day"]:
            current += 1
        else:
            if current:
                streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    assert max(streaks, default=0) <= 2


def test_build_eval_constraints_block_handles_missing_rollups(monkeypatch) -> None:
    import trailtraining.llm.guardrails as guardrails

    monkeypatch.setattr(
        guardrails,
        "_get_cfg",
        lambda: ConstraintConfig(
            max_ramp_pct=12.0, max_consecutive_hard=2, max_hard_per_7d=3, min_rest_per_7d=1
        ),
    )

    text = build_eval_constraints_block(None)

    assert "rollups last7 hours unavailable" in text
    assert "NEVER exceed 2 consecutive hard days" in text
    assert "hard days in any 7-day chunk MUST be <= 3" in text


def test_apply_eval_coach_guardrails_clamps_rest_and_enforces_ramp(monkeypatch) -> None:
    import trailtraining.llm.guardrails as guardrails

    plan_obj: dict[str, Any] = {
        "plan": {
            "weekly_totals": {
                "planned_distance_km": 70.0,
                "planned_moving_time_hours": 8.0,
                "planned_elevation_m": 1500.0,
            },
            "days": [
                {
                    "date": "2026-03-01",
                    "session_type": "easy",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 90,
                },
                {
                    "date": "2026-03-02",
                    "session_type": "tempo",
                    "is_rest_day": False,
                    "is_hard_day": True,
                    "duration_minutes": 70,
                },
                {
                    "date": "2026-03-03",
                    "session_type": "easy",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 90,
                },
                {
                    "date": "2026-03-04",
                    "session_type": "rest",
                    "is_rest_day": True,
                    "is_hard_day": False,
                    "duration_minutes": 90,
                },
                {
                    "date": "2026-03-05",
                    "session_type": "long",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 120,
                },
                {
                    "date": "2026-03-06",
                    "session_type": "easy",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 60,
                },
                {
                    "date": "2026-03-07",
                    "session_type": "strength",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 60,
                },
            ],
        },
        "data_notes": [],
    }
    rollups = {"windows": {"7": {"activities": {"total_moving_time_hours": 5.0}}}}
    cfg = ConstraintConfig(
        max_ramp_pct=0.0,
        max_consecutive_hard=2,
        max_hard_per_7d=3,
        min_rest_per_7d=1,
        rest_day_max_minutes=30,
        require_rest_session_type=True,
    )
    monkeypatch.setattr(guardrails, "_get_cfg", lambda: cfg)

    apply_eval_coach_guardrails(plan_obj, rollups)

    days = cast(list[dict[str, Any]], plan_obj["plan"]["days"])
    weekly_totals = cast(dict[str, float], plan_obj["plan"]["weekly_totals"])
    data_notes = cast(list[str], plan_obj["data_notes"])

    assert days[3]["is_rest_day"] is True
    assert days[3]["session_type"] == "rest"
    assert days[3]["duration_minutes"] == 0
    assert float(weekly_totals["planned_moving_time_hours"]) <= 5.0
    assert float(weekly_totals["planned_distance_km"]) < 70.0
    assert float(weekly_totals["planned_elevation_m"]) < 1500.0
    assert any("Guardrails: enforced ramp rate" in note for note in data_notes)


def test_apply_eval_coach_guardrails_without_rollups_only_normalizes_rest(monkeypatch) -> None:
    import trailtraining.llm.guardrails as guardrails

    plan_obj: dict[str, Any] = {
        "plan": {
            "weekly_totals": {
                "planned_distance_km": 20.0,
                "planned_moving_time_hours": 3.0,
                "planned_elevation_m": 300.0,
            },
            "days": [
                {
                    "date": "2026-03-01",
                    "session_type": "easy",
                    "is_rest_day": True,
                    "is_hard_day": False,
                    "duration_minutes": 50,
                },
            ],
        },
        "data_notes": [],
    }
    cfg = ConstraintConfig(rest_day_max_minutes=25)
    monkeypatch.setattr(guardrails, "_get_cfg", lambda: cfg)

    apply_eval_coach_guardrails(plan_obj, None)

    days = cast(list[dict[str, Any]], plan_obj["plan"]["days"])
    data_notes = cast(list[str], plan_obj["data_notes"])
    day = days[0]

    assert day["session_type"] == "rest"
    assert day["duration_minutes"] == 25
    assert not any("enforced ramp rate" in note for note in data_notes)
