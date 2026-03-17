from typing import Any, cast

from trailtraining.llm.constraints import (
    ConstraintConfig,
    _citation_value,
    _env_bool,
    _env_float,
    _env_int,
    _extract_forecast_context,
    _forecast_reason_text,
    _is_sparse_capability,
    constraint_config_from_env,
    evaluate_training_plan_quality,
    score_from_violations,
)


def test_env_parsers_handle_missing_blank_valid_and_invalid(monkeypatch):
    monkeypatch.delenv("TT_FLOAT", raising=False)
    assert _env_float("TT_FLOAT", 1.5) == 1.5
    monkeypatch.setenv("TT_FLOAT", " ")
    assert _env_float("TT_FLOAT", 1.5) == 1.5
    monkeypatch.setenv("TT_FLOAT", "2.5")
    assert _env_float("TT_FLOAT", 1.5) == 2.5
    monkeypatch.setenv("TT_FLOAT", "oops")
    assert _env_float("TT_FLOAT", 1.5) == 1.5

    monkeypatch.delenv("TT_INT", raising=False)
    assert _env_int("TT_INT", 7) == 7
    monkeypatch.setenv("TT_INT", "")
    assert _env_int("TT_INT", 7) == 7
    monkeypatch.setenv("TT_INT", "11")
    assert _env_int("TT_INT", 7) == 11
    monkeypatch.setenv("TT_INT", "bad")
    assert _env_int("TT_INT", 7) == 7

    monkeypatch.delenv("TT_BOOL", raising=False)
    assert _env_bool("TT_BOOL", True) is True
    monkeypatch.setenv("TT_BOOL", " ")
    assert _env_bool("TT_BOOL", False) is False
    monkeypatch.setenv("TT_BOOL", "YES")
    assert _env_bool("TT_BOOL", False) is True
    monkeypatch.setenv("TT_BOOL", "no")
    assert _env_bool("TT_BOOL", True) is False


def test_constraint_config_from_env_prefers_explicit_args(monkeypatch):
    monkeypatch.setenv("TRAILTRAINING_MAX_RAMP_PCT", "99")
    monkeypatch.setenv("TRAILTRAINING_MAX_CONSEC_HARD", "8")
    monkeypatch.setenv("TRAILTRAINING_MAX_HARD_PER_7D", "5")
    monkeypatch.setenv("TRAILTRAINING_WEEKLY_TIME_TOLERANCE_PCT", "12.5")
    monkeypatch.setenv("TRAILTRAINING_REQUIRE_REST_SESSION_TYPE", "off")

    cfg = constraint_config_from_env(max_ramp_pct=14.5, max_consecutive_hard=4)

    assert cfg.max_ramp_pct == 14.5
    assert cfg.max_consecutive_hard == 4
    assert cfg.max_hard_per_7d == 5
    assert cfg.weekly_time_tolerance_pct == 12.5
    assert cfg.require_rest_session_type is False


def test_citation_helpers_extract_and_normalize_forecast_context():
    plan_obj = {
        "citations": [
            {"signal_id": "forecast.readiness.status", "value": " Fatigued "},
            {"signal_id": "forecast.overreach_risk.level", "quote": " HIGH "},
            {"signal_id": "forecast.recovery_capability.key", "text": "Load_Only"},
            {
                "signal_id": "forecast.recovery_capability.label",
                "value": " I have load + sleep only ",
            },
        ]
    }

    assert _citation_value(plan_obj, "forecast.readiness.status") == " Fatigued "
    assert _citation_value(plan_obj, "forecast.overreach_risk.level") == " HIGH "
    assert _citation_value(plan_obj, "forecast.recovery_capability.key") == "Load_Only"
    assert (
        _citation_value(plan_obj, "forecast.recovery_capability.label")
        == " I have load + sleep only "
    )
    assert _citation_value(plan_obj, "missing.signal") is None

    ctx = _extract_forecast_context(plan_obj)
    assert ctx == {
        "readiness_status": "fatigued",
        "overreach_risk_level": "high",
        "recovery_capability_key": "load_only",
        "recovery_capability_label": "I have load + sleep only",
    }


def test_sparse_capability_and_forecast_reason_text_branches():
    assert _is_sparse_capability({"recovery_capability_key": "load_only"}) is True
    assert (
        _is_sparse_capability({"recovery_capability_label": "We only have training data right now"})
        is True
    )
    assert _is_sparse_capability({"recovery_capability_label": "I have load + sleep only"}) is True
    assert _is_sparse_capability({"recovery_capability_key": "full_stack"}) is False

    assert _forecast_reason_text({}) == "current forecast context"

    reason_text = _forecast_reason_text(
        {
            "readiness_status": "fatigued",
            "overreach_risk_level": "high",
            "recovery_capability_label": "I have load + sleep only",
        }
    )
    assert reason_text == (
        "readiness is fatigued; overreach risk is high; recovery telemetry is sparse"
    )


def test_score_from_violations_handles_bad_penalties_and_default_category():
    violations = cast(
        list[dict[str, Any]],
        [
            {"penalty": "oops", "category": "structure", "severity": "medium"},
            "not-a-dict",
            {"severity": "low"},
        ],
    )
    report = score_from_violations(violations)

    assert report["score"] == 87
    assert report["grade"] == "B"
    assert report["subscores"] == {"structure": 90, "other": 97}


def test_forecast_specific_limits_trigger_with_high_risk_and_sparse_context():
    plan_obj = {
        "plan": {
            "weekly_totals": {"planned_moving_time_hours": 15.0},
            "days": [
                {
                    "date": "2026-03-01",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 180,
                },
                {
                    "date": "2026-03-02",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 180,
                },
                {
                    "date": "2026-03-03",
                    "is_hard_day": True,
                    "is_rest_day": False,
                    "duration_minutes": 120,
                },
                {
                    "date": "2026-03-04",
                    "is_hard_day": False,
                    "is_rest_day": False,
                    "duration_minutes": 120,
                },
                {
                    "date": "2026-03-05",
                    "is_hard_day": False,
                    "is_rest_day": False,
                    "duration_minutes": 120,
                },
                {
                    "date": "2026-03-06",
                    "is_hard_day": False,
                    "is_rest_day": False,
                    "duration_minutes": 90,
                },
                {
                    "date": "2026-03-07",
                    "is_hard_day": False,
                    "is_rest_day": True,
                    "duration_minutes": 90,
                },
            ],
        },
        "citations": [
            {"signal_id": "forecast.readiness.status", "value": "Fatigued"},
            {"signal_id": "forecast.overreach_risk.level", "value": "High"},
            {"signal_id": "forecast.recovery_capability.key", "value": "load_only"},
        ],
    }
    rollups = {"windows": {"7": {"activities": {"total_moving_time_hours": 10.0}}}}
    cfg = ConstraintConfig(
        max_ramp_pct=100.0,
        max_consecutive_hard=99,
        max_hard_per_7d=99,
        min_rest_per_7d=0,
        min_signal_ids_per_day=0,
        weekly_time_tolerance_pct=999.0,
        rest_day_max_minutes=999,
        require_rest_session_type=False,
    )

    report = evaluate_training_plan_quality(plan_obj, rollups, cfg)
    codes = {v["code"] for v in report["violations"]}

    assert {
        "FORECAST_RAMP_TOO_AGGRESSIVE",
        "FORECAST_HARD_DAY_LIMIT",
        "FORECAST_CONSEC_HARD_LIMIT",
    }.issubset(codes)
    assert report["stats"]["effective_max_hard_per_7d"] == 1
    assert report["stats"]["effective_max_consecutive_hard"] == 1
    assert report["stats"]["effective_max_ramp_pct"] == 0.0
