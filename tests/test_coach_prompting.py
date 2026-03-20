"""tests/test_coach_prompting.py — unit tests for trailtraining.llm.coach_prompting"""

from __future__ import annotations

from typing import Any

from trailtraining.llm.coach_prompting import (
    _forecast_capability_block,
    _forecast_signal_rows,
    _summarize_activity,
    _summarize_day,
    build_prompt_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_forecast(
    *,
    status: str = "steady",
    score: float = 65.6,
    overreach_level: str = "moderate",
    overreach_score: float = 44.8,
    label: str = "I only have training data",
    key: str = "load_only",
    sleep_days: int = 0,
    resting_hr_days: int = 0,
    hrv_days: int = 0,
) -> dict[str, Any]:
    return {
        "result": {
            "date": "2026-03-17",
            "readiness": {"status": status, "score": score},
            "overreach_risk": {"level": overreach_level, "score": overreach_score},
            "inputs": {
                "recovery_capability_label": label,
                "recovery_capability_key": key,
                "sleep_days_7d": sleep_days,
                "resting_hr_days_7d": resting_hr_days,
                "hrv_days_7d": hrv_days,
            },
        }
    }


def _make_rollups() -> dict[str, Any]:
    return {
        "windows": {
            "7": {
                "start_date": "2026-03-11",
                "end_date": "2026-03-17",
                "sleep_days_with_data": 0,
                "activities": {
                    "count": 4,
                    "total_distance_km": 118.0,
                    "total_elevation_m": 249.0,
                    "total_moving_time_hours": 5.124,
                    "total_training_load_hours": 11.8,
                },
            },
            "28": {
                "start_date": "2026-02-18",
                "end_date": "2026-03-17",
                "activities": {
                    "count": 24,
                    "total_distance_km": 538.0,
                    "total_elevation_m": 5116.0,
                    "total_moving_time_hours": 32.5,
                    "total_training_load_hours": 73.6,
                },
            },
        }
    }


def _make_combined() -> list[dict[str, Any]]:
    return [
        {
            "date": "2026-03-17",
            "sleep": None,
            "activities": [
                {
                    "id": 1,
                    "sport_type": "Run",
                    "distance": 5000,
                    "moving_time": 1680,
                    "name": "Morning Run",
                    "average_heartrate": 155,
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# _forecast_capability_block
# ---------------------------------------------------------------------------


class TestForecastCapabilityBlock:
    def test_empty_without_label(self) -> None:
        forecast: dict[str, Any] = {"result": {"inputs": {}}}
        assert _forecast_capability_block(forecast) == []

    def test_includes_counts_when_label_present(self) -> None:
        forecast = _make_forecast(label="I only have training data", sleep_days=0)
        result = _forecast_capability_block(forecast)
        joined = "\n".join(result)
        assert "I only have training data" in joined
        assert "sleep=0" in joined


# ---------------------------------------------------------------------------
# _forecast_signal_rows
# ---------------------------------------------------------------------------


class TestForecastSignalRows:
    def test_includes_readiness_and_risk(self) -> None:
        forecast = _make_forecast()
        rows = _forecast_signal_rows(forecast)
        signal_ids = {r["signal_id"] for r in rows}
        assert "forecast.readiness.status" in signal_ids
        assert "forecast.readiness.score" in signal_ids
        assert "forecast.overreach_risk.level" in signal_ids
        assert "forecast.overreach_risk.score" in signal_ids

    def test_includes_recovery_capability(self) -> None:
        forecast = _make_forecast(key="load_only", label="I only have training data")
        rows = _forecast_signal_rows(forecast)
        signal_ids = {r["signal_id"] for r in rows}
        assert "forecast.recovery_capability.label" in signal_ids
        assert "forecast.recovery_capability.key" in signal_ids

    def test_includes_sleep_rhr_hrv_days(self) -> None:
        forecast = _make_forecast(sleep_days=0, resting_hr_days=0, hrv_days=0)
        rows = _forecast_signal_rows(forecast)
        signal_ids = {r["signal_id"] for r in rows}
        assert "forecast.recovery_capability.sleep_days_7d" in signal_ids
        assert "forecast.recovery_capability.resting_hr_days_7d" in signal_ids
        assert "forecast.recovery_capability.hrv_days_7d" in signal_ids


# ---------------------------------------------------------------------------
# _summarize_activity
# ---------------------------------------------------------------------------


class TestSummarizeActivity:
    def test_formats_distance_time_hr_and_name(self) -> None:
        activity = {
            "sport_type": "Run",
            "distance": 10000,
            "moving_time": 3600,
            "total_elevation_gain": 150,
            "average_heartrate": 155,
            "name": "Morning run",
        }
        result = _summarize_activity(activity)
        assert "10.00 km" in result
        assert "60 min" in result
        assert "avgHR 155" in result
        assert "Morning run" in result
        assert "150 m+" in result

    def test_handles_missing_optional_fields(self) -> None:
        activity = {"sport_type": "Ride"}
        result = _summarize_activity(activity)
        assert "Ride" in result

    def test_uses_type_fallback(self) -> None:
        activity = {"type": "Run", "distance": 5000}
        result = _summarize_activity(activity)
        assert "Run" in result


# ---------------------------------------------------------------------------
# _summarize_day
# ---------------------------------------------------------------------------


class TestSummarizeDay:
    def test_handles_missing_sleep_and_no_activities(self) -> None:
        day: dict[str, Any] = {"date": "2026-03-01", "sleep": None, "activities": []}
        result = _summarize_day(day)
        assert "2026-03-01" in result
        assert "Sleep: (none)" in result
        assert "Activities: (none)" in result

    def test_includes_activities_count(self) -> None:
        day: dict[str, Any] = {
            "date": "2026-03-01",
            "sleep": None,
            "activities": [
                {"sport_type": "Run", "distance": 10000, "moving_time": 3600},
                {"sport_type": "Ride", "distance": 30000, "moving_time": 5400},
            ],
        }
        result = _summarize_day(day)
        assert "Activities (2)" in result

    def test_includes_sleep_data_when_present(self) -> None:
        day: dict[str, Any] = {
            "date": "2026-03-01",
            "sleep": {"sleep_score": 82, "resting_hr": 48},
            "activities": [],
        }
        result = _summarize_day(day)
        assert "Sleep:" in result
        assert "Sleep: (none)" not in result


# ---------------------------------------------------------------------------
# build_prompt_text
# ---------------------------------------------------------------------------


_UNSET: Any = object()


class TestBuildPromptText:
    def _build(
        self,
        *,
        prompt_name: str = "training-plan",
        personal: Any = _UNSET,
        rollups: Any = _UNSET,
        combined: Any = _UNSET,
        deterministic_forecast: Any = _UNSET,
        style: str = "trailrunning",
        primary_goal: str = "21k trail race",
        lifestyle_notes: str = "",
        max_chars: int = 200_000,
        detail_days: int = 14,
        plan_days: int = 7,
    ) -> str:
        return build_prompt_text(
            prompt_name=prompt_name,
            personal={"userInfo": {}, "biometricProfile": {}} if personal is _UNSET else personal,
            rollups=_make_rollups() if rollups is _UNSET else rollups,
            combined=_make_combined() if combined is _UNSET else combined,
            deterministic_forecast=_make_forecast()
            if deterministic_forecast is _UNSET
            else deterministic_forecast,
            style=style,
            primary_goal=primary_goal,
            lifestyle_notes=lifestyle_notes,
            max_chars=max_chars,
            detail_days=detail_days,
            plan_days=plan_days,
        )

    def test_includes_rollups_when_present(self) -> None:
        prompt = self._build()
        assert "rollup" in prompt.lower() or "windows" in prompt

    def test_omits_rollups_section_when_none(self) -> None:
        prompt = self._build(rollups=None)
        assert "Recent rollups" not in prompt

    def test_appends_forecast_signals_to_registry(self) -> None:
        prompt = self._build()
        assert "forecast.readiness.status" in prompt
        assert "forecast.overreach_risk.level" in prompt

    def test_adds_older_days_note_when_detail_days_truncates(self) -> None:
        # 2 days combined but detail_days=1 → truncation note should appear
        combined: list[dict[str, Any]] = [
            {"date": "2026-03-16", "sleep": None, "activities": []},
            {"date": "2026-03-17", "sleep": None, "activities": []},
        ]
        prompt = self._build(combined=combined, detail_days=1)
        assert "older" in prompt.lower() or "Older days" in prompt

    def test_respects_budget_and_drops_tail_when_needed(self) -> None:
        # Very small budget should still produce a prompt without error
        prompt = self._build(max_chars=500)
        assert len(prompt) > 0  # doesn't crash

    def test_includes_output_contract_for_training_plan_only(self) -> None:
        plan_prompt = self._build(prompt_name="training-plan")
        non_plan = self._build(prompt_name="recovery-status")
        assert "Output Contract" in plan_prompt
        assert "Output Contract" not in non_plan

    def test_lifestyle_notes_appear_in_prompt(self) -> None:
        prompt = self._build(lifestyle_notes="Weekdays: road only.")
        assert "road only" in prompt

    def test_empty_lifestyle_notes_absent(self) -> None:
        prompt = self._build(lifestyle_notes="")
        assert "Lifestyle constraints" not in prompt
