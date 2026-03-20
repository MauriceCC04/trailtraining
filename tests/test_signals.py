"""tests/test_signals.py — unit tests for trailtraining.llm.signals"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest
from trailtraining.llm.signals import (
    build_retrieval_context,
    build_signal_registry,
    build_weekly_history,
)


def _day(
    date_str: str,
    *,
    distance: float = 10000.0,
    moving_time: int = 3600,
    elev: float = 100.0,
    avg_hr: float | None = 150.0,
    max_hr: float | None = 180.0,
    sleep_secs: int | None = None,
    rhr: int | None = None,
    hrv: int | None = None,
) -> dict[str, Any]:
    act: dict[str, Any] = {
        "id": hash((date_str, distance, moving_time, elev)),
        "sport_type": "Run",
        "distance": distance,
        "moving_time": moving_time,
        "total_elevation_gain": elev,
    }
    if avg_hr is not None:
        act["average_heartrate"] = avg_hr
    if max_hr is not None:
        act["max_heartrate"] = max_hr

    sleep: dict[str, Any] | None = None
    if sleep_secs is not None or rhr is not None or hrv is not None:
        sleep = {}
        if sleep_secs is not None:
            sleep["sleepTimeSeconds"] = sleep_secs
        if rhr is not None:
            sleep["restingHeartRate"] = rhr
        if hrv is not None:
            sleep["avgOvernightHrv"] = hrv

    return {"date": date_str, "sleep": sleep, "activities": [act]}


def _rest_day(date_str: str) -> dict[str, Any]:
    return {"date": date_str, "sleep": None, "activities": []}


def _make_rollups(
    *,
    last7_distance: float = 100.0,
    last7_moving: float = 5.0,
    last7_elev: float = 200.0,
    last7_load: float = 10.0,
    last7_count: int = 4,
    last7_sleep: int = 0,
    baseline28_distance: float = 500.0,
    baseline28_moving: float = 30.0,
    baseline28_elev: float = 5000.0,
    baseline28_load: float = 70.0,
    include_load_model: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "windows": {
            "7": {
                "start_date": "2026-03-11",
                "end_date": "2026-03-17",
                "sleep_days_with_data": last7_sleep,
                "activities": {
                    "count": last7_count,
                    "total_distance_km": last7_distance,
                    "total_elevation_m": last7_elev,
                    "total_moving_time_hours": last7_moving,
                    "total_training_load_hours": last7_load,
                },
            },
            "28": {
                "start_date": "2026-02-18",
                "end_date": "2026-03-17",
                "activities": {
                    "count": 24,
                    "total_distance_km": baseline28_distance,
                    "total_elevation_m": baseline28_elev,
                    "total_moving_time_hours": baseline28_moving,
                    "total_training_load_hours": baseline28_load,
                },
            },
        }
    }
    if include_load_model:
        out["load_model"] = {
            "atl_load_h": 12.3,
            "ctl_load_h": 10.1,
            "tsb_load_h": -2.2,
            "tau_atl_days": 7.0,
            "tau_ctl_days": 42.0,
        }
    return out


class TestBuildWeeklyHistory:
    def test_returns_empty_for_no_combined(self) -> None:
        assert build_weekly_history([], weeks=4) == []

    def test_returns_empty_when_last_date_invalid(self) -> None:
        combined: list[dict[str, Any]] = [{"date": "bad-date", "activities": [], "sleep": None}]
        assert build_weekly_history(combined, weeks=4) == []

    def test_groups_days_by_iso_week(self) -> None:
        combined: list[dict[str, Any]] = [
            _day("2026-03-02", distance=10000),
            _day("2026-03-07", distance=5000),
            _day("2026-03-09", distance=8000),
        ]

        result = build_weekly_history(combined, weeks=10)

        assert len(result) == 2
        weeks_sorted = sorted(result, key=lambda row: str(row["iso_week"]))
        assert weeks_sorted[0]["distance_km"] == pytest.approx(15.0)
        assert weeks_sorted[1]["distance_km"] == pytest.approx(8.0)

    def test_limits_to_requested_weeks(self) -> None:
        combined: list[dict[str, Any]] = []
        start = date(2026, 3, 2)
        for i in range(15):
            combined.append(_day((start + timedelta(days=i)).isoformat()))

        result = build_weekly_history(combined, weeks=2)
        iso_weeks = [str(row["iso_week"]) for row in result]

        assert len(result) == 2
        assert "2026-W10" not in iso_weeks

    def test_computes_sleep_hrv_rhr_means(self) -> None:
        combined: list[dict[str, Any]] = [
            _day("2026-03-02", sleep_secs=28800, rhr=50, hrv=65),
            _day("2026-03-03", sleep_secs=27000, rhr=52, hrv=60),
        ]

        result = build_weekly_history(combined, weeks=4)

        assert len(result) == 1
        entry = result[0]
        assert entry["sleep_hours_mean"] == pytest.approx(7.75)
        assert entry["rhr_mean"] == pytest.approx(51.0)
        assert entry["hrv_mean"] == pytest.approx(62.5)
        assert entry["days_with_sleep"] == 2

    def test_returns_none_for_missing_sleep_metrics(self) -> None:
        combined: list[dict[str, Any]] = [_day("2026-03-01")]
        result = build_weekly_history(combined, weeks=4)

        assert result[0]["sleep_hours_mean"] is None
        assert result[0]["rhr_mean"] is None
        assert result[0]["hrv_mean"] is None
        assert result[0]["days_with_sleep"] == 0


class TestBuildSignalRegistry:
    def test_returns_empty_for_no_combined(self) -> None:
        assert build_signal_registry([], None) == []

    def test_returns_empty_when_last_date_invalid(self) -> None:
        combined: list[dict[str, Any]] = [{"date": "bad-date", "activities": [], "sleep": None}]
        assert build_signal_registry(combined, None) == []

    def test_uses_rollups_when_present(self) -> None:
        combined: list[dict[str, Any]] = [_day("2026-03-17")]
        reg = build_signal_registry(combined, _make_rollups())

        signal_ids = {str(row["signal_id"]) for row in reg}

        assert "load.last7.distance_km" in signal_ids
        assert "load.last7.moving_time_hours" in signal_ids
        assert "load.last7.training_load_hours" in signal_ids
        assert "load.baseline28.distance_km" in signal_ids

    def test_falls_back_to_combined_summary_when_rollups_missing(self) -> None:
        combined: list[dict[str, Any]] = [
            _day("2026-03-11", distance=10000, moving_time=3600, elev=100),
            _day("2026-03-12", distance=5000, moving_time=1800, elev=50),
            _rest_day("2026-03-13"),
            _day("2026-03-17", distance=8000, moving_time=2400, elev=80),
        ]

        reg = build_signal_registry(combined, None)
        by_id = {str(row["signal_id"]): row for row in reg}

        assert by_id["load.last7.distance_km"]["value"] == pytest.approx(23.0)
        assert by_id["load.last7.moving_time_hours"]["value"] == pytest.approx(
            round(1.0 + 0.5 + (2400 / 3600), 3)
        )
        assert by_id["load.last7.elevation_m"]["value"] == pytest.approx(230.0)
        assert by_id["load.last7.activity_count"]["value"] == 3
        assert by_id["load.last7.sleep_days_with_data"]["value"] == 0

    def test_recovery_values_computed_from_combined(self) -> None:
        combined: list[dict[str, Any]] = [
            _day("2026-03-11", sleep_secs=28800, rhr=50, hrv=65),
            _day("2026-03-12", sleep_secs=27000, rhr=52, hrv=60),
            _day("2026-03-17"),
        ]

        reg = build_signal_registry(combined, None)
        by_id = {str(row["signal_id"]): row for row in reg}

        assert by_id["recovery.last7.sleep_hours_mean"]["value"] == pytest.approx(7.75)
        assert by_id["recovery.last7.rhr_mean"]["value"] == pytest.approx(51.0)
        assert by_id["recovery.last7.hrv_mean"]["value"] == pytest.approx(62.5)

    def test_emits_load_model_signals_from_rollups(self) -> None:
        combined: list[dict[str, Any]] = [_day("2026-03-17")]
        reg = build_signal_registry(combined, _make_rollups(include_load_model=True))
        by_id = {str(row["signal_id"]): row for row in reg}

        assert by_id["load.model.atl_hours"]["value"] == 12.3
        assert by_id["load.model.ctl_hours"]["value"] == 10.1
        assert by_id["load.model.tsb_hours"]["value"] == -2.2
        assert by_id["load.model.atl_tau_days"]["value"] == 7.0
        assert by_id["load.model.ctl_tau_days"]["value"] == 42.0

    def test_computes_load_model_when_rollups_missing(self) -> None:
        combined: list[dict[str, Any]] = []
        start = date(2026, 2, 11)
        for i in range(35):
            combined.append(_day((start + timedelta(days=i)).isoformat(), moving_time=3600))

        reg = build_signal_registry(combined, None)
        by_id = {str(row["signal_id"]): row for row in reg}

        assert "load.model.atl_hours" in by_id
        assert "load.model.ctl_hours" in by_id
        assert "load.model.tsb_hours" in by_id
        assert by_id["load.model.atl_hours"]["value"] is not None
        assert by_id["load.model.ctl_hours"]["value"] is not None
        assert by_id["load.model.tsb_hours"]["value"] is not None


class TestBuildRetrievalContext:
    def test_returns_both_sections(self) -> None:
        combined: list[dict[str, Any]] = [_day("2026-03-17")]
        ctx = build_retrieval_context(combined, None, retrieval_weeks=4)

        assert "weekly_history" in ctx
        assert "signal_registry" in ctx
        assert isinstance(ctx["weekly_history"], list)
        assert isinstance(ctx["signal_registry"], list)

    def test_passes_through_rollup_backed_and_derived_signals(self) -> None:
        combined: list[dict[str, Any]] = [_day("2026-03-17", sleep_secs=28800, rhr=50, hrv=60)]
        ctx = build_retrieval_context(
            combined,
            _make_rollups(include_load_model=True),
            retrieval_weeks=4,
        )

        signal_ids = {str(row["signal_id"]) for row in ctx["signal_registry"]}
        assert "load.last7.distance_km" in signal_ids
        assert "load.model.atl_hours" in signal_ids
        assert "recovery.last7.sleep_hours_mean" in signal_ids
