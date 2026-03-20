"""tests/test_signals.py — unit tests for trailtraining.llm.signals"""

from __future__ import annotations

from typing import Any

import pytest
from trailtraining.llm.signals import (
    build_retrieval_context,
    build_signal_registry,
    build_weekly_history,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _day(
    date_str: str,
    *,
    distance: float = 10000,
    moving_time: int = 3600,
    elev: float = 100,
    avg_hr: float | None = 150,
    max_hr: float | None = 180,
    sport: str = "Run",
    sleep_secs: int | None = None,
    rhr: int | None = None,
    hrv: int | None = None,
) -> dict[str, Any]:
    act: dict[str, Any] = {
        "id": hash(date_str + sport),
        "sport_type": sport,
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


# ---------------------------------------------------------------------------
# build_weekly_history
# ---------------------------------------------------------------------------


class TestBuildWeeklyHistory:
    def test_returns_empty_for_no_combined(self) -> None:
        assert build_weekly_history([], weeks=4) == []

    def test_skips_invalid_dates(self) -> None:
        combined = [
            {"date": "bad-date", "activities": [], "sleep": None},
            _day("2026-03-01"),
        ]
        result = build_weekly_history(combined, weeks=4)
        assert len(result) == 1  # only one valid week

    def test_groups_days_by_iso_week(self) -> None:
        # 2026-03-02 is a Monday (week 10), 2026-03-07 is Saturday (same week)
        # 2026-03-09 is a Monday (week 11)
        combined = [
            _day("2026-03-02", distance=10000),
            _day("2026-03-07", distance=5000),
            _day("2026-03-09", distance=8000),
        ]
        result = build_weekly_history(combined, weeks=10)
        assert len(result) == 2
        # Sort by iso_week
        weeks_sorted = sorted(result, key=lambda w: w["iso_week"])
        assert weeks_sorted[0]["distance_km"] == pytest.approx(15.0)
        assert weeks_sorted[1]["distance_km"] == pytest.approx(8.0)

    def test_each_entry_has_required_keys(self) -> None:
        combined = [_day("2026-03-01")]
        result = build_weekly_history(combined, weeks=4)
        assert len(result) == 1
        entry = result[0]
        for key in (
            "iso_week",
            "date_range",
            "distance_km",
            "moving_time_hours",
            "elevation_m",
            "training_load_hours",
            "sleep_hours_mean",
            "hrv_mean",
            "rhr_mean",
            "days_with_sleep",
        ):
            assert key in entry

    def test_limits_to_requested_weeks(self) -> None:
        # Build 3 weeks of data, request only 2
        combined = []
        # Week A: 2026-03-02..2026-03-06
        for i in range(5):
            from datetime import date, timedelta

            d = date(2026, 3, 2) + timedelta(days=i)
            combined.append(_day(d.isoformat()))
        # Week B: 2026-03-09..2026-03-13
        for i in range(5):
            from datetime import date, timedelta

            d = date(2026, 3, 9) + timedelta(days=i)
            combined.append(_day(d.isoformat()))
        # Week C: 2026-03-16..2026-03-20
        for i in range(5):
            from datetime import date, timedelta

            d = date(2026, 3, 16) + timedelta(days=i)
            combined.append(_day(d.isoformat()))

        result = build_weekly_history(combined, weeks=2)
        assert len(result) == 2
        # The oldest week (A) should be excluded
        iso_weeks = [r["iso_week"] for r in result]
        assert "2026-W10" not in iso_weeks  # week A

    def test_computes_sleep_hrv_rhr_means(self) -> None:
        combined = [
            _day("2026-03-02", sleep_secs=28800, rhr=50, hrv=65),
            _day("2026-03-03", sleep_secs=27000, rhr=52, hrv=60),
        ]
        result = build_weekly_history(combined, weeks=4)
        assert len(result) == 1
        entry = result[0]
        assert entry["sleep_hours_mean"] == pytest.approx(
            28800 / 3600 / 2 + 27000 / 3600 / 2, abs=0.1
        )
        assert entry["rhr_mean"] == pytest.approx(51.0)
        assert entry["hrv_mean"] == pytest.approx(62.5)
        assert entry["days_with_sleep"] == 2

    def test_none_for_missing_sleep_values(self) -> None:
        combined = [_day("2026-03-01")]  # no sleep, rhr, hrv
        result = build_weekly_history(combined, weeks=4)
        entry = result[0]
        assert entry["sleep_hours_mean"] is None
        assert entry["rhr_mean"] is None
        assert entry["hrv_mean"] is None
        assert entry["days_with_sleep"] == 0


# ---------------------------------------------------------------------------
# build_signal_registry
# ---------------------------------------------------------------------------


def _make_rollups(
    last7_distance: float = 100.0,
    last7_moving: float = 5.0,
    last7_elev: float = 200.0,
    last7_load: float = 10.0,
    last7_count: int = 4,
    last7_sleep: int = 0,
    w7_start: str = "2026-03-11",
    w7_end: str = "2026-03-17",
    w28_start: str = "2026-02-18",
    w28_end: str = "2026-03-17",
) -> dict[str, Any]:
    return {
        "windows": {
            "7": {
                "start_date": w7_start,
                "end_date": w7_end,
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
                "start_date": w28_start,
                "end_date": w28_end,
                "activities": {
                    "count": 24,
                    "total_distance_km": 500.0,
                    "total_elevation_m": 5000.0,
                    "total_moving_time_hours": 30.0,
                    "total_training_load_hours": 70.0,
                },
            },
        }
    }


class TestBuildSignalRegistry:
    def test_uses_rollups_when_present(self) -> None:
        combined = [_day("2026-03-17")]
        rollups = _make_rollups()
        reg = build_signal_registry(combined, rollups)
        signal_ids = {r["signal_id"] for r in reg}
        assert "load.last7.distance_km" in signal_ids
        assert "load.last7.moving_time_hours" in signal_ids
        assert "load.last7.training_load_hours" in signal_ids
        assert "load.baseline28.distance_km" in signal_ids

    def test_still_emits_recovery_signals_without_rollups(self) -> None:
        combined = [_day("2026-03-17", sleep_secs=28800, rhr=50)]
        reg = build_signal_registry(combined, None)
        signal_ids = {r["signal_id"] for r in reg}
        assert "recovery.last7.sleep_hours_mean" in signal_ids
        assert "recovery.last7.rhr_mean" in signal_ids
        assert "recovery.last7.hrv_mean" in signal_ids

    def test_returns_empty_for_no_combined(self) -> None:
        assert build_signal_registry([], None) == []

    def test_returns_empty_for_combined_with_no_date(self) -> None:
        combined = [{"no_date": True, "activities": [], "sleep": None}]
        assert build_signal_registry(combined, None) == []

    def test_signal_values_sourced_from_rollups(self) -> None:
        combined = [_day("2026-03-17")]
        rollups = _make_rollups(last7_distance=118.242, last7_moving=5.124)
        reg = build_signal_registry(combined, rollups)
        by_id = {r["signal_id"]: r for r in reg}
        assert by_id["load.last7.distance_km"]["value"] == 118.242
        assert by_id["load.last7.moving_time_hours"]["value"] == 5.124

    def test_recovery_values_computed_from_combined(self) -> None:
        combined = [
            _day("2026-03-11", sleep_secs=28800, rhr=50, hrv=65),
            _day("2026-03-12", sleep_secs=27000, rhr=52, hrv=60),
            _day("2026-03-17"),
        ]
        reg = build_signal_registry(combined, None)
        by_id = {r["signal_id"]: r for r in reg}
        # Both sleep days are within last 7 of 2026-03-17
        rhr_mean = by_id["recovery.last7.rhr_mean"]["value"]
        assert rhr_mean == pytest.approx(51.0)


# ---------------------------------------------------------------------------
# build_retrieval_context
# ---------------------------------------------------------------------------


class TestBuildRetrievalContext:
    def test_returns_both_sections(self) -> None:
        combined = [_day("2026-03-17")]
        ctx = build_retrieval_context(combined, None, retrieval_weeks=4)
        assert "weekly_history" in ctx
        assert "signal_registry" in ctx

    def test_weekly_history_is_list(self) -> None:
        combined = [_day("2026-03-17")]
        ctx = build_retrieval_context(combined, None, retrieval_weeks=4)
        assert isinstance(ctx["weekly_history"], list)

    def test_signal_registry_is_list(self) -> None:
        combined = [_day("2026-03-17")]
        ctx = build_retrieval_context(combined, None, retrieval_weeks=4)
        assert isinstance(ctx["signal_registry"], list)
