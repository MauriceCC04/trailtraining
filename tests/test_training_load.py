"""tests/test_training_load.py — unit tests for trailtraining.metrics.training_load"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest
from trailtraining.metrics.training_load import (
    build_atl_ctl_tsb_series,
    latest_atl_ctl_tsb,
)


def _day(
    date_str: str,
    *,
    moving_time: int = 3600,
    avg_hr: float | None = None,
    max_hr: float | None = None,
) -> dict[str, Any]:
    act: dict[str, Any] = {
        "id": hash(date_str),
        "sport_type": "Run",
        "distance": 10000,
        "moving_time": moving_time,
        "total_elevation_gain": 100,
    }
    if avg_hr is not None:
        act["average_heartrate"] = avg_hr
    if max_hr is not None:
        act["max_heartrate"] = max_hr
    return {"date": date_str, "sleep": None, "activities": [act]}


def _rest_day(date_str: str) -> dict[str, Any]:
    return {"date": date_str, "sleep": None, "activities": []}


class TestAtlCtlTsb:
    def test_returns_empty_series_for_no_combined(self) -> None:
        assert build_atl_ctl_tsb_series([]) == []

    def test_latest_returns_none_for_no_combined(self) -> None:
        assert latest_atl_ctl_tsb([]) is None

    def test_steady_load_converges_atl_and_ctl(self) -> None:
        combined: list[dict[str, Any]] = []

        start = date(2026, 1, 1)
        for i in range(90):
            d = start + timedelta(days=i)
            combined.append(_day(d.isoformat(), moving_time=3600))  # 1.0 load-h/day

        latest = latest_atl_ctl_tsb(combined)
        assert latest is not None

        assert latest["atl_load_h"] == pytest.approx(1.0, abs=0.05)
        assert latest["ctl_load_h"] == pytest.approx(1.0, abs=0.10)
        assert latest["tsb_load_h"] == pytest.approx(0.0, abs=0.10)

    def test_ctl_rises_slowly_when_series_starts_with_rest(self) -> None:
        combined: list[dict[str, Any]] = [_rest_day("2026-01-01")]

        start = date(2026, 1, 2)
        for i in range(90):
            d = start + timedelta(days=i)
            combined.append(_day(d.isoformat(), moving_time=3600))

        latest = latest_atl_ctl_tsb(combined)
        assert latest is not None
        assert latest["ctl_load_h"] == pytest.approx(0.883, abs=0.02)

    def test_hard_day_pushes_atl_above_ctl_and_tsb_negative(self) -> None:
        combined: list[dict[str, Any]] = []

        start = date(2026, 1, 1)
        for i in range(60):
            d = start + timedelta(days=i)
            combined.append(_day(d.isoformat(), moving_time=3600))  # steady 1.0 load-h/day

        combined.append(_day("2026-03-02", moving_time=5 * 3600))  # hard 5.0 load-h day

        latest = latest_atl_ctl_tsb(combined)
        assert latest is not None

        assert latest["atl_load_h"] > latest["ctl_load_h"]
        assert latest["tsb_load_h"] < 0

    def test_series_contains_expected_keys(self) -> None:
        combined = [
            _day("2026-03-01", moving_time=3600),
            _day("2026-03-02", moving_time=7200),
        ]
        series = build_atl_ctl_tsb_series(combined)

        assert len(series) == 2
        row = series[-1]
        assert row.date == "2026-03-02"
        assert isinstance(row.load_h, float)
        assert isinstance(row.atl_load_h, float)
        assert isinstance(row.ctl_load_h, float)
        assert isinstance(row.tsb_load_h, float)

    def test_latest_includes_metadata(self) -> None:
        combined = [_day("2026-03-01", moving_time=3600)]
        latest = latest_atl_ctl_tsb(combined)

        assert latest is not None
        assert latest["metric"] == "training_load_hours"
        assert latest["unit"] == "load_h"
        assert latest["tau_atl_days"] == 7.0
        assert latest["tau_ctl_days"] == 42.0
