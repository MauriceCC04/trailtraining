"""tests/test_combine.py — unit tests for trailtraining.data.combine"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from trailtraining.data.combine import (
    _compute_rollup,
    _extract_sleep_date,
    _load_activities_by_date,
    _load_sleep_by_date,
)

# ---------------------------------------------------------------------------
# _extract_sleep_date
# ---------------------------------------------------------------------------


class TestExtractSleepDate:
    @pytest.mark.parametrize(
        "key",
        ["calendarDate", "date", "day", "calendar_date", "id"],
    )
    def test_prefers_supported_top_level_keys(self, key: str) -> None:
        entry = {key: "2026-03-01T12:00:00"}
        result = _extract_sleep_date(entry)
        assert result == "2026-03-01"

    def test_reads_daily_sleep_dto(self) -> None:
        entry = {"dailySleepDTO": {"calendarDate": "2026-03-15T00:00:00"}}
        assert _extract_sleep_date(entry) == "2026-03-15"

    def test_reads_daily_sleep_dto_date_key(self) -> None:
        entry = {"dailySleepDTO": {"date": "2026-04-01"}}
        assert _extract_sleep_date(entry) == "2026-04-01"

    def test_returns_none_when_no_date_found(self) -> None:
        assert _extract_sleep_date({}) is None
        assert _extract_sleep_date({"foo": "bar"}) is None

    def test_returns_none_for_short_string(self) -> None:
        assert _extract_sleep_date({"calendarDate": "abc"}) is None

    def test_top_level_wins_over_dto(self) -> None:
        entry = {
            "calendarDate": "2026-01-01",
            "dailySleepDTO": {"calendarDate": "2026-06-15"},
        }
        assert _extract_sleep_date(entry) == "2026-01-01"


# ---------------------------------------------------------------------------
# _load_sleep_by_date
# ---------------------------------------------------------------------------


class TestLoadSleepByDate:
    def test_accepts_dict_input_with_datetime_keys(self, tmp_path: Path) -> None:
        data = {"2026-03-01T00:00:00": {"sleepTimeSeconds": 28800}}
        p = tmp_path / "sleep.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        result = _load_sleep_by_date(str(p))
        assert "2026-03-01" in result
        assert result["2026-03-01"]["sleepTimeSeconds"] == 28800

    def test_accepts_list_input_and_skips_invalid_items(self, tmp_path: Path) -> None:
        data = [
            {"calendarDate": "2026-03-01", "sleepTimeSeconds": 28000},
            "not-a-dict",
            123,
            {},  # no date key → skipped
            {"calendarDate": "2026-03-02", "sleepTimeSeconds": 29000},
        ]
        p = tmp_path / "sleep.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        result = _load_sleep_by_date(str(p))
        assert set(result.keys()) == {"2026-03-01", "2026-03-02"}

    def test_returns_empty_dict_when_file_missing(self, tmp_path: Path) -> None:
        result = _load_sleep_by_date(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_returns_empty_dict_for_unexpected_type(self, tmp_path: Path) -> None:
        p = tmp_path / "sleep.json"
        p.write_text('"just-a-string"', encoding="utf-8")
        assert _load_sleep_by_date(str(p)) == {}


# ---------------------------------------------------------------------------
# _load_activities_by_date
# ---------------------------------------------------------------------------


class TestLoadActivitiesByDate:
    def _write(self, tmp_path: Path, data: Any) -> str:
        p = tmp_path / "activities.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_groups_multiple_activities_per_day(self, tmp_path: Path) -> None:
        data = [
            {"id": 1, "start_date_local": "2026-03-01T08:00:00", "sport_type": "Run"},
            {"id": 2, "start_date_local": "2026-03-01T17:00:00", "sport_type": "Ride"},
            {"id": 3, "start_date_local": "2026-03-02T09:00:00", "sport_type": "TrailRun"},
        ]
        path = self._write(tmp_path, data)
        result = _load_activities_by_date(path)
        assert len(result["2026-03-01"]) == 2
        assert len(result["2026-03-02"]) == 1

    def test_skips_non_dict_items(self, tmp_path: Path) -> None:
        data = [{"id": 1, "start_date_local": "2026-03-01T08:00:00"}, "bad", None]
        path = self._write(tmp_path, data)
        result = _load_activities_by_date(path)
        assert len(result["2026-03-01"]) == 1

    def test_returns_empty_for_non_list(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"key": "value"})
        assert _load_activities_by_date(path) == {}

    def test_uses_start_date_as_fallback(self, tmp_path: Path) -> None:
        data = [{"id": 5, "start_date": "2026-04-10T06:00:00Z", "sport_type": "Run"}]
        path = self._write(tmp_path, data)
        result = _load_activities_by_date(path)
        assert "2026-04-10" in result


# ---------------------------------------------------------------------------
# _compute_rollup
# ---------------------------------------------------------------------------


class TestComputeRollup:
    def _make_day(
        self,
        d: str,
        *,
        distance: float = 10000,
        moving_time: int = 3600,
        elev: float = 100,
        avg_hr: float | None = 150,
        sport: str = "Run",
        sleep: dict | None = None,
    ) -> dict[str, Any]:
        act: dict[str, Any] = {
            "id": hash(d + sport),
            "sport_type": sport,
            "distance": distance,
            "moving_time": moving_time,
            "total_elevation_gain": elev,
        }
        if avg_hr is not None:
            act["average_heartrate"] = avg_hr
        return {"date": d, "sleep": sleep, "activities": [act]}

    def test_aggregates_totals_by_sport(self) -> None:
        combined = [
            self._make_day("2026-03-01", sport="Run", distance=10000),
            self._make_day("2026-03-02", sport="Ride", distance=30000),
            self._make_day("2026-03-03", sport="Run", distance=12000),
        ]
        result = _compute_rollup(combined, end_date=date(2026, 3, 3), window_days=7)

        acts = result["activities"]
        assert acts["count"] == 3
        assert "Run" in acts["count_by_sport"]
        assert acts["count_by_sport"]["Run"] == 2
        assert acts["count_by_sport"]["Ride"] == 1
        assert "Run" in acts["by_sport"]
        assert "Ride" in acts["by_sport"]

    def test_averages_heartrate_only_from_present_values(self) -> None:
        combined = [
            self._make_day("2026-03-01", avg_hr=150),
            self._make_day("2026-03-02", avg_hr=None),
            self._make_day("2026-03-03", avg_hr=160),
        ]
        result = _compute_rollup(combined, end_date=date(2026, 3, 3), window_days=7)
        # Average should only consider the two activities with HR
        assert result["activities"]["average_heartrate_mean"] == pytest.approx(155.0)

    def test_counts_sleep_days_with_data(self) -> None:
        combined = [
            self._make_day("2026-03-01", sleep={"sleepTimeSeconds": 28000}),
            self._make_day("2026-03-02", sleep=None),
            self._make_day("2026-03-03", sleep={"sleepTimeSeconds": 27000}),
        ]
        result = _compute_rollup(combined, end_date=date(2026, 3, 3), window_days=7)
        assert result["sleep_days_with_data"] == 2

    def test_window_excludes_days_outside_range(self) -> None:
        combined = [
            self._make_day("2026-02-20"),  # outside 7d window
            self._make_day("2026-03-01"),
            self._make_day("2026-03-03"),
        ]
        result = _compute_rollup(combined, end_date=date(2026, 3, 3), window_days=7)
        # Feb 20 is >7 days before Mar 3
        assert result["activities"]["count"] == 2

    def test_distance_converted_to_km(self) -> None:
        combined = [self._make_day("2026-03-01", distance=10000)]
        result = _compute_rollup(combined, end_date=date(2026, 3, 1), window_days=7)
        assert result["activities"]["total_distance_km"] == pytest.approx(10.0)

    def test_moving_time_converted_to_hours(self) -> None:
        combined = [self._make_day("2026-03-01", moving_time=7200)]
        result = _compute_rollup(combined, end_date=date(2026, 3, 1), window_days=7)
        assert result["activities"]["total_moving_time_hours"] == pytest.approx(2.0)

    def test_no_heartrate_returns_none(self) -> None:
        combined = [self._make_day("2026-03-01", avg_hr=None)]
        result = _compute_rollup(combined, end_date=date(2026, 3, 1), window_days=7)
        assert result["activities"]["average_heartrate_mean"] is None


# ---------------------------------------------------------------------------
# combine.main — integration tests with temp dirs
# ---------------------------------------------------------------------------


class TestCombineMain:
    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def _make_runtime(self, processing: Path, prompting: Path):
        import types

        return types.SimpleNamespace(
            paths=types.SimpleNamespace(
                processing_directory=processing,
                prompting_directory=prompting,
            )
        )

    def test_main_writes_summary_and_rollups_when_last_date_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trailtraining.data import combine as combine_mod

        processing = tmp_path / "processing"
        prompting = tmp_path / "prompting"
        processing.mkdir()
        prompting.mkdir()

        # Write minimal sleep and activity fixtures
        sleep_data = [{"calendarDate": "2026-03-01", "sleepTimeSeconds": 28000}]
        self._write_json(processing / "filtered_sleep.json", sleep_data)

        activity_data = [
            {
                "id": 1,
                "start_date_local": "2026-03-01T09:00:00",
                "sport_type": "Run",
                "distance": 10000,
                "moving_time": 3600,
            }
        ]
        self._write_json(processing / "strava_activities.json", activity_data)

        monkeypatch.setattr(combine_mod.config, "ensure_directories", lambda runtime=None: None)
        monkeypatch.setattr(
            combine_mod.config,
            "current",
            lambda: self._make_runtime(processing, prompting),
        )
        monkeypatch.setattr(combine_mod, "build_formatted_personal_profile", lambda **kwargs: None)

        combine_mod.main()

        assert (prompting / "combined_summary.json").exists()
        assert (prompting / "combined_rollups.json").exists()

        rollups = json.loads((prompting / "combined_rollups.json").read_text())
        assert "windows" in rollups
        assert "7" in rollups["windows"]
        assert "28" in rollups["windows"]

    def test_main_skips_rollups_when_last_combined_date_unparseable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trailtraining.data import combine as combine_mod

        processing = tmp_path / "processing"
        prompting = tmp_path / "prompting"
        processing.mkdir()
        prompting.mkdir()

        # Write files that produce combined data with a bad date at the end
        sleep_data = [{"calendarDate": "bad-date", "sleepTimeSeconds": 28000}]
        self._write_json(processing / "filtered_sleep.json", sleep_data)
        self._write_json(processing / "strava_activities.json", [])

        monkeypatch.setattr(combine_mod.config, "ensure_directories", lambda runtime=None: None)
        monkeypatch.setattr(
            combine_mod.config,
            "current",
            lambda: self._make_runtime(processing, prompting),
        )
        monkeypatch.setattr(combine_mod, "build_formatted_personal_profile", lambda **kwargs: None)

        combine_mod.main()

        assert (prompting / "combined_summary.json").exists()
        # Rollups must NOT be written because the date is bad
        assert not (prompting / "combined_rollups.json").exists()
