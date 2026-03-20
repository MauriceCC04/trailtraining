"""tests/test_garmin_formatting.py — unit tests for trailtraining.data.garmin formatting"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from trailtraining.data.garmin import (
    filter_sleep,
    format_personal_data,
    shorten_rhr,
    shorten_sleep,
)

# ---------------------------------------------------------------------------
# format_personal_data — redacts PII fields
# ---------------------------------------------------------------------------


class TestFormatPersonalData:
    def _write(self, tmp_path: Path, data: Any) -> tuple[str, str]:
        inp = tmp_path / "personal.json"
        outp = tmp_path / "formatted.json"
        inp.write_text(json.dumps(data), encoding="utf-8")
        return str(inp), str(outp)

    def test_redacts_pii_fields(self, tmp_path: Path) -> None:
        data = {
            "userInfo": {
                "email": "user@example.com",
                "locale": "en-US",
                "timeZone": "America/Los_Angeles",
                "countryCode": "US",
                "name": "Alice",
            },
            "biometricProfile": {
                "userId": 12345,
                "weight": 65,
            },
            "birthDate": "1990-01-01",
            "gender": "female",
            "timeZone": "UTC",
            "locale": "en",
        }
        inp, outp = self._write(tmp_path, data)
        format_personal_data(inp, outp)
        result = json.loads(Path(outp).read_text())

        # PII fields should be removed
        user_info = result.get("userInfo", {})
        assert "email" not in user_info
        assert "locale" not in user_info
        assert "timeZone" not in user_info
        assert "countryCode" not in user_info
        # Non-PII preserved
        assert user_info.get("name") == "Alice"

        # Top-level PII removed
        assert "birthDate" not in result
        assert "gender" not in result
        assert "timeZone" not in result
        assert "locale" not in result

    def test_handles_missing_sections_gracefully(self, tmp_path: Path) -> None:
        inp, outp = self._write(tmp_path, {})
        format_personal_data(inp, outp)
        result = json.loads(Path(outp).read_text())
        assert isinstance(result, dict)

    def test_handles_non_dict_gracefully(self, tmp_path: Path) -> None:
        inp, outp = self._write(tmp_path, ["not", "a", "dict"])
        format_personal_data(inp, outp)
        result = json.loads(Path(outp).read_text())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# shorten_rhr — keeps only recent entries
# ---------------------------------------------------------------------------


class TestShortenRhr:
    def _make_rhr_entry(self, date_str: str) -> dict[str, Any]:
        return {
            "allMetrics": {
                "metricsMap": {
                    "WELLNESS_RESTING_HEART_RATE": [{"calendarDate": date_str, "value": 50}]
                }
            }
        }

    def test_keeps_only_recent_entries(self, tmp_path: Path) -> None:
        today = datetime.now()
        recent = (today - timedelta(days=10)).strftime("%Y-%m-%d")
        old = (today - timedelta(days=250)).strftime("%Y-%m-%d")

        data = [self._make_rhr_entry(recent), self._make_rhr_entry(old)]
        inp = tmp_path / "rhr.json"
        outp = tmp_path / "short_rhr.json"
        inp.write_text(json.dumps(data), encoding="utf-8")

        shorten_rhr(str(inp), str(outp))
        result = json.loads(outp.read_text())
        assert len(result) == 1
        date_kept = result[0]["allMetrics"]["metricsMap"]["WELLNESS_RESTING_HEART_RATE"][0][
            "calendarDate"
        ]
        assert date_kept == recent

    def test_keeps_entries_at_boundary(self, tmp_path: Path) -> None:
        today = datetime.now()
        boundary = (today - timedelta(days=199)).strftime("%Y-%m-%d")
        data = [self._make_rhr_entry(boundary)]
        inp = tmp_path / "rhr.json"
        outp = tmp_path / "short_rhr.json"
        inp.write_text(json.dumps(data), encoding="utf-8")
        shorten_rhr(str(inp), str(outp))
        result = json.loads(outp.read_text())
        assert len(result) == 1

    def test_handles_empty_list(self, tmp_path: Path) -> None:
        inp = tmp_path / "rhr.json"
        outp = tmp_path / "short_rhr.json"
        inp.write_text("[]", encoding="utf-8")
        shorten_rhr(str(inp), str(outp))
        assert json.loads(outp.read_text()) == []


# ---------------------------------------------------------------------------
# shorten_sleep — keeps only recent entries
# ---------------------------------------------------------------------------


class TestShortenSleep:
    def _make_sleep_entry(self, date_str: str) -> dict[str, Any]:
        return {"calendarDate": date_str, "sleepTimeSeconds": 28000}

    def test_keeps_only_recent_entries(self, tmp_path: Path) -> None:
        today = datetime.now()
        recent = (today - timedelta(days=5)).strftime("%Y-%m-%d")
        old = (today - timedelta(days=300)).strftime("%Y-%m-%d")

        data = [self._make_sleep_entry(recent), self._make_sleep_entry(old)]
        inp = tmp_path / "sleep.json"
        outp = tmp_path / "short_sleep.json"
        inp.write_text(json.dumps(data), encoding="utf-8")

        shorten_sleep(str(inp), str(outp))
        result = json.loads(outp.read_text())
        assert len(result) == 1
        assert result[0]["calendarDate"] == recent


# ---------------------------------------------------------------------------
# filter_sleep — various input formats
# ---------------------------------------------------------------------------


class TestFilterSleep:
    def _write_and_filter(self, tmp_path: Path, data: Any) -> list[dict[str, Any]]:
        inp = tmp_path / "sleep.json"
        outp = tmp_path / "filtered.json"
        inp.write_text(json.dumps(data), encoding="utf-8")
        filter_sleep(str(inp), str(outp))
        result: list[dict[str, Any]] = json.loads(outp.read_text())
        return result

    def test_accepts_nested_list_input(self, tmp_path: Path) -> None:
        data = [
            [
                {
                    "dailySleepDTO": {
                        "calendarDate": "2026-03-01",
                        "sleepTimeSeconds": 28000,
                        "restingHeartRate": 50,
                        "avgOvernightHrv": 65,
                    }
                }
            ]
        ]
        result = self._write_and_filter(tmp_path, data)
        assert len(result) == 1
        assert result[0]["calendarDate"] == "2026-03-01"

    def test_preserves_hrv_status_when_present(self, tmp_path: Path) -> None:
        data = [
            {
                "calendarDate": "2026-03-01",
                "sleepTimeSeconds": 28000,
                "hrvStatus": "BALANCED",
            }
        ]
        result = self._write_and_filter(tmp_path, data)
        assert result[0]["hrvStatus"] == "BALANCED"

    def test_uses_minus_one_for_missing_numeric_values(self, tmp_path: Path) -> None:
        data = [{"calendarDate": "2026-03-01"}]
        result = self._write_and_filter(tmp_path, data)
        assert result[0]["sleepTimeSeconds"] == -1
        assert result[0]["restingHeartRate"] == -1
        assert result[0]["avgOvernightHrv"] == -1

    def test_accepts_dict_input(self, tmp_path: Path) -> None:
        data = {"calendarDate": "2026-03-01", "sleepTimeSeconds": 28000}
        result = self._write_and_filter(tmp_path, data)
        assert len(result) == 1
        assert result[0]["calendarDate"] == "2026-03-01"

    def test_reads_from_daily_sleep_dto(self, tmp_path: Path) -> None:
        data = [
            {
                "dailySleepDTO": {
                    "calendarDate": "2026-03-15",
                    "sleepTimeSeconds": 27000,
                    "restingHeartRate": 48,
                    "avgOvernightHrv": 70,
                },
                "extra_junk": [1, 2, 3],
            }
        ]
        result = self._write_and_filter(tmp_path, data)
        assert result[0]["calendarDate"] == "2026-03-15"
        assert result[0]["sleepTimeSeconds"] == 27000
        assert result[0]["restingHeartRate"] == 48
        assert result[0]["avgOvernightHrv"] == 70

    def test_dto_preferred_over_top_level(self, tmp_path: Path) -> None:
        # pick() checks dailySleepDTO first, so DTO values win when present
        data = [
            {
                "calendarDate": "2026-03-01",
                "sleepTimeSeconds": 30000,
                "dailySleepDTO": {
                    "calendarDate": "2026-06-15",
                    "sleepTimeSeconds": 1000,
                },
            }
        ]
        result = self._write_and_filter(tmp_path, data)
        assert result[0]["calendarDate"] == "2026-06-15"
        assert result[0]["sleepTimeSeconds"] == 1000
