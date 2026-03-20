"""tests/test_intervals_expanded.py — expanded unit tests for trailtraining.pipelines.intervals"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from trailtraining.pipelines.intervals import (
    _auth_headers,
    _validate_ymd,
    ensure_personal_stub,
    fetch_wellness,
    normalize_to_filtered_sleep,
)
from trailtraining.util.errors import ConfigError, DataValidationError, ExternalServiceError

# ---------------------------------------------------------------------------
# _auth_headers
# ---------------------------------------------------------------------------


class TestAuthHeaders:
    def test_prefers_bearer_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INTERVALS_ACCESS_TOKEN", "mytoken123")
        headers = _auth_headers()
        assert headers["Authorization"] == "Bearer mytoken123"

    def test_falls_back_to_basic_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("INTERVALS_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("INTERVALS_API_KEY", "secret-key")
        headers = _auth_headers()
        assert headers["Authorization"].startswith("Basic ")
        decoded = base64.b64decode(headers["Authorization"][6:]).decode()
        assert "secret-key" in decoded

    def test_raises_when_no_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("INTERVALS_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("INTERVALS_API_KEY", raising=False)
        monkeypatch.setattr(
            "trailtraining.pipelines.intervals.config.current",
            lambda: MagicMock(intervals_api_key=""),
        )
        with pytest.raises(ConfigError):
            _auth_headers()


# ---------------------------------------------------------------------------
# _validate_ymd
# ---------------------------------------------------------------------------


class TestValidateYmd:
    def test_accepts_iso_date(self) -> None:
        assert _validate_ymd("2026-03-15", "newest") == "2026-03-15"

    def test_rejects_bad_date(self) -> None:
        with pytest.raises(DataValidationError):
            _validate_ymd("not-a-date", "oldest")

    def test_rejects_partial_date(self) -> None:
        with pytest.raises(DataValidationError):
            _validate_ymd("2026-13", "oldest")


# ---------------------------------------------------------------------------
# fetch_wellness
# ---------------------------------------------------------------------------


class TestFetchWellness:
    def test_raises_on_non_json_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import trailtraining.pipelines.intervals as intervals_mod

        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("bad json")
        mock_resp.status_code = 200
        mock_resp.text = "not json"
        monkeypatch.setattr(
            intervals_mod,
            "_request_with_retry",
            lambda session, method, url, **kwargs: mock_resp,
        )
        monkeypatch.setenv("INTERVALS_ACCESS_TOKEN", "tok")

        with pytest.raises(ExternalServiceError, match="JSON parse"):
            fetch_wellness("2026-03-01", "2026-03-07")

    def test_raises_on_non_list_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import trailtraining.pipelines.intervals as intervals_mod

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "bad"}
        mock_resp.status_code = 200
        monkeypatch.setattr(
            intervals_mod,
            "_request_with_retry",
            lambda session, method, url, **kwargs: mock_resp,
        )
        monkeypatch.setenv("INTERVALS_ACCESS_TOKEN", "tok")

        with pytest.raises(DataValidationError):
            fetch_wellness("2026-03-01", "2026-03-07")


# ---------------------------------------------------------------------------
# normalize_to_filtered_sleep
# ---------------------------------------------------------------------------


class TestNormalizeToFilteredSleep:
    def test_basic_normalization(self) -> None:
        entry = {"id": "2026-02-27", "sleepSecs": 3600, "restingHR": 45, "avgOvernightHrv": 55}
        out = normalize_to_filtered_sleep(entry)
        assert out["calendarDate"] == "2026-02-27"
        assert out["sleepTimeSeconds"] == 3600
        assert out["restingHeartRate"] == 45
        assert out["avgOvernightHrv"] == 55

    def test_accepts_alternate_keys(self) -> None:
        entry = {
            "day": "2026-03-10",
            "sleep_seconds": 28800,
            "restingHeartRate": 52,
            "rmssd": 70,
        }
        out = normalize_to_filtered_sleep(entry)
        assert out["calendarDate"] == "2026-03-10"
        assert out["sleepTimeSeconds"] == 28800
        assert out["restingHeartRate"] == 52
        assert out["avgOvernightHrv"] == 70

    def test_accepts_date_key(self) -> None:
        entry = {"date": "2026-04-01", "sleepSecs": 25000}
        out = normalize_to_filtered_sleep(entry)
        assert out["calendarDate"] == "2026-04-01"

    def test_missing_values_become_minus_one(self) -> None:
        entry = {"id": "2026-03-15"}
        out = normalize_to_filtered_sleep(entry)
        assert out["sleepTimeSeconds"] == -1
        assert out["restingHeartRate"] == -1
        assert out["avgOvernightHrv"] == -1

    def test_raises_without_date(self) -> None:
        with pytest.raises(DataValidationError, match="date"):
            normalize_to_filtered_sleep({"sleepSecs": 3600})

    def test_date_truncated_to_10_chars(self) -> None:
        entry = {"id": "2026-03-01T00:00:00"}
        out = normalize_to_filtered_sleep(entry)
        assert out["calendarDate"] == "2026-03-01"


# ---------------------------------------------------------------------------
# ensure_personal_stub
# ---------------------------------------------------------------------------


class TestEnsurePersonalStub:
    def _make_runtime(self, prompting: Path):
        import types

        return types.SimpleNamespace(
            paths=types.SimpleNamespace(prompting_directory=prompting),
        )

    def test_is_noop_if_file_exists(self, tmp_path: Path) -> None:
        prompting = tmp_path / "prompting"
        prompting.mkdir()
        stub_path = prompting / "formatted_personal_data.json"
        original = {"existing": True}
        stub_path.write_text(json.dumps(original), encoding="utf-8")

        runtime = self._make_runtime(prompting)
        ensure_personal_stub(runtime)

        content = json.loads(stub_path.read_text())
        assert content == original  # unchanged

    def test_creates_minimal_payload(self, tmp_path: Path) -> None:
        prompting = tmp_path / "prompting"
        prompting.mkdir()
        runtime = self._make_runtime(prompting)

        ensure_personal_stub(runtime)

        stub_path = prompting / "formatted_personal_data.json"
        assert stub_path.exists()
        content = json.loads(stub_path.read_text())
        assert "userInfo" in content
        assert "biometricProfile" in content


# ---------------------------------------------------------------------------
# intervals.main
# ---------------------------------------------------------------------------


class TestIntervalsMain:
    def _make_runtime(self, processing: Path, prompting: Path):
        import types

        return types.SimpleNamespace(
            paths=types.SimpleNamespace(
                processing_directory=processing,
                prompting_directory=prompting,
            ),
            intervals_athlete_id="0",
        )

    def test_raises_when_oldest_gt_newest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import trailtraining.pipelines.intervals as intervals_mod

        # Patch config so no real directories are created as a side effect
        monkeypatch.setattr(intervals_mod.config, "ensure_directories", lambda runtime=None: None)
        monkeypatch.setattr(
            intervals_mod.config,
            "current",
            lambda: self._make_runtime(tmp_path / "processing", tmp_path / "prompting"),
        )

        with pytest.raises(DataValidationError, match="oldest must be"):
            intervals_mod.main(oldest="2026-03-10", newest="2026-03-01")

    def test_main_writes_sorted_filtered_sleep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import trailtraining.pipelines.intervals as intervals_mod

        processing = tmp_path / "processing"
        prompting = tmp_path / "prompting"
        processing.mkdir()
        prompting.mkdir()

        # Return unsorted entries
        unsorted_entries = [
            {"id": "2026-03-03", "sleepSecs": 28000},
            {"id": "2026-03-01", "sleepSecs": 27000},
            {"id": "2026-03-02", "sleepSecs": 29000},
        ]
        monkeypatch.setattr(
            intervals_mod, "fetch_wellness", lambda oldest, newest: unsorted_entries
        )
        monkeypatch.setattr(intervals_mod.config, "ensure_directories", lambda runtime=None: None)
        monkeypatch.setattr(
            intervals_mod.config,
            "current",
            lambda: self._make_runtime(processing, prompting),
        )
        monkeypatch.setattr(intervals_mod, "ensure_personal_stub", lambda runtime=None: None)

        intervals_mod.main(oldest="2026-03-01", newest="2026-03-07")

        out_path = processing / "filtered_sleep.json"
        assert out_path.exists()
        out = json.loads(out_path.read_text())
        dates = [entry["calendarDate"] for entry in out]
        assert dates == sorted(dates), "Output should be sorted by calendarDate"
