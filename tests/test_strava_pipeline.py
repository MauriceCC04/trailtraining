"""tests/test_strava_pipeline.py — unit tests for trailtraining.pipelines.strava"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from trailtraining.pipelines.strava import (
    _merge_by_id,
    _parse_strava_datetime,
    fetch_activities_incremental,
)
from trailtraining.util.errors import DataValidationError, ExternalServiceError

# ---------------------------------------------------------------------------
# _parse_strava_datetime
# ---------------------------------------------------------------------------


class TestParseStravaDatetime:
    def test_handles_z_suffix(self) -> None:
        result = _parse_strava_datetime("2026-03-01T12:00:00Z")
        assert result is not None
        assert result.year == 2026
        assert result.month == 3
        assert result.tzinfo is not None

    def test_handles_plus_offset(self) -> None:
        result = _parse_strava_datetime("2026-03-01T12:00:00+00:00")
        assert result is not None

    def test_returns_none_for_invalid(self) -> None:
        assert _parse_strava_datetime("not-a-date") is None
        assert _parse_strava_datetime("") is None
        assert _parse_strava_datetime(None) is None

    def test_returns_none_for_empty_string(self) -> None:
        assert _parse_strava_datetime("") is None


# ---------------------------------------------------------------------------
# _api_get / ExternalServiceError
# ---------------------------------------------------------------------------


class TestApiGet:
    def test_wraps_json_parse_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import trailtraining.pipelines.strava as strava_mod

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "not json at all"
        mock_resp.json.side_effect = ValueError("bad json")

        monkeypatch.setattr(
            strava_mod,
            "_request_with_retry",
            lambda session, method, url, **kwargs: mock_resp,
        )

        with pytest.raises(ExternalServiceError, match="JSON parse"):
            strava_mod._api_get(mock_session, "/athlete/activities", "fake_token")


# ---------------------------------------------------------------------------
# fetch_activities_incremental
# ---------------------------------------------------------------------------


def _make_activities(n: int, base_date: str = "2026-03-01") -> list[dict[str, Any]]:
    return [
        {
            "id": i + 1,
            "name": f"Activity {i + 1}",
            "start_date": f"{base_date}T{i:02d}:00:00Z",
            "start_date_local": f"{base_date}T{i:02d}:00:00",
            "sport_type": "Run",
            "distance": 10000.0,
            "moving_time": 3600,
        }
        for i in range(n)
    ]


class TestFetchActivitiesIncremental:
    def test_stops_on_short_page(self) -> None:
        """One page shorter than per_page → no more pages fetched."""
        items = _make_activities(3)

        with patch(
            "trailtraining.pipelines.strava._api_get",
            return_value=items,
        ):
            results, info = fetch_activities_incremental(
                session=MagicMock(),
                access_token="tok",
                after_unix=0,
                per_page=10,
                max_pages=0,
                hard_max_pages=100,
            )

        assert len(results) == 3
        assert info["hit_max_pages"] is False
        assert info["pages_fetched"] == 1

    def test_sets_hit_max_pages_when_limit_reached(self) -> None:
        """max_pages=2 with full pages → hit_max_pages=True."""
        full_page = _make_activities(3)

        call_count = [0]

        def _fake_api_get(session, path, token, params=None):
            call_count[0] += 1
            return full_page  # always full → never stops naturally

        with patch("trailtraining.pipelines.strava._api_get", side_effect=_fake_api_get):
            results, info = fetch_activities_incremental(
                session=MagicMock(),
                access_token="tok",
                after_unix=0,
                per_page=3,
                max_pages=2,
                hard_max_pages=100,
            )

        assert info["hit_max_pages"] is True
        assert info["pages_fetched"] == 2

    def test_raises_on_hard_max_pages(self) -> None:
        """Exceeding hard_max_pages raises RuntimeError."""
        full_page = _make_activities(3)

        with (
            patch("trailtraining.pipelines.strava._api_get", return_value=full_page),
            pytest.raises(RuntimeError, match="HARD_MAX_PAGES"),
        ):
            fetch_activities_incremental(
                session=MagicMock(),
                access_token="tok",
                after_unix=0,
                per_page=3,
                max_pages=0,  # unlimited
                hard_max_pages=1,
            )

    def test_rejects_non_dict_items(self) -> None:
        """Non-dict items in the list raise DataValidationError."""
        bad_page = ["not-a-dict", "also-bad"]

        with (
            patch("trailtraining.pipelines.strava._api_get", return_value=bad_page),
            pytest.raises(DataValidationError, match="activity item"),
        ):
            fetch_activities_incremental(
                session=MagicMock(),
                access_token="tok",
                after_unix=0,
                per_page=10,
                max_pages=0,
                hard_max_pages=10,
            )

    def test_raises_on_non_list_payload(self) -> None:
        with (
            patch("trailtraining.pipelines.strava._api_get", return_value={"error": "bad"}),
            pytest.raises(DataValidationError, match="Unexpected Strava"),
        ):
            fetch_activities_incremental(
                session=MagicMock(),
                access_token="tok",
                after_unix=0,
                per_page=10,
                max_pages=0,
                hard_max_pages=10,
            )

    def test_empty_first_page_returns_empty(self) -> None:
        with patch("trailtraining.pipelines.strava._api_get", return_value=[]):
            results, info = fetch_activities_incremental(
                session=MagicMock(),
                access_token="tok",
                after_unix=0,
                per_page=10,
                max_pages=0,
                hard_max_pages=10,
            )
        assert results == []
        assert info["pages_fetched"] == 0


# ---------------------------------------------------------------------------
# _merge_by_id
# ---------------------------------------------------------------------------


class TestMergeById:
    def test_prefers_new_item_for_same_id(self) -> None:
        existing = [
            {
                "id": 1,
                "start_date": "2026-03-01T08:00:00Z",
                "name": "Old name",
                "distance": 5000.0,
            }
        ]
        new_items = [
            {
                "id": 1,
                "start_date": "2026-03-01T08:00:00Z",
                "name": "Updated name",
                "distance": 6000.0,
            }
        ]
        merged = _merge_by_id(existing, new_items)
        assert len(merged) == 1
        assert merged[0]["name"] == "Updated name"
        assert merged[0]["distance"] == 6000.0

    def test_adds_new_items_without_collision(self) -> None:
        existing = [{"id": 1, "start_date": "2026-03-01T08:00:00Z"}]
        new_items = [{"id": 2, "start_date": "2026-03-02T08:00:00Z"}]
        merged = _merge_by_id(existing, new_items)
        assert len(merged) == 2

    def test_sorts_descending_by_start_date_then_id(self) -> None:
        existing = [
            {"id": 10, "start_date": "2026-03-01T08:00:00Z"},
            {"id": 20, "start_date": "2026-02-28T08:00:00Z"},
        ]
        new_items = [
            {"id": 30, "start_date": "2026-03-02T08:00:00Z"},
        ]
        merged = _merge_by_id(existing, new_items)
        dates = [m["start_date"] for m in merged]
        # Should be descending: Mar 2, Mar 1, Feb 28
        assert dates[0] > dates[1] > dates[2]

    def test_same_date_sorts_by_id_descending(self) -> None:
        existing = [{"id": 5, "start_date": "2026-03-01T08:00:00Z"}]
        new_items = [{"id": 10, "start_date": "2026-03-01T08:00:00Z"}]
        merged = _merge_by_id(existing, new_items)
        assert merged[0]["id"] == 10  # higher id first


# ---------------------------------------------------------------------------
# main — meta output
# ---------------------------------------------------------------------------


class TestStravaMain:
    def _patch_main(
        self,
        monkeypatch: pytest.MonkeyPatch,
        strava_mod: Any,
        processing: Path,
        fetch_return: tuple,
    ) -> None:
        """Apply all patches needed for strava.main() to run without real I/O or auth."""
        import types

        fake_runtime = types.SimpleNamespace(
            paths=types.SimpleNamespace(processing_directory=processing)
        )
        # Patch config.current so main() resolves paths to our tmp dir
        monkeypatch.setattr(strava_mod.config, "current", lambda: fake_runtime)
        monkeypatch.setattr(strava_mod.config, "ensure_directories", lambda runtime=None: None)
        # StravaOAuthConfig.from_env() raises if STRAVA_CLIENT_ID is unset
        monkeypatch.setattr(
            strava_mod.StravaOAuthConfig,
            "from_env",
            staticmethod(lambda: MagicMock()),
        )
        monkeypatch.setattr(
            strava_mod,
            "_get_or_auth_token",
            lambda cfg: {"access_token": "tok", "expires_at": time.time() + 3600},
        )
        monkeypatch.setattr(
            strava_mod,
            "fetch_activities_incremental",
            lambda session, access_token, after_unix, **kwargs: fetch_return,
        )

    def test_main_writes_meta_with_counts_and_after_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        import trailtraining.pipelines.strava as strava_mod

        processing = tmp_path / "processing"
        processing.mkdir()

        new_items = _make_activities(3)
        self._patch_main(
            monkeypatch,
            strava_mod,
            processing,
            fetch_return=(new_items, {"hit_max_pages": False, "pages_fetched": 1}),
        )

        strava_mod.main()

        meta_path = processing / "strava_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "count" in meta
        assert "new_count" in meta
        assert "pagination" in meta
        assert "after_used" in meta
        assert meta["count"] == 3
        assert meta["new_count"] == 3

    def test_main_prints_warning_when_hit_max_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        import trailtraining.pipelines.strava as strava_mod

        processing = tmp_path / "processing"
        processing.mkdir()

        new_items = _make_activities(2)
        self._patch_main(
            monkeypatch,
            strava_mod,
            processing,
            fetch_return=(new_items, {"hit_max_pages": True, "pages_fetched": 5}),
        )

        strava_mod.main()

        out = capsys.readouterr().out
        assert "TRAILTRAINING_STRAVA_MAX_PAGES" in out
