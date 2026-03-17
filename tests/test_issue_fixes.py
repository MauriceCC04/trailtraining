"""Tests for the nine identified issues.

Covers:
1. LLM fallback logic — only retries on unsupported-parameter errors
2. Contract field types — proper date/numeric validation
3. Coach missing-input fail-fast
4. Strava first-sync lookback
5. ICS export timezone correctness
6. Shared retry utility
7. (CLI parser — tested via existing smoke tests)
8. (README — manual review)
9. Comprehensive test coverage for above
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Issue 1: LLM fallback logic tests
# ---------------------------------------------------------------------------


class TestLLMFallbackClassification:
    """Only unsupported-parameter errors should trigger fallback."""

    def test_unsupported_param_detected(self) -> None:
        from trailtraining.util.llm_helpers import _is_unsupported_parameter_error

        unsupported_msgs = [
            "Unsupported parameter: reasoning",
            "unknown parameter 'verbosity'",
            "This model does not support json_schema response format",
            "Additional properties not allowed: 'reasoning'",
            "Unrecognized field 'text'",
        ]
        for msg in unsupported_msgs:
            exc = ValueError(msg)
            assert _is_unsupported_parameter_error(exc), f"Should detect: {msg}"

    def test_auth_error_not_classified_as_unsupported(self) -> None:
        from trailtraining.util.llm_helpers import _is_unsupported_parameter_error

        non_retryable = [
            "Invalid API key provided",
            "Authentication failed",
            "Rate limit exceeded",
            "Internal server error",
            "Connection timeout",
            "502 Bad Gateway",
        ]
        for msg in non_retryable:
            exc = RuntimeError(msg)
            assert not _is_unsupported_parameter_error(exc), f"Should NOT detect: {msg}"

    def test_classify_and_raise_converts_unsupported(self) -> None:
        from trailtraining.util.errors import LLMUnsupportedParameterError
        from trailtraining.util.llm_helpers import _classify_and_raise

        with pytest.raises(LLMUnsupportedParameterError):
            _classify_and_raise(ValueError("unsupported parameter 'reasoning'"))

    def test_classify_and_raise_preserves_auth_errors(self) -> None:
        from trailtraining.util.llm_helpers import _classify_and_raise

        with pytest.raises(RuntimeError, match="Invalid API key"):
            _classify_and_raise(RuntimeError("Invalid API key provided"))


# ---------------------------------------------------------------------------
# Issue 2: Contract field type tests
# ---------------------------------------------------------------------------


class TestContractTypes:
    """Verify proper types replace stringly-typed fields."""

    def test_training_meta_accepts_date_strings(self) -> None:
        from trailtraining.contracts import TrainingMeta

        meta = TrainingMeta(
            today="2026-03-13",
            plan_start="2026-03-14",
            plan_days=7,
            style="trailrunning",
            primary_goal="test",
        )
        assert isinstance(meta.today, date)
        assert isinstance(meta.plan_start, date)
        assert meta.today == date(2026, 3, 13)

    def test_training_meta_rejects_invalid_date(self) -> None:
        from trailtraining.contracts import TrainingMeta

        with pytest.raises(ValidationError):
            TrainingMeta(
                today="not-a-date",
                plan_start="2026-03-14",
                plan_days=7,
                style="trailrunning",
            )

    def test_plan_day_date_is_proper_date(self) -> None:
        from trailtraining.contracts import PlanDay

        day = PlanDay(
            date="2026-03-14",
            title="Easy run",
            session_type="easy",
            is_rest_day=False,
            is_hard_day=False,
            duration_minutes=45,
            target_intensity="easy",
            terrain="road",
            workout="45 min easy",
            purpose="aerobic",
            signal_ids=[],
        )
        assert isinstance(day.date, date)
        assert day.date == date(2026, 3, 14)

    def test_snapshot_stats_coerce_numeric_to_str(self) -> None:
        from trailtraining.contracts import SnapshotStats

        stats = SnapshotStats(
            distance_km=22.5,
            moving_time_hours=2.2,
            elevation_m=300,
            activity_count=2,
            sleep_hours_mean="",
            hrv_mean="",
            rhr_mean="46",
        )
        assert stats.distance_km == "22.5"
        assert stats.elevation_m == "300"

    def test_snapshot_stats_empty_string_preserved(self) -> None:
        from trailtraining.contracts import SnapshotStats

        stats = SnapshotStats(
            distance_km="",
            moving_time_hours="",
            elevation_m="",
            activity_count="",
            sleep_hours_mean="",
            hrv_mean="",
            rhr_mean="",
        )
        assert stats.distance_km == ""

    def test_forecast_inputs_date_is_proper(self) -> None:
        from trailtraining.contracts import ForecastInputs

        inputs = ForecastInputs(as_of_date="2026-03-13")
        assert isinstance(inputs.as_of_date, date)

    def test_full_training_plan_serialization_roundtrip(self) -> None:
        """Verify that model_dump + model_validate roundtrips correctly with new types."""
        from trailtraining.contracts import TrainingPlanArtifact

        payload = {
            "meta": {
                "today": "2026-03-13",
                "plan_start": "2026-03-14",
                "plan_days": 7,
                "style": "trailrunning",
                "primary_goal": "test goal",
            },
            "snapshot": {
                "last7": {
                    "distance_km": "22",
                    "moving_time_hours": "2.2",
                    "elevation_m": "300",
                    "activity_count": "2",
                    "sleep_hours_mean": "",
                    "hrv_mean": "",
                    "rhr_mean": "46",
                },
                "baseline28": {
                    "distance_km": "18",
                    "moving_time_hours": "1.9",
                    "elevation_m": "220",
                    "activity_count": "2",
                    "sleep_hours_mean": "",
                    "hrv_mean": "",
                    "rhr_mean": "45",
                },
                "notes": "",
            },
            "readiness": {
                "status": "steady",
                "rationale": "Test.",
                "signal_ids": [],
            },
            "plan": {
                "weekly_totals": {
                    "planned_distance_km": 30.0,
                    "planned_moving_time_hours": 4.5,
                    "planned_elevation_m": 600.0,
                },
                "days": [
                    {
                        "date": "2026-03-14",
                        "title": "Rest",
                        "session_type": "rest",
                        "is_rest_day": True,
                        "is_hard_day": False,
                        "duration_minutes": 0,
                        "target_intensity": "Off",
                        "terrain": "N/A",
                        "workout": "Rest day",
                        "purpose": "Recovery",
                        "signal_ids": [],
                    }
                ],
            },
            "recovery": {"actions": [], "signal_ids": []},
            "risks": [],
            "data_notes": [],
            "citations": [],
        }
        artifact = TrainingPlanArtifact.model_validate(payload)
        dumped = artifact.model_dump(mode="json")
        # Dates should serialize to ISO strings
        assert dumped["meta"]["today"] == "2026-03-13"
        assert dumped["plan"]["days"][0]["date"] == "2026-03-14"
        # Roundtrip
        artifact2 = TrainingPlanArtifact.model_validate(dumped)
        assert artifact2.meta.today == date(2026, 3, 13)


# ---------------------------------------------------------------------------
# Issue 3: Coach missing-input fail-fast (testing the concept)
# ---------------------------------------------------------------------------


class TestMissingArtifactError:
    """Verify MissingArtifactError can be raised for missing inputs."""

    def test_missing_artifact_error_is_trail_training_error(self) -> None:
        from trailtraining.util.errors import MissingArtifactError, TrailTrainingError

        err = MissingArtifactError(message="combined_summary.json is empty")
        assert isinstance(err, TrailTrainingError)
        assert str(err) == "combined_summary.json is empty"


# ---------------------------------------------------------------------------
# Issue 4: Strava first-sync lookback tests
# ---------------------------------------------------------------------------


class TestStravaFirstSyncLookback:
    """DEFAULT_LOOKBACK_DAYS must be respected on first sync."""

    def test_first_sync_uses_lookback_days(self) -> None:
        from trailtraining.pipelines.strava import _compute_after_unix

        # No existing activities, no meta → should use lookback
        result = _compute_after_unix([], {}, lookback_days=90)
        now_ts = int(datetime.now(tz=timezone.utc).timestamp())
        expected_min = now_ts - 90 * 86400 - 60  # allow 60s tolerance
        expected_max = now_ts - 90 * 86400 + 60
        assert expected_min <= result <= expected_max

    def test_first_sync_never_returns_zero(self) -> None:
        """Previously returned 0 (epoch) which fetched ALL history."""
        from trailtraining.pipelines.strava import _compute_after_unix

        result = _compute_after_unix([], {}, lookback_days=365)
        assert result > 0

    def test_incremental_sync_uses_meta_ts(self) -> None:
        from trailtraining.pipelines.strava import AFTER_BUFFER_SECONDS, _compute_after_unix

        meta = {"max_start_date_ts": 1700000000}
        result = _compute_after_unix([], meta)
        assert result == max(0, 1700000000 - AFTER_BUFFER_SECONDS)

    def test_incremental_sync_from_existing_activities(self) -> None:
        from trailtraining.pipelines.strava import AFTER_BUFFER_SECONDS, _compute_after_unix

        existing = [{"start_date": "2026-03-01T12:00:00Z"}]
        result = _compute_after_unix(existing, {})
        dt = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        expected = max(0, int(dt.timestamp()) - AFTER_BUFFER_SECONDS)
        assert result == expected

    def test_default_lookback_days_value(self) -> None:
        from trailtraining.pipelines.strava import DEFAULT_LOOKBACK_DAYS

        assert DEFAULT_LOOKBACK_DAYS == 365  # default from env


# ---------------------------------------------------------------------------
# Issue 5: ICS export timezone tests
# ---------------------------------------------------------------------------


class TestICSTimezone:
    """ICS export must not hardcode America/Los_Angeles."""

    def _make_artifact(self) -> Any:
        from trailtraining.contracts import TrainingPlanArtifact

        return TrainingPlanArtifact.model_validate(
            {
                "meta": {
                    "today": "2026-03-13",
                    "plan_start": "2026-03-14",
                    "plan_days": 7,
                    "style": "trailrunning",
                    "primary_goal": "test",
                },
                "snapshot": {
                    "last7": {
                        "distance_km": "22",
                        "moving_time_hours": "2.2",
                        "elevation_m": "300",
                        "activity_count": "2",
                        "sleep_hours_mean": "",
                        "hrv_mean": "",
                        "rhr_mean": "46",
                    },
                    "baseline28": {
                        "distance_km": "18",
                        "moving_time_hours": "1.9",
                        "elevation_m": "220",
                        "activity_count": "2",
                        "sleep_hours_mean": "",
                        "hrv_mean": "",
                        "rhr_mean": "45",
                    },
                    "notes": "",
                },
                "readiness": {
                    "status": "steady",
                    "rationale": "Test.",
                    "signal_ids": [],
                },
                "plan": {
                    "weekly_totals": {
                        "planned_distance_km": 30.0,
                        "planned_moving_time_hours": 4.5,
                        "planned_elevation_m": 600.0,
                    },
                    "days": [
                        {
                            "date": "2026-03-14",
                            "title": "Rest + mobility",
                            "session_type": "rest",
                            "is_rest_day": True,
                            "is_hard_day": False,
                            "duration_minutes": 0,
                            "target_intensity": "Off",
                            "terrain": "N/A",
                            "workout": "Rest day.",
                            "purpose": "Recovery",
                            "signal_ids": [],
                        },
                        {
                            "date": "2026-03-15",
                            "title": "Easy run",
                            "session_type": "easy",
                            "is_rest_day": False,
                            "is_hard_day": False,
                            "duration_minutes": 60,
                            "target_intensity": "Easy",
                            "terrain": "Flat",
                            "workout": "60 min easy run.",
                            "purpose": "Aerobic",
                            "signal_ids": [],
                        },
                    ],
                },
                "recovery": {"actions": [], "signal_ids": []},
                "risks": [],
                "data_notes": [],
                "citations": [],
            }
        )

    def test_no_hardcoded_los_angeles(self) -> None:
        """The old code had X-WR-TIMEZONE:America/Los_Angeles hardcoded."""
        from trailtraining.ics_export import plan_to_ics

        artifact = self._make_artifact()
        ics = plan_to_ics(artifact)
        assert "America/Los_Angeles" not in ics

    def test_default_floating_local_times(self) -> None:
        """Without timezone_id, events should use floating local times (no TZID)."""
        from trailtraining.ics_export import plan_to_ics

        artifact = self._make_artifact()
        ics = plan_to_ics(artifact, start_hour=7)

        # Should NOT contain TZID
        assert "TZID" not in ics
        # Should NOT contain X-WR-TIMEZONE
        assert "X-WR-TIMEZONE" not in ics
        # Timed event should have bare DTSTART (floating)
        assert "DTSTART:20260315T070000\r\n" in ics
        # All-day event should have VALUE=DATE
        assert "DTSTART;VALUE=DATE:20260314\r\n" in ics

    def test_explicit_timezone_emits_tzid(self) -> None:
        """When timezone_id is set, DTSTART should include TZID."""
        from trailtraining.ics_export import plan_to_ics

        artifact = self._make_artifact()
        ics = plan_to_ics(artifact, start_hour=8, timezone_id="Europe/Rome")

        assert "X-WR-TIMEZONE:Europe/Rome" in ics
        assert "DTSTART;TZID=Europe/Rome:20260315T080000\r\n" in ics
        assert "DTEND;TZID=Europe/Rome:" in ics

    def test_start_hour_respected(self) -> None:
        from trailtraining.ics_export import plan_to_ics

        artifact = self._make_artifact()
        ics = plan_to_ics(artifact, start_hour=6)
        assert "DTSTART:20260315T060000\r\n" in ics

    def test_duration_property_present(self) -> None:
        from trailtraining.ics_export import plan_to_ics

        artifact = self._make_artifact()
        ics = plan_to_ics(artifact)
        assert "DURATION:PT60M\r\n" in ics


# ---------------------------------------------------------------------------
# Issue 6: Shared retry utility tests
# ---------------------------------------------------------------------------


class TestSharedRetryUtility:
    """Verify the shared retry utility handles all expected error types."""

    def test_raises_external_service_error_on_4xx(self) -> None:
        from trailtraining.util.errors import ExternalServiceError
        from trailtraining.util.http_retry import request_with_retry

        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 403
        resp.text = "Forbidden"
        session.request.return_value = resp

        with pytest.raises(ExternalServiceError, match="403"):
            request_with_retry(session, "GET", "http://example.com", service_name="Test")

    def test_retries_on_429(self) -> None:
        from trailtraining.util.http_retry import request_with_retry

        session = MagicMock()
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "0"}
        resp_200 = MagicMock()
        resp_200.status_code = 200
        session.request.side_effect = [resp_429, resp_200]

        result = request_with_retry(session, "GET", "http://example.com", max_retries=3)
        assert result == resp_200

    def test_retries_on_5xx(self) -> None:
        from trailtraining.util.errors import ExternalServiceError
        from trailtraining.util.http_retry import request_with_retry

        session = MagicMock()
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Server error"
        session.request.return_value = resp_500

        with pytest.raises(ExternalServiceError, match="server error"):
            request_with_retry(session, "GET", "http://example.com", max_retries=2)

    def test_retries_on_timeout(self) -> None:
        import requests as req
        from trailtraining.util.errors import ExternalServiceError
        from trailtraining.util.http_retry import request_with_retry

        session = MagicMock()
        session.request.side_effect = req.Timeout("timed out")

        with pytest.raises(ExternalServiceError, match="failed after"):
            request_with_retry(session, "GET", "http://example.com", max_retries=2)

    def test_success_on_first_try(self) -> None:
        from trailtraining.util.http_retry import request_with_retry

        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        session.request.return_value = resp

        result = request_with_retry(session, "GET", "http://example.com")
        assert result == resp
        assert session.request.call_count == 1


# ---------------------------------------------------------------------------
# Issue 9: Additional regression tests
# ---------------------------------------------------------------------------


class TestContractValidationEdgeCases:
    """Ensure contract changes don't break existing serialization patterns."""

    def test_plan_day_date_serializes_as_iso_string(self) -> None:
        from trailtraining.contracts import PlanDay

        day = PlanDay(
            date="2026-03-14",
            title="Test",
            session_type="rest",
            is_rest_day=True,
            is_hard_day=False,
            duration_minutes=0,
            target_intensity="Off",
            terrain="N/A",
            workout="Rest",
            purpose="Recovery",
            signal_ids=[],
        )
        dumped = day.model_dump(mode="json")
        assert dumped["date"] == "2026-03-14"
        # Verify it can be loaded back
        PlanDay.model_validate(dumped)

    def test_training_meta_serializes_dates_as_iso(self) -> None:
        from trailtraining.contracts import TrainingMeta

        meta = TrainingMeta(
            today="2026-03-13",
            plan_start="2026-03-14",
            plan_days=7,
            style="trailrunning",
        )
        dumped = meta.model_dump(mode="json")
        assert dumped["today"] == "2026-03-13"
        assert dumped["plan_start"] == "2026-03-14"
