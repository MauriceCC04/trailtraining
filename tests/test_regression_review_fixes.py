from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest


def _make_days(session_types: list[str]) -> list[dict[str, object]]:
    return [
        {
            "date": f"2026-03-{i + 1:02d}",
            "session_type": st,
            "is_rest_day": st == "rest",
            "is_hard_day": st in {"tempo", "intervals", "hills"},
            "duration_minutes": 0 if st == "rest" else 60,
        }
        for i, st in enumerate(session_types)
    ]


def test_guardrails_enforce_hard_limit_on_rolling_windows() -> None:
    from trailtraining.llm.guardrails import _enforce_max_hard_per_7d

    days = _make_days(
        [
            "easy",
            "easy",
            "easy",
            "easy",
            "tempo",
            "intervals",
            "hills",
            "tempo",
            "easy",
            "easy",
            "easy",
            "easy",
            "easy",
            "easy",
        ]
    )

    changed = _enforce_max_hard_per_7d(days, max_hard=3)
    assert changed

    for i in range(len(days) - 6):
        window = days[i : i + 7]
        hard = sum(1 for d in window if d["is_hard_day"] and not d["is_rest_day"])
        assert hard <= 3


def test_call_with_param_fallback_only_retries_unsupported_parameter_errors() -> None:
    from trailtraining.llm.coach import _call_with_param_fallback

    responses = MagicMock()
    responses.create.side_effect = [
        ValueError("unsupported parameter: reasoning"),
        types.SimpleNamespace(output_text="ok"),
    ]
    client: Any = types.SimpleNamespace(responses=responses)

    result = _call_with_param_fallback(
        cast(Any, client),
        {
            "model": "x",
            "input": "hi",
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "medium"},
        },
    )

    assert result.output_text == "ok"
    assert responses.create.call_count == 2


def test_call_with_param_fallback_does_not_retry_non_parameter_failures() -> None:
    from trailtraining.llm.coach import _call_with_param_fallback

    responses = MagicMock()
    responses.create.side_effect = RuntimeError("Invalid API key provided")
    client: Any = types.SimpleNamespace(responses=responses)

    with pytest.raises(RuntimeError, match="Invalid API key"):
        _call_with_param_fallback(
            cast(Any, client),
            {
                "model": "x",
                "input": "hi",
                "reasoning": {"effort": "medium"},
                "text": {"verbosity": "medium"},
            },
        )

    assert responses.create.call_count == 1


def test_strava_fetch_rejects_non_list_payload() -> None:
    from trailtraining.pipelines import strava as strava_mod
    from trailtraining.util.errors import DataValidationError

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(strava_mod, "_api_get", lambda *args, **kwargs: {"message": "bad"})
        with pytest.raises(DataValidationError, match="Unexpected Strava activities response"):
            strava_mod.fetch_activities_incremental(
                session=MagicMock(),
                access_token="token",
                after_unix=0,
                per_page=2,
                max_pages=1,
                hard_max_pages=2,
            )


def test_combine_main_reports_skipped_rollups_when_no_combined_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from trailtraining.data import combine as combine_mod

    processing = tmp_path / "processing"
    prompting = tmp_path / "prompting"
    processing.mkdir()
    prompting.mkdir()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(combine_mod.config, "ensure_directories", lambda: None)
        mp.setattr(combine_mod.config, "PROCESSING_DIRECTORY", str(processing))
        mp.setattr(combine_mod.config, "PROMPTING_DIRECTORY", str(prompting))
        mp.setattr(combine_mod, "build_formatted_personal_profile", lambda **kwargs: None)

        combine_mod.main()

    out = capsys.readouterr().out
    assert "Skipped rollups:" in out


def test_deterministic_forecast_logs_save_failures_but_returns_payload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    from typing import Any, Callable

    from trailtraining.llm import coach as coach_mod

    fake_module = types.ModuleType("trailtraining.forecast.forecast")

    class FakeForecast:
        def __init__(self) -> None:
            self.date = "2026-03-17"
            self.readiness_score = 60
            self.readiness_status = "steady"
            self.overreach_risk_score = 20
            self.overreach_risk_level = "low"
            self.inputs: dict[str, int] = {"x": 1}
            self.drivers: list[str] = []

    def fake_compute_readiness_and_risk(combined: list[dict[str, object]]) -> FakeForecast:
        return FakeForecast()

    fake_module_any = cast(Any, fake_module)
    fake_module_any.compute_readiness_and_risk = cast(
        Callable[[list[dict[str, object]]], FakeForecast],
        fake_compute_readiness_and_risk,
    )

    monkeypatch.setitem(sys.modules, "trailtraining.forecast.forecast", fake_module)
    monkeypatch.setattr(
        coach_mod,
        "save_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    caplog.set_level("WARNING")
    payload = coach_mod._load_or_compute_deterministic_forecast(tmp_path, combined=[])

    assert isinstance(payload, dict)
    assert payload["result"]["readiness"]["status"] == "steady"
    assert "Failed to persist deterministic forecast" in caplog.text
