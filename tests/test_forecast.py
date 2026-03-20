"""tests/test_forecast.py — unit tests for trailtraining.forecast.forecast"""

from __future__ import annotations

import json
import types
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from trailtraining.forecast.forecast import compute_readiness_and_risk, run_forecasts


def _day(
    date_str: str,
    *,
    moving_time: int = 3600,
    sleep_secs: int = 8 * 3600,
    rhr: int = 50,
    hrv: int = 60,
) -> dict[str, Any]:
    return {
        "date": date_str,
        "sleep": {
            "sleepTimeSeconds": sleep_secs,
            "restingHeartRate": rhr,
            "avgOvernightHrv": hrv,
        },
        "activities": [
            {
                "id": hash(date_str),
                "sport_type": "Run",
                "distance": 10000,
                "moving_time": moving_time,
                "total_elevation_gain": 100,
            }
        ],
    }


def _steady_block(days: int, *, start: date = date(2026, 2, 1)) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(days):
        d = start + timedelta(days=i)
        out.append(_day(d.isoformat()))
    return out


class TestForecastAtlCtlTsb:
    def test_compute_readiness_and_risk_exposes_load_model_inputs(self) -> None:
        combined = _steady_block(35)

        result = compute_readiness_and_risk(combined)

        assert result.inputs["atl_load_hours"] is not None
        assert result.inputs["ctl_load_hours"] is not None
        assert result.inputs["tsb_load_hours"] is not None
        assert any("ATL/CTL/TSB computed" in note for note in result.inputs["notes"])

    def test_compute_readiness_and_risk_still_scores_normally(self) -> None:
        combined = _steady_block(35)

        result = compute_readiness_and_risk(combined)

        assert result.readiness_status in {"primed", "steady", "fatigued"}
        assert result.overreach_risk_level in {"low", "moderate", "high"}
        assert isinstance(result.readiness_score, float)
        assert isinstance(result.overreach_risk_score, float)

    def test_run_forecasts_writes_valid_artifact_with_load_model_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trailtraining.forecast import forecast as forecast_mod

        prompting = tmp_path / "prompting"
        prompting.mkdir(parents=True, exist_ok=True)

        combined = _steady_block(35)
        (prompting / "combined_summary.json").write_text(
            json.dumps(combined),
            encoding="utf-8",
        )

        runtime = types.SimpleNamespace(paths=types.SimpleNamespace(prompting_directory=prompting))

        monkeypatch.setattr(forecast_mod.config, "current", lambda: runtime)
        monkeypatch.setattr(forecast_mod.config, "ensure_directories", lambda runtime=None: None)

        out = run_forecasts()

        saved_path = Path(out["saved"])
        assert saved_path.exists()

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        inputs = payload["result"]["inputs"]

        assert inputs["atl_load_hours"] is not None
        assert inputs["ctl_load_hours"] is not None
        assert inputs["tsb_load_hours"] is not None
        assert any("ATL/CTL/TSB computed" in note for note in inputs["notes"])
