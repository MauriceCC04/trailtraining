from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from trailtraining.contracts import EvaluationReportArtifact, ForecastArtifact
from trailtraining.util.state import save_json


def write_combined_summary(path: Path) -> None:
    sample = [
        {
            "date": "2026-03-01",
            "sleep": {"restingHeartRate": 45},
            "activities": [
                {
                    "id": 1,
                    "sport_type": "Run",
                    "distance": 10000,
                    "moving_time": 3600,
                    "average_heartrate": 150,
                }
            ],
        },
        {
            "date": "2026-03-02",
            "sleep": {"restingHeartRate": 47},
            "activities": [
                {
                    "id": 2,
                    "sport_type": "Run",
                    "distance": 12000,
                    "moving_time": 4200,
                    "average_heartrate": 152,
                }
            ],
        },
    ]
    save_json(path, sample, compact=False)


def write_rollups(path: Path) -> None:
    payload = {
        "windows": {
            "7": {
                "end_date": "2026-03-02",
                "activities": {
                    "total_moving_time_hours": 2.166,
                    "total_training_load_hours": 2.166,
                },
            }
        }
    }
    save_json(path, payload, compact=False)


def write_training_plan(path: Path) -> None:
    payload = {
        "meta": {
            "today": "2026-03-02",
            "plan_start": "2026-03-03",
            "plan_days": 7,
            "style": "trailrunning",
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
            "rationale": "Stable recent load.",
            "signal_ids": ["forecast.readiness.status"],
        },
        "plan": {
            "weekly_totals": {
                "planned_distance_km": 30.0,
                "planned_moving_time_hours": 4.5,
                "planned_elevation_m": 600.0,
            },
            "days": [
                {
                    "date": "2026-03-03",
                    "title": "Easy run",
                    "session_type": "easy",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 45,
                    "target_intensity": "easy",
                    "terrain": "road",
                    "workout": "45 min easy",
                    "purpose": "aerobic maintenance",
                    "signal_ids": ["forecast.readiness.status"],
                }
            ],
        },
        "recovery": {
            "actions": ["Sleep 8h"],
            "signal_ids": ["forecast.readiness.status"],
        },
        "risks": [],
        "data_notes": [],
        "citations": [
            {
                "signal_id": "forecast.readiness.status",
                "source": "readiness_and_risk_forecast.json:result.readiness.status",
                "date_range": "2026-03-02..2026-03-02",
                "value": "steady",
            }
        ],
    }
    save_json(path, payload, compact=False)


def test_forecast_cli_writes_valid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_combined_summary(tmp_path / "combined_summary.json")
    write_rollups(tmp_path / "combined_rollups.json")

    from trailtraining.cli import main

    outp = tmp_path / "forecast.json"
    main(["forecast", "--input", str(tmp_path), "--output", str(outp)])

    payload = json.loads(outp.read_text(encoding="utf-8"))
    artifact = ForecastArtifact.model_validate(payload)

    assert artifact.result.readiness.status in {"primed", "steady", "fatigued"}
    assert artifact.result.overreach_risk.level in {"low", "moderate", "high"}


def test_eval_coach_cli_writes_valid_report(tmp_path: Path) -> None:
    write_training_plan(tmp_path / "coach_brief_training-plan.json")
    write_rollups(tmp_path / "combined_rollups.json")

    from trailtraining.cli import main

    report_path = tmp_path / "eval_report.json"
    violations_path = tmp_path / "violations.json"

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "eval-coach",
                "--input",
                str(tmp_path / "coach_brief_training-plan.json"),
                "--report",
                str(report_path),
                "--output",
                str(violations_path),
            ]
        )

    assert exc.value.code in {0, 1}

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report = EvaluationReportArtifact.model_validate(payload)
    assert isinstance(report.violations, list)


def test_run_all_cli_calls_pipeline_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAILTRAINING_BASE_DIR", str(tmp_path))

    import trailtraining.data.combine as combine_mod
    import trailtraining.pipelines.intervals as intervals_mod
    import trailtraining.pipelines.run_all as run_all
    import trailtraining.pipelines.strava as strava_mod
    from trailtraining import config

    importlib.reload(config)
    importlib.reload(run_all)

    calls: list[str] = []

    def fake_intervals_main() -> None:
        calls.append("intervals")

    def fake_strava_main() -> None:
        calls.append("strava")

    def fake_combine_main() -> None:
        calls.append("combine")

    monkeypatch.setattr(run_all, "_detect_provider", lambda explicit=None: "intervals")
    monkeypatch.setattr(intervals_mod, "main", fake_intervals_main)
    monkeypatch.setattr(strava_mod, "main", fake_strava_main)
    monkeypatch.setattr(combine_mod, "main", fake_combine_main)

    from trailtraining.cli import main

    main(["run-all"])

    assert calls == ["intervals", "strava", "combine"]
