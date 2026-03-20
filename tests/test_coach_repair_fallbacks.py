from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from trailtraining.util.errors import MissingArtifactError

from tests._artifact_builders import make_training_plan_artifact


class _Response:
    def __init__(self, text: str) -> None:
        self.output_text = text


def test_apply_deterministic_readiness_updates_plan_and_notes() -> None:
    import trailtraining.llm.coach as coach

    plan_obj = make_training_plan_artifact()
    det = {
        "result": {
            "readiness": {
                "status": "fatigued",
                "score": 42,
            }
        }
    }

    coach._apply_deterministic_readiness(plan_obj, det)

    assert plan_obj["readiness"]["status"] == "fatigued"
    assert plan_obj["readiness"]["rationale"].startswith("Deterministic readiness: fatigued")
    assert any(
        "Readiness status was set from deterministic" in note for note in plan_obj["data_notes"]
    )


def test_parse_training_plan_invalid_output_triggers_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trailtraining.llm.coach as coach

    repaired = make_training_plan_artifact()

    monkeypatch.setattr(
        coach,
        "_call_with_param_fallback",
        lambda *args, **kwargs: _Response(json.dumps(repaired)),
    )

    parsed = coach._parse_training_plan(
        "not valid json",
        client=cast(Any, object()),
        cfg=coach.CoachConfig(),
        system_instructions="return json",
    )

    assert parsed["meta"]["plan_days"] == 7
    assert parsed["plan"]["weekly_totals"]["planned_moving_time_hours"] >= 0


def test_run_coach_brief_raises_when_combined_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trailtraining.llm.coach as coach

    runtime = SimpleNamespace(paths=SimpleNamespace(prompting_directory=tmp_path))
    source_data = SimpleNamespace(personal={"ok": True}, combined=[], rollups=None)

    monkeypatch.setattr(coach.config, "current", lambda: runtime)
    monkeypatch.setattr(coach.config, "ensure_directories", lambda runtime=None: None)
    coach_paths = SimpleNamespace(summary_path=tmp_path / "combined_summary.json")
    monkeypatch.setattr(coach, "resolve_input_paths", lambda *args, **kwargs: coach_paths)
    monkeypatch.setattr(coach, "load_coach_source_data", lambda *args, **kwargs: source_data)

    with pytest.raises(MissingArtifactError, match="contains no usable day objects"):
        coach.run_coach_brief(prompt="training-plan", cfg=coach.CoachConfig())


def test_run_coach_brief_training_plan_uses_repair_and_saves_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trailtraining.llm.coach as coach

    runtime = SimpleNamespace(paths=SimpleNamespace(prompting_directory=tmp_path))
    source_data = SimpleNamespace(
        personal={"athlete": "x"},
        combined=[{"date": "2026-03-01", "sleep": None, "activities": []}],
        rollups={"windows": {"7": {"activities": {"total_moving_time_hours": 4.0}}}},
    )
    repaired = make_training_plan_artifact()

    monkeypatch.setattr(coach.config, "current", lambda: runtime)
    monkeypatch.setattr(coach.config, "ensure_directories", lambda runtime=None: None)
    coach_paths = SimpleNamespace(summary_path=tmp_path / "combined_summary.json")
    monkeypatch.setattr(coach, "resolve_input_paths", lambda *args, **kwargs: coach_paths)
    monkeypatch.setattr(coach, "load_coach_source_data", lambda *args, **kwargs: source_data)
    monkeypatch.setattr(
        coach,
        "get_or_create_deterministic_forecast",
        lambda *args, **kwargs: {"result": {"readiness": {"status": "primed", "score": 88}}},
    )
    monkeypatch.setattr(coach, "_build_prompt_text", lambda *args, **kwargs: "PROMPT")
    monkeypatch.setattr(coach, "_make_openrouter_client", lambda: object())
    monkeypatch.setattr(
        coach, "_call_with_schema", lambda *args, **kwargs: _Response("not valid json")
    )
    monkeypatch.setattr(
        coach, "_call_with_param_fallback", lambda *args, **kwargs: _Response(json.dumps(repaired))
    )
    monkeypatch.setattr(
        coach,
        "apply_eval_coach_guardrails",
        lambda obj, rollups: obj["data_notes"].append("guardrails applied"),
    )
    monkeypatch.setattr(
        coach,
        "save_training_plan_output",
        lambda output_path, prompting_dir, plan_obj: tmp_path / "coach_brief_training-plan.json",
    )

    rendered, saved = coach.run_coach_brief(prompt="training-plan", cfg=coach.CoachConfig())
    payload = json.loads(rendered)

    assert saved is not None
    assert saved.endswith("coach_brief_training-plan.json")
    assert payload["readiness"]["status"] == "primed"
    assert "guardrails applied" in payload["data_notes"]


def test_run_coach_brief_markdown_path_uses_param_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trailtraining.llm.coach as coach

    runtime = SimpleNamespace(paths=SimpleNamespace(prompting_directory=tmp_path))
    source_data = SimpleNamespace(
        personal={"athlete": "x"},
        combined=[{"date": "2026-03-01", "sleep": None, "activities": []}],
        rollups=None,
    )

    monkeypatch.setattr(coach.config, "current", lambda: runtime)
    monkeypatch.setattr(coach.config, "ensure_directories", lambda runtime=None: None)
    coach_paths = SimpleNamespace(summary_path=tmp_path / "combined_summary.json")
    monkeypatch.setattr(coach, "resolve_input_paths", lambda *args, **kwargs: coach_paths)
    monkeypatch.setattr(coach, "load_coach_source_data", lambda *args, **kwargs: source_data)
    monkeypatch.setattr(coach, "get_or_create_deterministic_forecast", lambda *args, **kwargs: None)
    monkeypatch.setattr(coach, "_build_prompt_text", lambda *args, **kwargs: "PROMPT")
    monkeypatch.setattr(coach, "_make_openrouter_client", lambda: object())
    monkeypatch.setattr(
        coach, "_call_with_param_fallback", lambda *args, **kwargs: _Response("# coach brief")
    )
    monkeypatch.setattr(
        coach,
        "save_markdown_output",
        lambda output_path, prompt_name, prompting_dir, text: tmp_path / "coach_brief_weekly.md",
    )

    rendered, saved = coach.run_coach_brief(prompt="weekly", cfg=coach.CoachConfig())

    assert rendered == "# coach brief"
    assert saved is not None
    assert saved.endswith("coach_brief_weekly.md")
