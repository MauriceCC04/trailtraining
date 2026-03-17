from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import trailtraining.llm as llm_pkg
import trailtraining.util as util_pkg
from trailtraining.commands import llm_commands as lc


def _install_module(monkeypatch, package, full_name: str, attr_name: str, **attrs):
    module = ModuleType(full_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, full_name, module)
    monkeypatch.setattr(package, attr_name, module, raising=False)
    return module


@dataclass
class FakeCoachConfig:
    model: str
    reasoning_effort: str
    verbosity: str
    days: int
    max_chars: int
    temperature: float | None = None
    style: str | None = None
    primary_goal: str | None = None

    @classmethod
    def from_env(cls):
        return cls(
            model="base-model",
            reasoning_effort="medium",
            verbosity="low",
            days=7,
            max_chars=4000,
            temperature=None,
            style=None,
            primary_goal="base-goal",
        )


@dataclass
class FakeSoftEvalConfig:
    enabled: bool = False
    model: str = "soft-model"
    reasoning_effort: str = "medium"
    verbosity: str = "low"
    primary_goal: str = "consistency"

    @classmethod
    def from_env(cls):
        return cls()


@dataclass
class FakeRevisePlanConfig:
    model: str
    reasoning_effort: str
    verbosity: str
    temperature: float | None = None
    primary_goal: str | None = None


def test_cmd_coach_builds_config_and_prints_saved_files(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run_coach_brief(**kwargs):
        captured.update(kwargs)
        out_path = Path(kwargs["output_path"])
        out_path.write_text("{}", encoding="utf-8")
        txt_path = out_path.parent / f"{out_path.stem}.txt"
        txt_path.write_text("summary", encoding="utf-8")
        return "coach output", str(out_path)

    monkeypatch.setattr(lc, "_run", lambda func: func())
    monkeypatch.setattr(lc, "default_primary_goal_for_style", lambda style: f"default-{style}")
    monkeypatch.setenv("TRAILTRAINING_PRIMARY_GOAL", "env-goal")
    _install_module(
        monkeypatch,
        llm_pkg,
        "trailtraining.llm.coach",
        "coach",
        CoachConfig=FakeCoachConfig,
        run_coach_brief=fake_run_coach_brief,
    )

    args = argparse.Namespace(
        style=None,
        goal=None,
        model=None,
        reasoning_effort=None,
        verbosity=None,
        days=None,
        max_chars=None,
        temperature=0.2,
        prompt="training-plan",
        input="combined.json",
        personal="personal.json",
        summary="summary.json",
        output=str(tmp_path / "coach_brief_training-plan.json"),
    )

    lc.cmd_coach(args)

    out = capsys.readouterr().out
    assert "coach output" in out
    assert "coach_brief_training-plan.txt" in out
    assert captured["prompt"] == "training-plan"
    assert captured["cfg"].model == "base-model"
    assert captured["cfg"].temperature == 0.2
    assert captured["cfg"].style == "trailrunning"
    assert captured["cfg"].primary_goal == "env-goal"


def test_cmd_eval_coach_prints_report_and_exits_zero(monkeypatch, tmp_path, capsys):
    saved = []

    monkeypatch.setattr(lc, "_run", lambda func: func())
    monkeypatch.setattr("trailtraining.config.PROMPTING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(
        "trailtraining.llm.constraints.constraint_config_from_env",
        lambda **kwargs: kwargs,
    )
    _install_module(
        monkeypatch,
        llm_pkg,
        "trailtraining.llm.eval",
        "eval",
        evaluate_training_plan_quality_file=lambda *args, **kwargs: (
            {
                "score": 88,
                "grade": "B",
                "subscores": {"safety": 95},
                "soft_assessment": {
                    "overall_score": 91,
                    "grade": "A",
                    "rubric_scores": {"specificity": {"score": 92}},
                },
                "violations": [],
            },
            {},
        ),
    )
    _install_module(
        monkeypatch,
        llm_pkg,
        "trailtraining.llm.soft_eval",
        "soft_eval",
        SoftEvalConfig=FakeSoftEvalConfig,
    )
    _install_module(
        monkeypatch,
        util_pkg,
        "trailtraining.util.state",
        "state",
        save_json=lambda path, obj, compact=False: saved.append((Path(path), obj, compact)),
    )

    args = argparse.Namespace(
        input=None,
        report=None,
        max_ramp_pct=12.0,
        max_consecutive_hard=3,
        soft_eval=True,
        soft_eval_model=None,
        soft_eval_reasoning_effort=None,
        soft_eval_verbosity=None,
        goal="race",
        output=str(tmp_path / "violations.json"),
        rollups="rollups.json",
    )

    with pytest.raises(SystemExit) as exc:
        lc.cmd_eval_coach(args)

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Deterministic score: 88/100 (B)" in out
    assert "Soft quality score: 91/100 (A)" in out
    assert "Rubric scores: specificity=92" in out
    assert "no violations" in out
    assert saved[0][0] == tmp_path / "violations.json"
    assert saved[1][0] == tmp_path / "eval_report.json"


def test_cmd_eval_coach_exits_one_for_high_severity_violation(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lc, "_run", lambda func: func())
    monkeypatch.setattr("trailtraining.config.PROMPTING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(
        "trailtraining.llm.constraints.constraint_config_from_env",
        lambda **kwargs: kwargs,
    )
    _install_module(
        monkeypatch,
        llm_pkg,
        "trailtraining.llm.eval",
        "eval",
        evaluate_training_plan_quality_file=lambda *args, **kwargs: (
            {
                "score": 50,
                "grade": "F",
                "subscores": {},
                "violations": [
                    {
                        "severity": "high",
                        "code": "FORECAST_HARD_DAY_LIMIT",
                        "message": "too many hard days",
                    }
                ],
            },
            {},
        ),
    )
    _install_module(
        monkeypatch,
        llm_pkg,
        "trailtraining.llm.soft_eval",
        "soft_eval",
        SoftEvalConfig=FakeSoftEvalConfig,
    )
    _install_module(
        monkeypatch,
        util_pkg,
        "trailtraining.util.state",
        "state",
        save_json=lambda *args, **kwargs: None,
    )

    args = argparse.Namespace(
        input=None,
        report=None,
        max_ramp_pct=12.0,
        max_consecutive_hard=3,
        soft_eval=False,
        goal=None,
        output=None,
        rollups=None,
    )

    with pytest.raises(SystemExit) as exc:
        lc.cmd_eval_coach(args)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FORECAST_HARD_DAY_LIMIT" in out
    assert "too many hard days" in out


def test_cmd_revise_plan_prints_saved_artifacts(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lc, "_run", lambda func: func())
    monkeypatch.setattr("trailtraining.config.PROMPTING_DIRECTORY", str(tmp_path))
    _install_module(
        monkeypatch,
        llm_pkg,
        "trailtraining.llm.coach",
        "coach",
        CoachConfig=FakeCoachConfig,
    )
    _install_module(
        monkeypatch,
        llm_pkg,
        "trailtraining.llm.revise",
        "revise",
        RevisePlanConfig=FakeRevisePlanConfig,
        run_revise_plan=lambda **kwargs: (
            "revised plan",
            str(tmp_path / "coach_brief_training-plan.revised.json"),
        ),
    )

    revised_json = tmp_path / "coach_brief_training-plan.revised.json"
    revised_json.write_text("{}", encoding="utf-8")
    revised_txt = revised_json.parent / f"{revised_json.stem}.txt"
    revised_txt.write_text("summary", encoding="utf-8")

    args = argparse.Namespace(
        model=None,
        reasoning_effort=None,
        verbosity=None,
        temperature=0.1,
        goal="B-race",
        input=None,
        report=None,
        output=str(revised_json),
        rollups="rollups.json",
    )

    lc.cmd_revise_plan(args)

    out = capsys.readouterr().out
    assert "revised plan" in out
    assert "coach_brief_training-plan.revised.json" in out
    assert "coach_brief_training-plan.revised.txt" in out
