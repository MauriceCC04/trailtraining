from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._artifact_builders import make_evaluation_report_artifact, make_training_plan_artifact


class _Response:
    def __init__(self, text: str) -> None:
        self.output_text = text


def test_summarize_eval_targets_includes_soft_sections() -> None:
    from trailtraining.llm.revise import _summarize_eval_targets

    report = {
        "violations": [{"severity": "high", "code": "MAX_RAMP_PCT", "message": "Ramp too high."}],
        "soft_assessment": {
            "summary": "Overall solid.",
            "strengths": ["Good long run"],
            "concerns": ["Too much intensity"],
            "suggested_improvements": ["Add a rest day"],
        },
    }

    lines = _summarize_eval_targets(report)
    text = "\n".join(lines)

    assert "## Deterministic issues to fix" in text
    assert "## Soft assessment summary" in text
    assert "## Strengths to preserve" in text
    assert "## Suggested improvements to implement" in text


def test_build_revise_prompt_adds_multiweek_rules() -> None:
    import trailtraining.llm.revise as revise

    plan = make_training_plan_artifact(plan_days=14)
    report = make_evaluation_report_artifact()

    prompt = revise._build_revise_prompt(
        plan,
        report,
        style="trailrunning",
        primary_goal="build trail endurance",
        lifestyle_notes="weekday road only",
    )

    assert "## Multi-week plan rules (plan_days > 7)" in prompt
    assert "The revised plan MUST contain exactly 14 days" in prompt
    assert "weekday road only" in prompt


def test_run_revise_plan_invalid_first_output_triggers_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trailtraining.llm.revise as revise

    runtime = SimpleNamespace(paths=SimpleNamespace(prompting_directory=tmp_path))
    plan = make_training_plan_artifact()
    report = make_evaluation_report_artifact()
    repaired = make_training_plan_artifact()
    input_plan_path = tmp_path / "plan.json"
    eval_report_path = tmp_path / "eval_report.json"
    input_plan_path.write_text(json.dumps(plan), encoding="utf-8")
    eval_report_path.write_text(json.dumps(report), encoding="utf-8")

    calls = {"n": 0}

    def fake_call_with_schema(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response("not valid json")
        return _Response(json.dumps(repaired))

    monkeypatch.setattr(revise.config, "current", lambda: runtime)
    monkeypatch.setattr(revise.config, "ensure_directories", lambda runtime=None: None)
    monkeypatch.setattr(revise, "_make_openrouter_client", lambda: object())
    monkeypatch.setattr(revise, "_call_with_schema", fake_call_with_schema)
    monkeypatch.setattr(
        revise, "_load_rollups_near", lambda *args, **kwargs: {"windows": {"7": {}}}
    )
    monkeypatch.setattr(
        revise,
        "apply_eval_coach_guardrails",
        lambda obj, rollups: obj["data_notes"].append("guardrails applied"),
    )
    monkeypatch.setattr(revise, "training_plan_to_text", lambda obj: "plan text")

    rendered, saved = revise.run_revise_plan(
        cfg=revise.RevisePlanConfig(),
        input_plan_path=str(input_plan_path),
        eval_report_path=str(eval_report_path),
    )
    payload = json.loads(rendered)

    assert calls["n"] == 2
    assert saved.endswith("revised-plan.json")
    assert payload["meta"]["lifestyle_notes"] == "weekday road only"
    assert "guardrails applied" in payload["data_notes"]


def test_run_auto_reeval_handles_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import trailtraining.llm.revise as revise

    revised_plan = tmp_path / "revised-plan.json"
    revised_plan.write_text("{}", encoding="utf-8")

    def fake_eval(*args, **kwargs):
        raise RuntimeError("reeval failed")

    monkeypatch.setattr(
        __import__("trailtraining.llm.eval", fromlist=["evaluate_training_plan_quality_file"]),
        "evaluate_training_plan_quality_file",
        fake_eval,
    )

    revise._run_auto_reeval(
        revised_plan_path=revised_plan,
        original_report_obj={"score": 80},
        rollups_path=None,
    )


def test_run_auto_reeval_warns_on_degraded_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import trailtraining.llm.revise as revise

    revised_plan = tmp_path / "revised-plan.json"
    revised_plan.write_text("{}", encoding="utf-8")

    def fake_eval(*args, **kwargs):
        return ({"score": 70, "violations": [], "grade": "C"}, {})

    monkeypatch.setattr(
        __import__("trailtraining.llm.eval", fromlist=["evaluate_training_plan_quality_file"]),
        "evaluate_training_plan_quality_file",
        fake_eval,
    )

    revise._run_auto_reeval(
        revised_plan_path=revised_plan,
        original_report_obj={"score": 80},
        rollups_path=None,
    )

    output = capsys.readouterr().out
    assert "Revision degraded deterministic score" in output
