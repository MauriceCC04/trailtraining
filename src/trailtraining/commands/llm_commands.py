from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from trailtraining.commands.common import _run
from trailtraining.llm.rubrics import default_primary_goal_for_style


def _resolve_lifestyle_notes(args: argparse.Namespace) -> str:
    """Resolve lifestyle notes from CLI flag or env var."""
    cli_val: str = str(getattr(args, "lifestyle_notes", None) or "")
    if cli_val.strip():
        return cli_val.strip()
    return (os.getenv("TRAILTRAINING_LIFESTYLE_NOTES") or "").strip()


def _format_score(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return str(value)

    if score.is_integer():
        return str(int(score))
    return f"{score:.1f}"


def _require_output_path(value: str | None, *, step: str) -> str:
    if not value:
        raise RuntimeError(f"{step} did not produce an output path.")
    return str(Path(value).expanduser().resolve())


def _default_eval_report_path(prompting_dir: Path, input_path: str) -> tuple[Path, list[Path]]:
    in_path = Path(input_path).expanduser().resolve()
    name = in_path.name

    if name == "revised-plan.json":
        return prompting_dir / "eval_report.revised.json", []

    return prompting_dir / "eval_report.json", []


def _build_soft_eval_cfg(args: argparse.Namespace, *, enabled: bool) -> Any:
    from trailtraining.llm.soft_eval import SoftEvalConfig

    if not enabled:
        return None

    default_soft = SoftEvalConfig.from_env()
    lifestyle_notes = _resolve_lifestyle_notes(args) or default_soft.lifestyle_notes
    return SoftEvalConfig(
        enabled=True,
        model=getattr(args, "soft_eval_model", None) or default_soft.model,
        reasoning_effort=(
            getattr(args, "soft_eval_reasoning_effort", None) or default_soft.reasoning_effort
        ),
        verbosity=(getattr(args, "soft_eval_verbosity", None) or default_soft.verbosity),
        primary_goal=getattr(args, "goal", None) or default_soft.primary_goal,
        lifestyle_notes=lifestyle_notes,
        skip_synthesis=getattr(args, "skip_synthesis", False) or default_soft.skip_synthesis,
        parallel_batches=(
            not getattr(args, "no_parallel_batches", False) and default_soft.parallel_batches
        ),
    )


def _print_eval_summary(report: dict[str, Any]) -> None:
    deterministic_score = report.get("deterministic_score", report.get("score", 0))
    deterministic_grade = report.get("deterministic_grade", report.get("grade", "?"))
    overall_score = report.get("score", deterministic_score)
    overall_grade = report.get("grade", deterministic_grade)

    print(f"Deterministic score: {_format_score(deterministic_score)}/100 ({deterministic_grade})")
    if (
        report.get("soft_assessment")
        or str(overall_grade) != str(deterministic_grade)
        or _format_score(overall_score) != _format_score(deterministic_score)
    ):
        print(f"Overall score: {_format_score(overall_score)}/100 ({overall_grade})")

    subs = report.get("subscores", {}) or {}
    if subs:
        parts = [f"{k}={subs[k]}" for k in sorted(subs.keys())]
        print("Deterministic subscores:", ", ".join(parts))

    soft = report.get("soft_assessment") or {}
    if isinstance(soft, dict) and soft:
        soft_score = soft.get("overall_score", overall_score)
        soft_grade = soft.get("grade", overall_grade)
        print(f"Soft quality score: {_format_score(soft_score)}/100 ({soft_grade})")

        rubric_scores = soft.get("rubric_scores", {}) or {}
        if rubric_scores:
            parts = []
            for key in sorted(rubric_scores.keys()):
                item = rubric_scores.get(key) or {}
                parts.append(f"{key}={item.get('score', 0)}")
            print("Rubric scores:", ", ".join(parts))

        stats = report.get("stats", {}) or {}
        n_runs = stats.get("inter_rater_runs")
        if n_runs and int(n_runs) > 1:
            print(f"Inter-rater runs: {n_runs}")
            method = str(stats.get("inter_rater_consensus_method", "") or "").strip()
            if method:
                print(f"Consensus method: {method}")
            high_var = stats.get("high_variance_markers", {})
            if high_var:
                flagged = ", ".join(
                    f"{mid}(std={std:.2f})" for mid, std in sorted(high_var.items())
                )
                print(f"High-variance markers (ambiguous rubrics): {flagged}")


def _print_eval_violations(violations: list[dict[str, Any]]) -> None:
    if not violations:
        print("eval-coach: no violations")
        return

    print("eval-coach violations:")
    for violation in violations:
        sev = violation.get("severity", "unknown")
        code = violation.get("code", "UNKNOWN")
        msg = violation.get("message", "")
        print(f"- [{sev}] {code}: {msg}")


def _evaluate_training_plan(
    args: argparse.Namespace,
    *,
    input_path: str,
    report_path: str | None = None,
    output_path: str | None = None,
    soft_eval_enabled: bool | None = None,
    soft_eval_runs: int | None = None,
    exit_on_violations: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    from trailtraining import config
    from trailtraining.llm.constraints import constraint_config_from_env
    from trailtraining.llm.eval import evaluate_training_plan_quality_file
    from trailtraining.util.state import save_json

    runtime = config.current()
    prompting_dir = Path(runtime.paths.prompting_directory)

    in_path = str(Path(input_path).expanduser().resolve())
    primary_report_path, aliases = _default_eval_report_path(prompting_dir, in_path)
    explicit_report_path = Path(report_path).expanduser().resolve() if report_path else None
    final_report_path = explicit_report_path or primary_report_path

    cfg = constraint_config_from_env(
        max_ramp_pct=float(args.max_ramp_pct),
        max_consecutive_hard=int(args.max_consecutive_hard),
    )

    enabled = (
        bool(getattr(args, "soft_eval", False)) if soft_eval_enabled is None else soft_eval_enabled
    )
    soft_cfg = _build_soft_eval_cfg(args, enabled=enabled)
    runs = max(
        1,
        int(
            (soft_eval_runs if soft_eval_runs is not None else getattr(args, "soft_eval_runs", 1))
            or 1
        ),
    )

    report, _obj = evaluate_training_plan_quality_file(
        in_path,
        rollups_path=getattr(args, "rollups", None),
        cfg=cfg,
        soft_eval_cfg=soft_cfg,
        primary_goal=getattr(args, "goal", None),
        soft_eval_runs=runs,
    )
    violations = report.get("violations", []) or []

    if output_path:
        outp = Path(output_path).expanduser().resolve()
        save_json(outp, violations, compact=False)
        print(f"[Saved] {outp}")

    save_json(final_report_path, report, compact=False)
    print(f"[Saved] {final_report_path}")
    if explicit_report_path is None:
        for alias in aliases:
            if alias == final_report_path:
                continue
            save_json(alias, report, compact=False)
            print(f"[Saved] {alias}")

    _print_eval_summary(report)
    _print_eval_violations(violations)

    if exit_on_violations:
        if not violations:
            raise SystemExit(0)
        if any(v.get("severity") == "high" for v in violations if isinstance(v, dict)):
            raise SystemExit(1)
        raise SystemExit(0)

    return report, violations, final_report_path


def cmd_coach(args: argparse.Namespace) -> None:
    from trailtraining.llm.coach import CoachConfig, run_coach_brief

    def _inner() -> None:
        base_cfg = CoachConfig.from_env()
        style = args.style or base_cfg.style or "trailrunning"
        primary_goal = (
            args.goal
            or os.getenv("TRAILTRAINING_PRIMARY_GOAL")
            or default_primary_goal_for_style(style)
        )
        lifestyle_notes = _resolve_lifestyle_notes(args) or base_cfg.lifestyle_notes

        cfg = CoachConfig(
            model=args.model or base_cfg.model,
            reasoning_effort=args.reasoning_effort or base_cfg.reasoning_effort,
            verbosity=args.verbosity or base_cfg.verbosity,
            days=args.days if args.days is not None else base_cfg.days,
            max_chars=args.max_chars if args.max_chars is not None else base_cfg.max_chars,
            temperature=args.temperature,
            style=style,
            primary_goal=primary_goal,
            plan_days=getattr(args, "plan_days", None) or base_cfg.plan_days,
            lifestyle_notes=lifestyle_notes,
        )
        text, out_path = run_coach_brief(
            prompt=args.prompt,
            cfg=cfg,
            input_path=args.input,
            personal_path=args.personal,
            summary_path=args.summary,
            output_path=args.output,
        )
        print(text)
        if out_path:
            print(f"\n[Saved] {out_path}")
            if args.prompt == "training-plan":
                p = Path(out_path)
                txt_p = p.parent / f"{p.stem}.txt"
                if txt_p.exists():
                    print(f"[Saved] {txt_p}")

    _run(_inner)


def cmd_eval_coach(args: argparse.Namespace) -> None:
    def _inner() -> None:
        from trailtraining import config

        runtime = config.current()
        prompting_dir = Path(runtime.paths.prompting_directory)
        input_path = args.input or str(prompting_dir / "coach_brief_training-plan.json")

        _evaluate_training_plan(
            args,
            input_path=input_path,
            report_path=getattr(args, "report", None),
            output_path=getattr(args, "output", None),
            soft_eval_enabled=bool(getattr(args, "soft_eval", False)),
            soft_eval_runs=getattr(args, "soft_eval_runs", 1),
            exit_on_violations=True,
        )

    _run(_inner)


def cmd_revise_plan(args: argparse.Namespace) -> None:
    from trailtraining import config
    from trailtraining.llm.coach import CoachConfig
    from trailtraining.llm.revise import RevisePlanConfig, run_revise_plan

    def _inner() -> None:
        runtime = config.current()
        prompting_dir = Path(runtime.paths.prompting_directory)

        base_cfg = CoachConfig.from_env()
        lifestyle_notes = _resolve_lifestyle_notes(args) or base_cfg.lifestyle_notes

        revise_cfg = RevisePlanConfig(
            model=args.model or base_cfg.model,
            reasoning_effort=args.reasoning_effort or base_cfg.reasoning_effort,
            verbosity=args.verbosity or base_cfg.verbosity,
            temperature=args.temperature,
            primary_goal=args.goal or base_cfg.primary_goal,
            lifestyle_notes=lifestyle_notes,
        )

        input_path = args.input or str(prompting_dir / "coach_brief_training-plan.json")
        report_path = args.report or str(prompting_dir / "eval_report.json")
        auto_reeval = bool(getattr(args, "auto_reeval", False))

        text, out_path = run_revise_plan(
            cfg=revise_cfg,
            input_plan_path=input_path,
            eval_report_path=report_path,
            output_path=args.output,
            rollups_path=args.rollups,
            auto_reeval=auto_reeval,
        )

        print(text)
        if out_path:
            print(f"\n[Saved] {out_path}")
            p = Path(out_path)
            txt_p = p.parent / f"{p.stem}.txt"
            if txt_p.exists():
                print(f"[Saved] {txt_p}")
            comparison_p = p.parent / f"{p.stem}-comparison.json"
            if comparison_p.exists():
                print(f"[Saved] {comparison_p}")
            selected_json = p.parent / "selected-plan.json"
            selected_txt = p.parent / "selected-plan.txt"
            if selected_json.exists():
                print(f"[Saved] {selected_json}")
            if selected_txt.exists():
                print(f"[Saved] {selected_txt}")
            if auto_reeval:
                reeval_p = p.parent / f"{p.stem}-reeval.json"
                if reeval_p.exists():
                    print(f"[Saved] {reeval_p}")

    _run(_inner)


def cmd_run_training_cycle(args: argparse.Namespace) -> None:
    from trailtraining import config
    from trailtraining.forecast.forecast import run_forecasts
    from trailtraining.llm.coach import CoachConfig, run_coach_brief
    from trailtraining.llm.revise import RevisePlanConfig, run_revise_plan
    from trailtraining.pipelines import run_all_intervals

    def _inner() -> None:
        runtime = config.current()
        prompting_dir = Path(runtime.paths.prompting_directory)

        print("==> Step 1/6: run-all-intervals")
        run_all_intervals.main(
            clean=getattr(args, "clean", False),
            clean_processing=getattr(args, "clean_processing", False),
            clean_prompting=getattr(args, "clean_prompting", False),
        )

        print("==> Step 2/6: forecast")
        forecast_result = run_forecasts(input_dir=str(prompting_dir), output_path=None)
        print(f"[Saved] {forecast_result['saved']}")
        print(forecast_result["result"])

        print("==> Step 3/6: coach training-plan")
        base_cfg = CoachConfig.from_env()
        style = args.style or base_cfg.style or "trailrunning"
        primary_goal = (
            args.goal
            or os.getenv("TRAILTRAINING_PRIMARY_GOAL")
            or default_primary_goal_for_style(style)
        )
        lifestyle_notes = _resolve_lifestyle_notes(args) or base_cfg.lifestyle_notes
        coach_cfg = CoachConfig(
            model=args.model or base_cfg.model,
            reasoning_effort=args.reasoning_effort or base_cfg.reasoning_effort,
            verbosity=args.verbosity or base_cfg.verbosity,
            days=args.days if getattr(args, "days", None) is not None else base_cfg.days,
            max_chars=(
                args.max_chars
                if getattr(args, "max_chars", None) is not None
                else base_cfg.max_chars
            ),
            temperature=args.temperature,
            style=style,
            primary_goal=primary_goal,
            plan_days=getattr(args, "plan_days", None) or 28,
            lifestyle_notes=lifestyle_notes,
        )
        plan_text, raw_plan_path = run_coach_brief(
            prompt="training-plan",
            cfg=coach_cfg,
            input_path=getattr(args, "input", None),
            personal_path=getattr(args, "personal", None),
            summary_path=getattr(args, "summary", None),
            output_path=None,
        )
        plan_path = _require_output_path(raw_plan_path, step="coach training-plan")

        print(plan_text)
        print(f"[Saved] {plan_path}")
        plan_txt = Path(plan_path).parent / f"{Path(plan_path).stem}.txt"
        if plan_txt.exists():
            print(f"[Saved] {plan_txt}")

        print("==> Step 4/6: eval original plan")
        original_report_path = prompting_dir / "eval_report.original.json"
        _evaluate_training_plan(
            args,
            input_path=plan_path,
            report_path=str(original_report_path),
            output_path=None,
            soft_eval_enabled=True,
            soft_eval_runs=max(1, int(getattr(args, "soft_eval_runs", 2) or 2)),
            exit_on_violations=False,
        )
        canonical_report = prompting_dir / "eval_report.json"
        if canonical_report.exists():
            print(f"[Saved] {canonical_report}")

        print("==> Step 5/6: revise-plan")
        revise_cfg = RevisePlanConfig(
            model=args.model or base_cfg.model,
            reasoning_effort=args.reasoning_effort or base_cfg.reasoning_effort,
            verbosity=args.verbosity or base_cfg.verbosity,
            temperature=args.temperature,
            primary_goal=primary_goal,
            lifestyle_notes=lifestyle_notes,
        )
        revised_text, raw_revised_path = run_revise_plan(
            cfg=revise_cfg,
            input_plan_path=plan_path,
            eval_report_path=str(original_report_path),
            output_path=None,
            rollups_path=getattr(args, "rollups", None),
            auto_reeval=True,
        )
        revised_path = _require_output_path(raw_revised_path, step="revise-plan")

        print(revised_text)
        print(f"[Saved] {revised_path}")
        revised_txt = Path(revised_path).parent / f"{Path(revised_path).stem}.txt"
        if revised_txt.exists():
            print(f"[Saved] {revised_txt}")
        comparison_p = Path(revised_path).parent / f"{Path(revised_path).stem}-comparison.json"
        if comparison_p.exists():
            print(f"[Saved] {comparison_p}")
        selected_json = Path(revised_path).parent / "selected-plan.json"
        selected_txt = Path(revised_path).parent / "selected-plan.txt"
        if selected_json.exists():
            print(f"[Saved] {selected_json}")
        if selected_txt.exists():
            print(f"[Saved] {selected_txt}")
        reeval_p = Path(revised_path).parent / f"{Path(revised_path).stem}-reeval.json"
        if reeval_p.exists():
            print(f"[Saved] {reeval_p}")

        print("==> Step 6/6: eval revised plan")
        revised_report_path = prompting_dir / "eval_report.revised.json"
        _evaluate_training_plan(
            args,
            input_path=revised_path,
            report_path=str(revised_report_path),
            output_path=None,
            soft_eval_enabled=True,
            soft_eval_runs=max(1, int(getattr(args, "soft_eval_runs", 2) or 2)),
            exit_on_violations=False,
        )

        print("Workflow complete.")
        print(f"Original eval: {original_report_path}")
        print(f"Revised eval:  {revised_report_path}")
        print(f"Revised plan:  {revised_path}")

    _run(_inner)
