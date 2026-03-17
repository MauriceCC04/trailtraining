from __future__ import annotations

import argparse
import os
from pathlib import Path

from trailtraining.commands.common import _run
from trailtraining.llm.rubrics import default_primary_goal_for_style


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

        cfg = CoachConfig(
            model=args.model or base_cfg.model,
            reasoning_effort=args.reasoning_effort or base_cfg.reasoning_effort,
            verbosity=args.verbosity or base_cfg.verbosity,
            days=args.days if args.days is not None else base_cfg.days,
            max_chars=args.max_chars if args.max_chars is not None else base_cfg.max_chars,
            temperature=args.temperature,
            style=style,
            primary_goal=primary_goal,
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
    from trailtraining import config
    from trailtraining.llm.constraints import constraint_config_from_env
    from trailtraining.llm.eval import evaluate_training_plan_quality_file
    from trailtraining.llm.soft_eval import SoftEvalConfig
    from trailtraining.util.state import save_json

    def _inner() -> None:
        input_path = args.input or str(
            Path(config.PROMPTING_DIRECTORY) / "coach_brief_training-plan.json"
        )
        report_path = args.report or str(Path(config.PROMPTING_DIRECTORY) / "eval_report.json")

        cfg = constraint_config_from_env(
            max_ramp_pct=float(args.max_ramp_pct),
            max_consecutive_hard=int(args.max_consecutive_hard),
        )

        soft_cfg = None
        if getattr(args, "soft_eval", False):
            default_soft = SoftEvalConfig.from_env()
            soft_cfg = SoftEvalConfig(
                enabled=True,
                model=args.soft_eval_model or default_soft.model,
                reasoning_effort=(
                    getattr(args, "soft_eval_reasoning_effort", None)
                    or default_soft.reasoning_effort
                ),
                verbosity=(getattr(args, "soft_eval_verbosity", None) or default_soft.verbosity),
                primary_goal=args.goal or default_soft.primary_goal,
            )

        report, _obj = evaluate_training_plan_quality_file(
            input_path,
            rollups_path=args.rollups,
            cfg=cfg,
            soft_eval_cfg=soft_cfg,
            primary_goal=args.goal,
        )
        violations = report.get("violations", [])

        if args.output:
            outp = Path(args.output).expanduser().resolve()
            save_json(outp, violations, compact=False)
            print(f"[Saved] {outp}")

        outp = Path(report_path).expanduser().resolve()
        save_json(outp, report, compact=False)
        print(f"[Saved] {outp}")

        score = report.get("score", 0)
        grade = report.get("grade", "?")
        print(f"Deterministic score: {score}/100 ({grade})")

        subs = report.get("subscores", {}) or {}
        if subs:
            parts = [f"{k}={subs[k]}" for k in sorted(subs.keys())]
            print("Deterministic subscores:", ", ".join(parts))

        soft = report.get("soft_assessment") or {}
        if isinstance(soft, dict) and soft:
            soft_score = soft.get("overall_score", 0)
            soft_grade = soft.get("grade", "?")
            print(f"Soft quality score: {soft_score}/100 ({soft_grade})")

            rubric_scores = soft.get("rubric_scores", {}) or {}
            if rubric_scores:
                parts = []
                for key in sorted(rubric_scores.keys()):
                    item = rubric_scores.get(key) or {}
                    parts.append(f"{key}={item.get('score', 0)}")
                print("Rubric scores:", ", ".join(parts))

        if not violations:
            print("✅ eval-coach: no violations")
            raise SystemExit(0)

        print("⚠️  eval-coach violations:")
        for violation in violations:
            sev = violation.get("severity", "unknown")
            code = violation.get("code", "UNKNOWN")
            msg = violation.get("message", "")
            print(f"- [{sev}] {code}: {msg}")

        if any(v.get("severity") == "high" for v in violations):
            raise SystemExit(1)
        raise SystemExit(0)

    _run(_inner)


def cmd_revise_plan(args: argparse.Namespace) -> None:
    from trailtraining import config
    from trailtraining.llm.coach import CoachConfig
    from trailtraining.llm.revise import RevisePlanConfig, run_revise_plan

    def _inner() -> None:
        base_cfg = CoachConfig.from_env()
        revise_cfg = RevisePlanConfig(
            model=args.model or base_cfg.model,
            reasoning_effort=args.reasoning_effort or base_cfg.reasoning_effort,
            verbosity=args.verbosity or base_cfg.verbosity,
            temperature=args.temperature,
            primary_goal=args.goal or base_cfg.primary_goal,
        )

        input_path = args.input or str(
            Path(config.PROMPTING_DIRECTORY) / "coach_brief_training-plan.json"
        )
        report_path = args.report or str(Path(config.PROMPTING_DIRECTORY) / "eval_report.json")

        text, out_path = run_revise_plan(
            cfg=revise_cfg,
            input_plan_path=input_path,
            eval_report_path=report_path,
            output_path=args.output,
            rollups_path=args.rollups,
        )

        print(text)
        if out_path:
            print(f"\n[Saved] {out_path}")
            p = Path(out_path)
            txt_p = p.parent / f"{p.stem}.txt"
            if txt_p.exists():
                print(f"[Saved] {txt_p}")

    _run(_inner)
