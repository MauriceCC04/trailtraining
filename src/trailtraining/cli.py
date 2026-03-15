# src/trailtraining/cli.py
import argparse
import logging
import os
import sys
from pathlib import Path

from trailtraining.llm.rubrics import default_primary_goal_for_style
from trailtraining.util.logging_config import configure_logging


def _run(func):
    try:
        func()
    except SystemExit:
        raise
    except Exception:
        logging.getLogger(__name__).exception("Unhandled error")
        sys.exit(1)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def apply_profile(profile: str) -> str:
    profile = (profile or "default").strip() or "default"
    os.environ["TRAILTRAINING_PROFILE"] = profile

    env_path = Path.home() / ".trailtraining" / "profiles" / f"{profile}.env"
    _load_env_file(env_path)

    os.environ.setdefault(
        "TRAILTRAINING_BASE_DIR", str(Path.home() / "trailtraining-data" / profile)
    )

    return profile


def cmd_auth_strava(args):
    from trailtraining.pipelines import strava

    _run(lambda: strava.auth_main(force=getattr(args, "force", False)))


def cmd_fetch_strava(_args):
    from trailtraining.pipelines import strava

    _run(strava.main)


def cmd_fetch_garmin(_args):
    from trailtraining.pipelines import garmin

    _run(garmin.main)


def cmd_combine(_args):
    from trailtraining.data import combine

    _run(combine.main)


def cmd_run_all(args):
    from trailtraining.pipelines import run_all

    _run(
        lambda: run_all.main(
            clean=getattr(args, "clean", False),
            clean_processing=getattr(args, "clean_processing", False),
            clean_prompting=getattr(args, "clean_prompting", False),
            wellness_provider=getattr(args, "wellness_provider", None),
        )
    )


def cmd_fetch_intervals(args):
    from trailtraining.pipelines import intervals

    _run(
        lambda: intervals.main(
            oldest=getattr(args, "oldest", None),
            newest=getattr(args, "newest", None),
        )
    )


def cmd_run_all_intervals(args):
    from trailtraining.pipelines import run_all_intervals

    _run(
        lambda: run_all_intervals.main(
            clean=getattr(args, "clean", False),
            clean_processing=getattr(args, "clean_processing", False),
            clean_prompting=getattr(args, "clean_prompting", False),
        )
    )

    from trailtraining import config

    if (config.INTERVALS_API_KEY or "").strip():
        return "intervals"
    if (config.GARMIN_EMAIL or "").strip() and (config.GARMIN_PASSWORD or "").strip():
        return "garmin"
    return "intervals"


def cmd_doctor(_args):
    from trailtraining import doctor

    _run(doctor.main)


def cmd_forecast(args):
    from trailtraining.forecast.forecast import run_forecasts

    r = run_forecasts(input_dir=args.input, output_path=args.output)
    print(f"[Saved] {r['saved']}")
    print(r["result"])


def cmd_coach(args):
    from trailtraining.llm.coach import CoachConfig, run_coach_brief

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


def cmd_eval_coach(args):
    from trailtraining import config
    from trailtraining.llm.constraints import constraint_config_from_env
    from trailtraining.llm.eval import evaluate_training_plan_quality_file
    from trailtraining.llm.soft_eval import SoftEvalConfig
    from trailtraining.util.state import save_json

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
                getattr(args, "soft_eval_reasoning_effort", None) or default_soft.reasoning_effort
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
    for v in violations:
        sev = v.get("severity", "unknown")
        code = v.get("code", "UNKNOWN")
        msg = v.get("message", "")
        print(f"- [{sev}] {code}: {msg}")

    if any(v.get("severity") == "high" for v in violations):
        raise SystemExit(1)
    raise SystemExit(0)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trailtraining", description="TrailTraining CLI")

    parser.add_argument(
        "--profile",
        default=os.getenv("TRAILTRAINING_PROFILE", "default"),
        help="Profile name (loads ~/.trailtraining/profiles/<profile>.env and isolates data dirs).",
    )
    parser.add_argument(
        "--log-level",
        default=((os.getenv("TRAILTRAINING_LOG_LEVEL") or "INFO").upper()),
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (or set TRAILTRAINING_LOG_LEVEL).",
    )
    configure_logging(os.getenv("TRAILTRAINING_LOG_LEVEL", "INFO"))

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check configuration + dependencies").set_defaults(
        func=cmd_doctor
    )

    auth_p = sub.add_parser("auth-strava", help="Run Strava auth flow (opens local server)")
    auth_p.add_argument(
        "--force",
        action="store_true",
        help="Force reauthorization even if a token exists (useful if you authorized the wrong account).",
    )
    auth_p.set_defaults(func=cmd_auth_strava)
    sub.add_parser("fetch-strava", help="Fetch activities from Strava").set_defaults(
        func=cmd_fetch_strava
    )
    sub.add_parser("fetch-garmin", help="Fetch/process data from Garmin").set_defaults(
        func=cmd_fetch_garmin
    )
    sub.add_parser("combine", help="Combine Garmin + Strava JSONs").set_defaults(func=cmd_combine)

    run_all_p = sub.add_parser(
        "run-all", help="Run full pipeline (auto: Garmin OR Intervals → Strava → Combine)"
    )
    run_all_p.add_argument(
        "--clean",
        action="store_true",
        help="Delete files in BOTH processing/ and prompting/ before running (disables incremental Strava).",
    )
    run_all_p.add_argument(
        "--clean-processing",
        action="store_true",
        help="Delete files in processing/ before running (disables incremental Strava).",
    )
    run_all_p.add_argument(
        "--clean-prompting", action="store_true", help="Delete files in prompting/ before running."
    )
    run_all_p.add_argument(
        "--wellness-provider",
        default=None,
        choices=["auto", "garmin", "intervals"],
        help="Override wellness provider (default: env TRAILTRAINING_WELLNESS_PROVIDER/WELLNESS_PROVIDER or auto).",
    )
    run_all_p.set_defaults(func=cmd_run_all)

    coach_p = sub.add_parser(
        "coach", help="LLM coach analysis on combined_summary.json + formatted_personal_data.json"
    )
    coach_p.add_argument(
        "--prompt",
        default="training-plan",
        choices=["training-plan", "recovery-status", "meal-plan"],
    )
    coach_p.add_argument("--model", default=None)
    coach_p.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    coach_p.add_argument(
        "--verbosity",
        default=None,
        choices=["low", "medium", "high"],
    )
    coach_p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Only used if --reasoning-effort none (API restriction).",
    )
    coach_p.add_argument("--days", type=int, default=None)
    coach_p.add_argument("--max-chars", type=int, default=None)
    coach_p.add_argument(
        "--goal",
        default=None,
        help="Primary athlete goal used by generation and soft evaluation.",
    )
    coach_p.add_argument(
        "--output",
        default=None,
        help="Output file. Default: training-plan -> .json, others -> .md in <prompting_dir>/coach_brief_<prompt>.*",
    )
    coach_p.add_argument(
        "--input",
        default=None,
        help="Directory containing the two JSON files. Default: prompting directory",
    )
    coach_p.add_argument(
        "--personal",
        default=None,
        help="Explicit path to formatted_personal_data.json (overrides --input)",
    )
    coach_p.add_argument(
        "--summary",
        default=None,
        help="Explicit path to combined_summary.json (overrides --input)",
    )
    coach_p.add_argument(
        "--style",
        default=None,
        choices=["trailrunning", "triathlon"],
        help="Prompt preset: changes system instructions; training-plan prompt is sport-specific.",
    )
    coach_p.set_defaults(func=cmd_coach)
    # eval-coach
    eval_p = sub.add_parser(
        "eval-coach", help="Evaluate coach training-plan JSON output against constraints"
    )
    eval_p.add_argument(
        "--input",
        default=None,
        help="Path to coach_brief_training-plan.json (default: <prompting_dir>/coach_brief_training-plan.json)",
    )
    eval_p.add_argument(
        "--rollups",
        default=None,
        help="Optional path to combined_rollups.json (default: same dir as resolved input)",
    )
    eval_p.add_argument(
        "--max-ramp-pct", type=float, default=float(os.getenv("TRAILTRAINING_MAX_RAMP_PCT", "10"))
    )
    eval_p.add_argument(
        "--max-consecutive-hard",
        type=int,
        default=int(os.getenv("TRAILTRAINING_MAX_CONSEC_HARD", "2")),
    )
    eval_p.add_argument("--output", default=None, help="Optional path to write violations JSON")
    eval_p.add_argument(
        "--report", default=None, help="Optional path to write full scoring report JSON"
    )
    eval_p.add_argument("--soft-eval", action="store_true", help="Run rubric-based soft evaluation")
    eval_p.add_argument(
        "--soft-eval-model",
        default=None,
        help="OpenRouter model for soft evaluation",
    )
    eval_p.add_argument(
        "--goal",
        default=None,
        help="Primary athlete goal used for soft evaluation",
    )
    eval_p.add_argument(
        "--soft-eval-reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    eval_p.add_argument(
        "--soft-eval-verbosity",
        default=None,
        choices=["low", "medium", "high"],
    )
    eval_p.set_defaults(func=cmd_eval_coach)
    intervals_p = sub.add_parser(
        "fetch-intervals", help="Fetch sleep + resting HR from Intervals.icu"
    )
    intervals_p.add_argument("--oldest", default=None, help="YYYY-MM-DD (default: lookback window)")
    intervals_p.add_argument("--newest", default=None, help="YYYY-MM-DD (default: today)")
    intervals_p.set_defaults(func=cmd_fetch_intervals)

    run_all_int_p = sub.add_parser(
        "run-all-intervals", help="Run full pipeline (Intervals → Strava → Combine)"
    )
    run_all_int_p.add_argument(
        "--clean",
        action="store_true",
        help="Delete files in BOTH processing/ and prompting/ before running (disables incremental Strava).",
    )
    run_all_int_p.add_argument(
        "--clean-processing",
        action="store_true",
        help="Delete files in processing/ before running (disables incremental Strava).",
    )
    run_all_int_p.add_argument(
        "--clean-prompting", action="store_true", help="Delete files in prompting/ before running."
    )
    run_all_int_p.set_defaults(func=cmd_run_all_intervals)

    forecast_p = sub.add_parser(
        "forecast", help="Readiness forecast + overreach risk (from combined_summary.json)"
    )
    forecast_p.add_argument(
        "--input",
        default=None,
        help="Directory containing combined_summary.json (default: prompting dir)",
    )
    forecast_p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <input>/readiness_and_risk_forecast.json)",
    )
    forecast_p.set_defaults(func=cmd_forecast)

    args = parser.parse_args(argv)

    apply_profile(args.profile)
    configure_logging(args.log_level)

    args.func(args)


if __name__ == "__main__":
    main()
