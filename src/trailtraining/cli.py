# src/trailtraining/cli.py
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

def configure_logging(level: str) -> None:
    """
    Central logging setup for the CLI.

    Priority:
      1) CLI arg (--log-level)
      2) env TRAILTRAINING_LOG_LEVEL
      3) default INFO
    """
    raw = (level or os.getenv("TRAILTRAINING_LOG_LEVEL") or "INFO").upper().strip()
    if raw not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raw = "INFO"

    logging.basicConfig(
        level=getattr(logging, raw),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _run(func):
    try:
        func()
    except SystemExit:
        raise
    except Exception:
        logging.getLogger(__name__).exception("Unhandled error")
        sys.exit(1)


def _load_env_file(path: Path) -> None:
    """
    Load KEY=VALUE lines from a .env-style file into os.environ.
    - Ignores blank lines and comments (# ...)
    - Does NOT override already-set environment variables
    - Supports simple quoted values
    """
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
    """
    Applies a profile by:
      1) Setting TRAILTRAINING_PROFILE
      2) Loading ~/.trailtraining/profiles/<profile>.env (unless vars already set)
      3) Setting a default per-profile TRAILTRAINING_BASE_DIR (unless already set)
    """
    profile = (profile or "default").strip() or "default"
    os.environ["TRAILTRAINING_PROFILE"] = profile

    # Load per-profile secrets/config if present
    env_path = Path.home() / ".trailtraining" / "profiles" / f"{profile}.env"
    _load_env_file(env_path)

    # If still not set, isolate data by profile
    os.environ.setdefault("TRAILTRAINING_BASE_DIR", str(Path.home() / "trailtraining-data" / profile))

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


def _detect_provider_for_doctor() -> str:
    # Prefer explicit env var, otherwise auto.
    env_v = ((os.getenv("TRAILTRAINING_WELLNESS_PROVIDER") or "").strip() or (os.getenv("WELLNESS_PROVIDER") or "").strip())
    v = env_v.lower() if env_v else "auto"
    if v in {"garmin", "intervals"}:
        return v

    # auto-detect
    from trailtraining import config

    if (config.INTERVALS_API_KEY or "").strip():
        return "intervals"
    if (config.GARMIN_EMAIL or "").strip() and (config.GARMIN_PASSWORD or "").strip():
        return "garmin"
    return "intervals"


def cmd_doctor(_args):
    from trailtraining import config
    from trailtraining.data.strava import default_token_path

    def ok(label: str, msg: str = "") -> None:
        print(f"✅ {label}" + (f" — {msg}" if msg else ""))

    def warn(label: str, msg: str = "") -> None:
        print(f"⚠️  {label}" + (f" — {msg}" if msg else ""))

    def bad(label: str, msg: str = "") -> None:
        print(f"❌ {label}" + (f" — {msg}" if msg else ""))

    print("TrailTraining doctor\n")

    config.ensure_directories()
    profile = os.getenv("TRAILTRAINING_PROFILE", "default")
    base_dir = os.getenv("TRAILTRAINING_BASE_DIR", "")
    ok("Profile", profile)
    ok("Base dir", base_dir or "(not set)")

    issues = 0

    # ---- Strava ----
    if config.STRAVA_ID and config.STRAVA_ID != 0:
        ok("STRAVA_CLIENT_ID set")
    else:
        bad("STRAVA_CLIENT_ID missing", "Set STRAVA_CLIENT_ID in your profile env.")
        issues += 1

    if (config.STRAVA_SECRET or "").strip():
        ok("STRAVA_CLIENT_SECRET set")
    else:
        bad("STRAVA_CLIENT_SECRET missing", "Set STRAVA_CLIENT_SECRET in your profile env.")
        issues += 1

    if (config.STRAVA_REDIRECT_URI or "").strip():
        ok("STRAVA_REDIRECT_URI set", config.STRAVA_REDIRECT_URI)
    else:
        warn("STRAVA_REDIRECT_URI missing", "Default will be used, but set it explicitly to match your Strava app.")

    token_path = default_token_path()
    if token_path.exists():
        ok("Strava token", str(token_path))
    else:
        warn("Strava token not found", f"Run: trailtraining --profile {profile} auth-strava")

    # ---- Wellness provider ----
    provider = _detect_provider_for_doctor()
    ok("Wellness provider", provider)

    if provider == "intervals":
        if (config.INTERVALS_API_KEY or "").strip():
            ok("INTERVALS_API_KEY set")
        else:
            bad("INTERVALS_API_KEY missing", "Set INTERVALS_API_KEY (or choose Garmin).")
            issues += 1

        athlete_id = (config.INTERVALS_ATHLETE_ID or "").strip()
        if athlete_id:
            ok("INTERVALS_ATHLETE_ID", athlete_id)
        else:
            warn("INTERVALS_ATHLETE_ID not set", "Default '0' may still work (current athlete).")

    if provider == "garmin":
        if (config.GARMIN_EMAIL or "").strip():
            ok("GARMIN_EMAIL set")
        else:
            bad("GARMIN_EMAIL missing")
            issues += 1

        if (config.GARMIN_PASSWORD or "").strip():
            ok("GARMIN_PASSWORD set")
        else:
            bad("GARMIN_PASSWORD missing")
            issues += 1

        script = os.environ.get("GARMINGDB_CLI") or shutil.which("garmindb_cli") or shutil.which("garmindb_cli.py")
        if script:
            ok("GarminDb CLI found", script)
        else:
            bad("GarminDb CLI missing", "Install GarminDb and ensure garmindb_cli is on PATH (or set GARMINGDB_CLI).")
            issues += 1

    # ---- Optional OpenAI ----
    if os.getenv("OPENAI_API_KEY") or os.getenv("TRAILTRAINING_OPENAI_API_KEY"):
        ok("OpenAI API key set (coach enabled)")
    else:
        warn("OpenAI API key not set", "Coach won’t run until you set OPENAI_API_KEY (recommended).")

    print("\nSummary:")
    if issues:
        bad("Doctor found issues", f"{issues} blocking issue(s).")
        raise SystemExit(1)

    ok("Doctor OK", "No blocking issues found.")
    raise SystemExit(0)


def cmd_coach(args):
    from trailtraining.llm.coach import CoachConfig, run_coach_brief

    cfg = CoachConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        days=args.days,
        max_chars=args.max_chars,
        temperature=args.temperature,
        style=args.style,
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
    """
    Evaluate a coach training-plan JSON output against constraints + quality scoring.
    """
    from trailtraining.llm.constraints import ConstraintConfig
    from trailtraining.llm.eval import evaluate_training_plan_quality_file
    from trailtraining.util.state import save_json

    cfg = ConstraintConfig(
        max_ramp_pct=float(args.max_ramp_pct),
        max_consecutive_hard=int(args.max_consecutive_hard),
    )

    report, _obj = evaluate_training_plan_quality_file(
        args.input,
        rollups_path=args.rollups,
        cfg=cfg,
    )
    violations = report.get("violations", [])

    # Save outputs (keep --output behavior = violations JSON)
    if args.output:
        outp = Path(args.output).expanduser().resolve()
        save_json(outp, violations, compact=False)
        print(f"[Saved] {outp}")

    # New: save full report
    if getattr(args, "report", None):
        outp = Path(args.report).expanduser().resolve()
        save_json(outp, report, compact=False)
        print(f"[Saved] {outp}")

    score = report.get("score", 0)
    grade = report.get("grade", "?")
    print(f"Score: {score}/100 ({grade})")

    subs = report.get("subscores", {}) or {}
    if subs:
        # stable ordering for readability
        parts = [f"{k}={subs[k]}" for k in sorted(subs.keys())]
        print("Subscores:", ", ".join(parts))

    if not violations:
        print("✅ eval-coach: no violations")
        raise SystemExit(0)

    print("⚠️  eval-coach violations:")
    for v in violations:
        sev = v.get("severity", "unknown")
        code = v.get("code", "UNKNOWN")
        msg = v.get("message", "")
        print(f"- [{sev}] {code}: {msg}")

    # Fail on any high severity (same behavior as before)
    if any(v.get("severity") == "high" for v in violations):
        raise SystemExit(1)
    raise SystemExit(0)

def cmd_forecast(args):
    from trailtraining.forecast.forecast import run_forecasts
    r = run_forecasts(input_dir=args.input, output_path=args.output)
    print(f"[Saved] {r['saved']}")
    print(r["result"])

def main(argv=None):
    parser = argparse.ArgumentParser(prog="trailtraining", description="TrailTraining CLI")

    # Global profile support
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

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check configuration + dependencies").set_defaults(func=cmd_doctor)

    auth_p = sub.add_parser("auth-strava", help="Run Strava auth flow (opens local server)")
    auth_p.add_argument(
        "--force",
        action="store_true",
        help="Force reauthorization even if a token exists (useful if you authorized the wrong account).",
    )
    auth_p.set_defaults(func=cmd_auth_strava)
    sub.add_parser("fetch-strava", help="Fetch activities from Strava").set_defaults(func=cmd_fetch_strava)
    sub.add_parser("fetch-garmin", help="Fetch/process data from Garmin").set_defaults(func=cmd_fetch_garmin)
    sub.add_parser("combine", help="Combine Garmin + Strava JSONs").set_defaults(func=cmd_combine)

    run_all_p = sub.add_parser("run-all", help="Run full pipeline (auto: Garmin OR Intervals → Strava → Combine)")
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
    run_all_p.add_argument("--clean-prompting", action="store_true", help="Delete files in prompting/ before running.")
    run_all_p.add_argument(
        "--wellness-provider",
        default=None,
        choices=["auto", "garmin", "intervals"],
        help="Override wellness provider (default: env TRAILTRAINING_WELLNESS_PROVIDER/WELLNESS_PROVIDER or auto).",
    )
    run_all_p.set_defaults(func=cmd_run_all)

    # coach
    coach_p = sub.add_parser("coach", help="LLM coach analysis on combined_summary.json + formatted_personal_data.json")
    coach_p.add_argument("--prompt", default="training-plan", choices=["training-plan", "recovery-status", "meal-plan"])
    coach_p.add_argument("--model", default=os.getenv("TRAILTRAINING_LLM_MODEL", "gpt-5.2"))
    coach_p.add_argument(
        "--reasoning-effort",
        default=os.getenv("TRAILTRAINING_REASONING_EFFORT", "medium"),
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    coach_p.add_argument("--verbosity", default=os.getenv("TRAILTRAINING_VERBOSITY", "medium"), choices=["low", "medium", "high"])
    coach_p.add_argument("--temperature", type=float, default=None, help="Only used if --reasoning-effort none (API restriction).")
    coach_p.add_argument("--days", type=int, default=int(os.getenv("TRAILTRAINING_COACH_DAYS", "60")))
    coach_p.add_argument("--max-chars", type=int, default=int(os.getenv("TRAILTRAINING_COACH_MAX_CHARS", "200000")))
    coach_p.add_argument(
        "--output",
        default=None,
        help="Output file. Default: training-plan -> .json, others -> .md in <prompting_dir>/coach_brief_<prompt>.*",
    )
    coach_p.add_argument("--input", default=None, help="Directory containing the two JSON files. Default: prompting directory")
    coach_p.add_argument("--personal", default=None, help="Explicit path to formatted_personal_data.json (overrides --input)")
    coach_p.add_argument("--summary", default=None, help="Explicit path to combined_summary.json (overrides --input)")

    # style preset
    coach_p.add_argument(
        "--style",
        default=os.getenv("TRAILTRAINING_COACH_STYLE", "trailrunning"),
        choices=["trailrunning", "triathlon"],
        help="Prompt preset: changes system instructions; training-plan prompt is sport-specific.",
    )
    coach_p.set_defaults(func=cmd_coach)

    # eval-coach
    eval_p = sub.add_parser("eval-coach", help="Evaluate coach training-plan JSON output against constraints")
    eval_p.add_argument("--input", required=True, help="Path to coach_brief_training-plan.json (or any training-plan JSON)")
    eval_p.add_argument("--rollups", default=None, help="Optional path to combined_rollups.json (default: same dir as --input)")
    eval_p.add_argument("--max-ramp-pct", type=float, default=float(os.getenv("TRAILTRAINING_MAX_RAMP_PCT", "10")))
    eval_p.add_argument("--max-consecutive-hard", type=int, default=int(os.getenv("TRAILTRAINING_MAX_CONSEC_HARD", "2")))
    eval_p.add_argument("--output", default=None, help="Optional path to write violations JSON")
    eval_p.add_argument("--report", default=None, help="Optional path to write full scoring report JSON")
    eval_p.set_defaults(func=cmd_eval_coach)

    # intervals
    intervals_p = sub.add_parser("fetch-intervals", help="Fetch sleep + resting HR from Intervals.icu")
    intervals_p.add_argument("--script", default=None, help="(deprecated) Old node script path. Ignored; Python Intervals fetch is used.")
    intervals_p.add_argument("--oldest", default=None, help="YYYY-MM-DD (default: lookback window)")
    intervals_p.add_argument("--newest", default=None, help="YYYY-MM-DD (default: today)")
    intervals_p.set_defaults(func=cmd_fetch_intervals)

    run_all_int_p = sub.add_parser("run-all-intervals", help="Run full pipeline (Intervals → Strava → Combine)")
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
    run_all_int_p.add_argument("--clean-prompting", action="store_true", help="Delete files in prompting/ before running.")
    run_all_int_p.set_defaults(func=cmd_run_all_intervals)

    forecast_p = sub.add_parser("forecast", help="Readiness forecast + overreach risk (from combined_summary.json)")
    forecast_p.add_argument("--input", default=None,
                            help="Directory containing combined_summary.json (default: prompting dir)")
    forecast_p.add_argument("--output", default=None,
                            help="Output JSON path (default: <input>/readiness_and_risk_forecast.json)")
    forecast_p.set_defaults(func=cmd_forecast)

    args = parser.parse_args(argv)

    # Apply profile BEFORE running command imports
    apply_profile(args.profile)

    # Configure logging AFTER profile loads (so env-based defaults are present)
    configure_logging(args.log_level)

    args.func(args)


if __name__ == "__main__":
    main()