# src/trailtraining/cli.py
import argparse
import os
import sys
from pathlib import Path


def _run(func):
    try:
        func()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
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


def cmd_auth_strava(_args):
    from trailtraining.pipelines import strava
    _run(strava.main)


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
    _run(lambda: run_all.main(
        clean=getattr(args, "clean", False),
        clean_processing=getattr(args, "clean_processing", False),
        clean_prompting=getattr(args, "clean_prompting", False),
    ))


def cmd_fetch_intervals(args):
    from trailtraining.pipelines import intervals
    _run(lambda: intervals.main(
        oldest=getattr(args, "oldest", None),
        newest=getattr(args, "newest", None),
    ))


def cmd_run_all_intervals(args):
    from trailtraining.pipelines import run_all_intervals
    _run(lambda: run_all_intervals.main(
        clean=getattr(args, "clean", False),
        clean_processing=getattr(args, "clean_processing", False),
        clean_prompting=getattr(args, "clean_prompting", False),
    ))


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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trailtraining", description="TrailTraining CLI")

    # Global profile support
    parser.add_argument(
        "--profile",
        default=os.getenv("TRAILTRAINING_PROFILE", "default"),
        help="Profile name (loads ~/.trailtraining/profiles/<profile>.env and isolates data dirs).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("auth-strava", help="Run Strava auth flow (opens local server)").set_defaults(func=cmd_auth_strava)
    sub.add_parser("fetch-strava", help="Fetch activities from Strava").set_defaults(func=cmd_fetch_strava)
    sub.add_parser("fetch-garmin", help="Fetch/process data from Garmin").set_defaults(func=cmd_fetch_garmin)
    sub.add_parser("combine", help="Combine Garmin + Strava JSONs").set_defaults(func=cmd_combine)

    run_all_p = sub.add_parser("run-all", help="Run full pipeline (Garmin → Strava → Combine)")
    run_all_p.add_argument("--clean", action="store_true",
                           help="Delete files in BOTH processing/ and prompting/ before running (disables incremental Strava).")
    run_all_p.add_argument("--clean-processing", action="store_true",
                           help="Delete files in processing/ before running (disables incremental Strava).")
    run_all_p.add_argument("--clean-prompting", action="store_true",
                           help="Delete files in prompting/ before running.")
    run_all_p.set_defaults(func=cmd_run_all)

    # coach
    coach_p = sub.add_parser("coach", help="LLM coach analysis on combined_summary.json + formatted_personal_data.json")
    coach_p.add_argument("--prompt", default="training-plan", choices=["training-plan", "recovery-status", "meal-plan"])
    coach_p.add_argument("--model", default=os.getenv("TRAILTRAINING_LLM_MODEL", "gpt-5.2"))
    coach_p.add_argument("--reasoning-effort", default=os.getenv("TRAILTRAINING_REASONING_EFFORT", "medium"),
                         choices=["none", "low", "medium", "high", "xhigh"])
    coach_p.add_argument("--verbosity", default=os.getenv("TRAILTRAINING_VERBOSITY", "medium"),
                         choices=["low", "medium", "high"])
    coach_p.add_argument("--temperature", type=float, default=None,
                         help="Only used if --reasoning-effort none (API restriction).")
    coach_p.add_argument("--days", type=int, default=int(os.getenv("TRAILTRAINING_COACH_DAYS", "60")))
    coach_p.add_argument("--max-chars", type=int, default=int(os.getenv("TRAILTRAINING_COACH_MAX_CHARS", "200000")))
    coach_p.add_argument("--output", default=None,
                         help="Output markdown file. Default: <prompting_dir>/coach_brief_<prompt>.md")
    coach_p.add_argument("--input", default=None,
                         help="Directory containing the two JSON files. Default: prompting directory")
    coach_p.add_argument("--personal", default=None,
                         help="Explicit path to formatted_personal_data.json (overrides --input)")
    coach_p.add_argument("--summary", default=None,
                         help="Explicit path to combined_summary.json (overrides --input)")

    # NEW: style preset (affects system instructions + training-plan prompt only)
    coach_p.add_argument(
        "--style",
        default=os.getenv("TRAILTRAINING_COACH_STYLE", "trailrunning"),
        choices=["trailrunning", "triathlon"],
        help="Prompt preset: changes system instructions; training-plan prompt is sport-specific.",
    )

    coach_p.set_defaults(func=cmd_coach)

    # intervals
    intervals_p = sub.add_parser("fetch-intervals", help="Fetch sleep + resting HR from Intervals.icu")
    intervals_p.add_argument("--script", default=None,
                             help="(deprecated) Old node script path. Ignored; Python Intervals fetch is used.")
    intervals_p.add_argument("--oldest", default=None, help="YYYY-MM-DD (default: lookback window)")
    intervals_p.add_argument("--newest", default=None, help="YYYY-MM-DD (default: today)")
    intervals_p.set_defaults(func=cmd_fetch_intervals)

    run_all_int_p = sub.add_parser("run-all-intervals", help="Run full pipeline (Intervals → Strava → Combine)")
    run_all_int_p.add_argument("--clean", action="store_true",
                               help="Delete files in BOTH processing/ and prompting/ before running (disables incremental Strava).")
    run_all_int_p.add_argument("--clean-processing", action="store_true",
                               help="Delete files in processing/ before running (disables incremental Strava).")
    run_all_int_p.add_argument("--clean-prompting", action="store_true",
                               help="Delete files in prompting/ before running.")
    run_all_int_p.set_defaults(func=cmd_run_all_intervals)

    args = parser.parse_args(argv)

    # Apply profile BEFORE running command imports
    apply_profile(args.profile)

    args.func(args)


if __name__ == "__main__":
    main()