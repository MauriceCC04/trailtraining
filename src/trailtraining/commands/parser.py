from __future__ import annotations

import argparse
import os

from trailtraining.commands.llm_commands import (
    cmd_coach,
    cmd_eval_coach,
    cmd_revise_plan,
)
from trailtraining.commands.pipeline_commands import (
    cmd_auth_strava,
    cmd_combine,
    cmd_doctor,
    cmd_fetch_garmin,
    cmd_fetch_intervals,
    cmd_fetch_strava,
    cmd_forecast,
    cmd_run_all,
    cmd_run_all_intervals,
)


def build_parser() -> argparse.ArgumentParser:
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
        "run-all",
        help="Run full pipeline (auto: Garmin OR Intervals → Strava → Combine)",
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
        "--clean-prompting",
        action="store_true",
        help="Delete files in prompting/ before running.",
    )
    run_all_p.add_argument(
        "--wellness-provider",
        default=None,
        choices=["auto", "garmin", "intervals"],
        help="Override wellness provider (default: env TRAILTRAINING_WELLNESS_PROVIDER/WELLNESS_PROVIDER or auto).",
    )
    run_all_p.set_defaults(func=cmd_run_all)

    coach_p = sub.add_parser(
        "coach",
        help="LLM coach analysis on combined_summary.json + formatted_personal_data.json",
    )
    coach_p.add_argument(
        "--prompt",
        default="training-plan",
        choices=["training-plan", "recovery-status", "meal-plan", "session-review"],
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

    eval_p = sub.add_parser(
        "eval-coach",
        help="Evaluate coach training-plan JSON output against constraints",
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
        "--max-ramp-pct",
        type=float,
        default=float(os.getenv("TRAILTRAINING_MAX_RAMP_PCT", "10")),
    )
    eval_p.add_argument(
        "--max-consecutive-hard",
        type=int,
        default=int(os.getenv("TRAILTRAINING_MAX_CONSEC_HARD", "2")),
    )
    eval_p.add_argument(
        "--output",
        default=None,
        help="Optional path to write violations JSON",
    )
    eval_p.add_argument(
        "--report",
        default=None,
        help="Optional path to write full scoring report JSON",
    )
    eval_p.add_argument(
        "--soft-eval",
        action="store_true",
        help="Run rubric-based soft evaluation",
    )
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

    revise_p = sub.add_parser(
        "revise-plan",
        help="Revise coach_brief_training-plan.json using eval_report.json",
    )
    revise_p.add_argument(
        "--input",
        default=None,
        help="Path to original coach_brief_training-plan.json (default: <prompting_dir>/coach_brief_training-plan.json)",
    )
    revise_p.add_argument(
        "--report",
        default=None,
        help="Path to eval_report.json (default: <prompting_dir>/eval_report.json)",
    )
    revise_p.add_argument(
        "--rollups",
        default=None,
        help="Optional path to combined_rollups.json (default: same dir as input plan if present)",
    )
    revise_p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <prompting_dir>/revised-plan.json)",
    )
    revise_p.add_argument("--model", default=None)
    revise_p.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    revise_p.add_argument(
        "--verbosity",
        default=None,
        choices=["low", "medium", "high"],
    )
    revise_p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Only used if --reasoning-effort none.",
    )
    revise_p.add_argument(
        "--goal",
        default=None,
        help="Optional primary-goal override for the revised plan.",
    )
    revise_p.set_defaults(func=cmd_revise_plan)

    intervals_p = sub.add_parser(
        "fetch-intervals",
        help="Fetch sleep + resting HR from Intervals.icu",
    )
    intervals_p.add_argument(
        "--oldest",
        default=None,
        help="YYYY-MM-DD (default: lookback window)",
    )
    intervals_p.add_argument(
        "--newest",
        default=None,
        help="YYYY-MM-DD (default: today)",
    )
    intervals_p.set_defaults(func=cmd_fetch_intervals)

    run_all_int_p = sub.add_parser(
        "run-all-intervals",
        help="Run full pipeline (Intervals → Strava → Combine)",
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
        "--clean-prompting",
        action="store_true",
        help="Delete files in prompting/ before running.",
    )
    run_all_int_p.set_defaults(func=cmd_run_all_intervals)

    forecast_p = sub.add_parser(
        "forecast",
        help="Readiness forecast + overreach risk (from combined_summary.json)",
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

    return parser
