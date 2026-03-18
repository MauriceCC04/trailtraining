# src/trailtraining/commands/parser.py
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
    cmd_plan_to_ics,
    cmd_run_all,
    cmd_run_all_intervals,
)

# ---------------------------------------------------------------------------
# Shared argument-group helpers
# ---------------------------------------------------------------------------


def _add_llm_model_args(parser: argparse.ArgumentParser) -> None:
    """Add the common --model / --reasoning-effort / --verbosity / --temperature group."""
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument(
        "--verbosity",
        default=None,
        choices=["low", "medium", "high"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Only used if --reasoning-effort none (API restriction).",
    )


def _add_goal_arg(parser: argparse.ArgumentParser) -> None:
    """Add the common --goal argument."""
    parser.add_argument(
        "--goal",
        default=None,
        help="Primary athlete goal used by generation and soft evaluation.",
    )


def _add_clean_args(parser: argparse.ArgumentParser) -> None:
    """Add the common --clean / --clean-processing / --clean-prompting group."""
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete files in BOTH processing/ and prompting/ before running.",
    )
    parser.add_argument(
        "--clean-processing",
        action="store_true",
        help="Delete files in processing/ before running.",
    )
    parser.add_argument(
        "--clean-prompting",
        action="store_true",
        help="Delete files in prompting/ before running.",
    )


def _add_input_output_args(
    parser: argparse.ArgumentParser,
    *,
    input_help: str = "Input path",
    output_help: str = "Output path",
    add_input: bool = True,
    add_output: bool = True,
) -> None:
    """Add common --input / --output arguments."""
    if add_input:
        parser.add_argument("--input", default=None, help=input_help)
    if add_output:
        parser.add_argument("--output", default=None, help=output_help)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


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

    # ---- Simple commands ----
    sub.add_parser("doctor", help="Check configuration + dependencies").set_defaults(
        func=cmd_doctor
    )

    auth_p = sub.add_parser("auth-strava", help="Run Strava auth flow (opens local server)")
    auth_p.add_argument(
        "--force",
        action="store_true",
        help="Force reauthorization even if a token exists.",
    )
    auth_p.set_defaults(func=cmd_auth_strava)

    sub.add_parser("fetch-strava", help="Fetch activities from Strava").set_defaults(
        func=cmd_fetch_strava
    )
    sub.add_parser("fetch-garmin", help="Fetch/process data from Garmin").set_defaults(
        func=cmd_fetch_garmin
    )
    sub.add_parser("combine", help="Combine Garmin + Strava JSONs").set_defaults(func=cmd_combine)

    # ---- run-all ----
    run_all_p = sub.add_parser(
        "run-all",
        help="Run full pipeline (auto: Garmin OR Intervals → Strava → Combine)",
    )
    _add_clean_args(run_all_p)
    run_all_p.add_argument(
        "--wellness-provider",
        default=None,
        choices=["auto", "garmin", "intervals"],
        help="Override wellness provider.",
    )
    run_all_p.set_defaults(func=cmd_run_all)

    # ---- coach ----
    coach_p = sub.add_parser(
        "coach",
        help="LLM coach analysis on combined_summary.json + formatted_personal_data.json",
    )
    coach_p.add_argument(
        "--prompt",
        default="training-plan",
        choices=["training-plan", "recovery-status", "meal-plan", "session-review"],
    )
    _add_llm_model_args(coach_p)
    coach_p.add_argument("--days", type=int, default=None)
    coach_p.add_argument(
        "--plan-days",
        type=int,
        default=None,
        dest="plan_days",
        choices=[7, 14, 21, 28],
        help="Output plan duration in days (default: 7, or TRAILTRAINING_PLAN_DAYS env var).",
    )
    coach_p.add_argument("--max-chars", type=int, default=None)
    _add_goal_arg(coach_p)
    _add_input_output_args(
        coach_p,
        input_help="Directory containing the two JSON files. Default: prompting directory",
        output_help="Output file. Default: training-plan -> .json, others -> .md",
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
        help="Prompt preset: changes system instructions.",
    )
    coach_p.set_defaults(func=cmd_coach)

    # ---- eval-coach ----
    eval_p = sub.add_parser(
        "eval-coach",
        help="Evaluate coach training-plan JSON output against constraints",
    )
    _add_input_output_args(
        eval_p,
        input_help="Path to coach_brief_training-plan.json",
        output_help="Optional path to write violations JSON",
    )
    eval_p.add_argument(
        "--rollups",
        default=None,
        help="Optional path to combined_rollups.json",
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
        "--report",
        default=None,
        help="Optional path to write full scoring report JSON",
    )
    eval_p.add_argument("--soft-eval", action="store_true", help="Run rubric-based soft evaluation")
    eval_p.add_argument(
        "--soft-eval-model", default=None, help="OpenRouter model for soft evaluation"
    )
    eval_p.add_argument(
        "--soft-eval-runs",
        type=int,
        default=1,
        dest="soft_eval_runs",
        metavar="N",
        help=(
            "Run the soft evaluator N times and report per-marker score variance. "
            "Values > 1 measure inter-rater reliability. "
            "Requires temperature variance (temperature > 0 used automatically). "
            "Markers with std > 0.5 on a 1-5 scale are flagged as potentially ambiguous."
        ),
    )
    _add_goal_arg(eval_p)
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

    # ---- revise-plan ----
    revise_p = sub.add_parser(
        "revise-plan",
        help="Revise coach_brief_training-plan.json using eval_report.json",
    )
    _add_input_output_args(
        revise_p,
        input_help="Path to original coach_brief_training-plan.json",
        output_help="Output JSON path (default: <prompting_dir>/revised-plan.json)",
    )
    revise_p.add_argument(
        "--report",
        default=None,
        help="Path to eval_report.json",
    )
    revise_p.add_argument(
        "--rollups",
        default=None,
        help="Optional path to combined_rollups.json",
    )
    revise_p.add_argument(
        "--auto-reeval",
        action="store_true",
        dest="auto_reeval",
        help=(
            "After revising, immediately re-evaluate the revised plan with the deterministic "
            "constraint engine and write a delta report to <stem>-reeval.json. "
            "A warning is printed if the revision degraded the score."
        ),
    )
    _add_llm_model_args(revise_p)
    _add_goal_arg(revise_p)
    revise_p.set_defaults(func=cmd_revise_plan)

    # ---- fetch-intervals ----
    intervals_p = sub.add_parser(
        "fetch-intervals",
        help="Fetch sleep + resting HR from Intervals.icu",
    )
    intervals_p.add_argument("--oldest", default=None, help="YYYY-MM-DD (default: lookback window)")
    intervals_p.add_argument("--newest", default=None, help="YYYY-MM-DD (default: today)")
    intervals_p.set_defaults(func=cmd_fetch_intervals)

    # ---- run-all-intervals ----
    run_all_int_p = sub.add_parser(
        "run-all-intervals",
        help="Run full pipeline (Intervals → Strava → Combine)",
    )
    _add_clean_args(run_all_int_p)
    run_all_int_p.set_defaults(func=cmd_run_all_intervals)

    # ---- forecast ----
    forecast_p = sub.add_parser(
        "forecast",
        help="Readiness forecast + overreach risk (from combined_summary.json)",
    )
    _add_input_output_args(
        forecast_p,
        input_help="Directory containing combined_summary.json (default: prompting dir)",
        output_help="Output JSON path (default: <input>/readiness_and_risk_forecast.json)",
    )
    forecast_p.set_defaults(func=cmd_forecast)

    # ---- plan-to-ics ----
    ics_p = sub.add_parser(
        "plan-to-ics",
        help="Export most recent training plan to a .ics calendar file",
    )
    _add_input_output_args(
        ics_p,
        input_help="Prompting directory containing the plan JSON (default: prompting dir)",
        output_help="Output .ics path (default: <prompting_dir>/training-plan.ics)",
    )
    ics_p.add_argument(
        "--start-hour",
        type=int,
        default=7,
        dest="start_hour",
        help="Hour (0-23) for timed events to start (default: 7)",
    )
    ics_p.add_argument(
        "--timezone",
        default=None,
        dest="timezone_id",
        help="IANA timezone for events (e.g. 'Europe/Rome'). Default: floating local time.",
    )
    ics_p.add_argument(
        "--no-open",
        action="store_true",
        dest="no_open",
        help="Do not auto-open with Calendar.app after writing (macOS only)",
    )
    ics_p.set_defaults(func=cmd_plan_to_ics)

    return parser
