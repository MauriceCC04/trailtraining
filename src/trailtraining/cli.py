from __future__ import annotations

import os

from trailtraining.commands.common import _env_truthy, _load_env_file, _run, apply_profile
from trailtraining.commands.llm_commands import (
    cmd_coach,
    cmd_eval_coach,
    cmd_revise_plan,
)
from trailtraining.commands.parser import build_parser
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
from trailtraining.util.logging_config import configure_logging

__all__ = [
    "_env_truthy",
    "_load_env_file",
    "_run",
    "apply_profile",
    "cmd_auth_strava",
    "cmd_fetch_strava",
    "cmd_fetch_garmin",
    "cmd_combine",
    "cmd_run_all",
    "cmd_fetch_intervals",
    "cmd_run_all_intervals",
    "cmd_doctor",
    "cmd_forecast",
    "cmd_coach",
    "cmd_eval_coach",
    "cmd_revise_plan",
    "main",
]


def main(argv: list[str] | None = None) -> None:
    configure_logging(os.getenv("TRAILTRAINING_LOG_LEVEL", "INFO"))
    parser = build_parser()
    args = parser.parse_args(argv)

    apply_profile(args.profile)
    configure_logging(args.log_level)

    args.func(args)


if __name__ == "__main__":
    main()
