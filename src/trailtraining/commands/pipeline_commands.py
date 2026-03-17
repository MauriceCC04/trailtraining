from __future__ import annotations

import argparse

from trailtraining.commands.common import _run


def cmd_auth_strava(args: argparse.Namespace) -> None:
    from trailtraining.pipelines import strava

    _run(lambda: strava.auth_main(force=getattr(args, "force", False)))


def cmd_fetch_strava(args: argparse.Namespace) -> None:
    from trailtraining.pipelines import strava

    _run(strava.main)


def cmd_fetch_garmin(args: argparse.Namespace) -> None:
    from trailtraining.pipelines import garmin

    _run(garmin.main)


def cmd_combine(args: argparse.Namespace) -> None:
    from trailtraining.data import combine

    _run(combine.main)


def cmd_run_all(args: argparse.Namespace) -> None:
    from trailtraining.pipelines import run_all

    _run(
        lambda: run_all.main(
            clean=getattr(args, "clean", False),
            clean_processing=getattr(args, "clean_processing", False),
            clean_prompting=getattr(args, "clean_prompting", False),
            wellness_provider=getattr(args, "wellness_provider", None),
        )
    )


def cmd_fetch_intervals(args: argparse.Namespace) -> None:
    from trailtraining.pipelines import intervals

    _run(
        lambda: intervals.main(
            oldest=getattr(args, "oldest", None),
            newest=getattr(args, "newest", None),
        )
    )


def cmd_run_all_intervals(args: argparse.Namespace) -> None:
    from trailtraining.pipelines import run_all_intervals

    _run(
        lambda: run_all_intervals.main(
            clean=getattr(args, "clean", False),
            clean_processing=getattr(args, "clean_processing", False),
            clean_prompting=getattr(args, "clean_prompting", False),
        )
    )


def cmd_doctor(args: argparse.Namespace) -> None:
    from trailtraining import doctor

    _run(doctor.main)


def cmd_forecast(args: argparse.Namespace) -> None:
    from trailtraining.forecast.forecast import run_forecasts

    def _inner() -> None:
        result = run_forecasts(input_dir=args.input, output_path=args.output)
        print(f"[Saved] {result['saved']}")
        print(result["result"])

    _run(_inner)


def cmd_plan_to_ics(args: argparse.Namespace) -> None:
    import subprocess
    import sys

    from trailtraining import config
    from trailtraining.ics_export import export_plan_to_ics

    def _inner() -> None:
        prompting_dir = getattr(args, "input", None) or config.prompting_directory()
        plan_path, ics_path = export_plan_to_ics(
            prompting_dir,
            output_path=getattr(args, "output", None),
            start_hour=getattr(args, "start_hour", 7),
            timezone_id=getattr(args, "timezone_id", None),
        )
        print(f"[Read]  {plan_path}")
        print(f"[Saved] {ics_path}")

        if not getattr(args, "no_open", False) and sys.platform == "darwin":
            subprocess.run(["open", str(ics_path)], check=False)
            print("[Opened] Calendar.app — accept the import prompt to add events.")

    _run(_inner)
