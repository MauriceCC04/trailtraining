from __future__ import annotations

from trailtraining.commands.common import _run


def cmd_auth_strava(args) -> None:
    from trailtraining.pipelines import strava

    _run(lambda: strava.auth_main(force=getattr(args, "force", False)))


def cmd_fetch_strava(_args) -> None:
    from trailtraining.pipelines import strava

    _run(strava.main)


def cmd_fetch_garmin(_args) -> None:
    from trailtraining.pipelines import garmin

    _run(garmin.main)


def cmd_combine(_args) -> None:
    from trailtraining.data import combine

    _run(combine.main)


def cmd_run_all(args) -> None:
    from trailtraining.pipelines import run_all

    _run(
        lambda: run_all.main(
            clean=getattr(args, "clean", False),
            clean_processing=getattr(args, "clean_processing", False),
            clean_prompting=getattr(args, "clean_prompting", False),
            wellness_provider=getattr(args, "wellness_provider", None),
        )
    )


def cmd_fetch_intervals(args) -> None:
    from trailtraining.pipelines import intervals

    _run(
        lambda: intervals.main(
            oldest=getattr(args, "oldest", None),
            newest=getattr(args, "newest", None),
        )
    )


def cmd_run_all_intervals(args) -> None:
    from trailtraining.pipelines import run_all_intervals

    _run(
        lambda: run_all_intervals.main(
            clean=getattr(args, "clean", False),
            clean_processing=getattr(args, "clean_processing", False),
            clean_prompting=getattr(args, "clean_prompting", False),
        )
    )


def cmd_doctor(_args) -> None:
    from trailtraining import doctor

    _run(doctor.main)


def cmd_forecast(args) -> None:
    from trailtraining.forecast.forecast import run_forecasts

    def _inner() -> None:
        result = run_forecasts(input_dir=args.input, output_path=args.output)
        print(f"[Saved] {result['saved']}")
        print(result["result"])

    _run(_inner)
