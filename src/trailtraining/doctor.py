# src/trailtraining/doctor.py
from __future__ import annotations

import os
import shutil
from pathlib import Path

from trailtraining import config
from trailtraining.data.strava import default_token_path
from trailtraining.providers import resolve_wellness_provider


def _ok(label: str, msg: str = "") -> None:
    print(f"✅ {label}" + (f" - {msg}" if msg else ""))


def _warn(label: str, msg: str = "") -> None:
    print(f"⚠️ {label}" + (f" - {msg}" if msg else ""))


def _bad(label: str, msg: str = "") -> None:
    print(f"❌ {label}" + (f" - {msg}" if msg else ""))


def main() -> None:
    print("TrailTraining doctor\n")
    config.ensure_directories()

    profile = os.getenv("TRAILTRAINING_PROFILE", "default")
    base_dir = Path(config.base_dir())
    _ok("Profile", profile)
    _ok("Base dir", str(base_dir))

    issues = 0

    # ---- Strava ----
    if config.strava_id() and config.strava_id() != 0:
        _ok("STRAVA_CLIENT_ID set")
    else:
        _bad("STRAVA_CLIENT_ID missing", "Set STRAVA_CLIENT_ID in your profile env.")
        issues += 1

    if config.strava_secret():
        _ok("STRAVA_CLIENT_SECRET set")
    else:
        _bad(
            "STRAVA_CLIENT_SECRET missing",
            "Set STRAVA_CLIENT_SECRET in your profile env.",
        )
        issues += 1

    if config.strava_redirect_uri():
        _ok("STRAVA_REDIRECT_URI set", config.strava_redirect_uri())
    else:
        _warn(
            "STRAVA_REDIRECT_URI missing",
            "Default will be used, but set it explicitly to match your Strava app.",
        )

    token_path = default_token_path()
    if token_path.exists():
        _ok("Strava token", str(token_path))
    else:
        _warn(
            "Strava token not found",
            f"Run: trailtraining --profile {profile} auth-strava",
        )

    # ---- Wellness provider ----
    resolution = resolve_wellness_provider()
    provider = resolution.provider
    _ok(
        "Wellness provider",
        f"{provider} (requested={resolution.requested}, source={resolution.source})",
    )

    if provider == "intervals":
        if config.intervals_api_key():
            _ok("INTERVALS_API_KEY set")
        else:
            _bad(
                "INTERVALS_API_KEY missing",
                "Set INTERVALS_API_KEY (or set provider to garmin).",
            )
            issues += 1

        athlete_id = config.intervals_athlete_id()
        if athlete_id:
            _ok("INTERVALS_ATHLETE_ID", athlete_id)
        else:
            _warn(
                "INTERVALS_ATHLETE_ID not set",
                "Default '0' may still work (current athlete).",
            )

    if provider == "garmin":
        if config.garmin_email():
            _ok("GARMIN_EMAIL set")
        else:
            _bad("GARMIN_EMAIL missing")
            issues += 1

        if config.garmin_password():
            _ok("GARMIN_PASSWORD set")
        else:
            _bad("GARMIN_PASSWORD missing")
            issues += 1

        script = (
            os.environ.get("GARMINGDB_CLI")
            or shutil.which("garmindb_cli")
            or shutil.which("garmindb_cli.py")
        )
        if script:
            _ok("GarminDb CLI found", script)
        else:
            _bad(
                "GarminDb CLI missing",
                "Install GarminDb and ensure garmindb_cli is on PATH " "(or set GARMINGDB_CLI).",
            )
            issues += 1

    # ---- Optional OpenRouter ----
    if (
        os.getenv("OPENROUTER_API_KEY") or os.getenv("TRAILTRAINING_OPENROUTER_API_KEY") or ""
    ).strip():
        _ok("OpenRouter API key set (coach enabled)")
    else:
        _warn(
            "OpenRouter API key not set",
            "Coach won't run until you set OPENROUTER_API_KEY.",
        )

    print("\nSummary:")
    if issues:
        _bad("Doctor found issues", f"{issues} blocking issue(s).")
        raise SystemExit(1)

    _ok("Doctor OK", "No blocking issues found.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
