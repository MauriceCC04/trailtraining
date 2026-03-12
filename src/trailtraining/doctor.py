# src/trailtraining/doctor.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from trailtraining import config
from trailtraining.data.strava import default_token_path


def _ok(label: str, msg: str = "") -> None:
    print(f"✅ {label}" + (f" - {msg}" if msg else ""))


def _warn(label: str, msg: str = "") -> None:
    print(f"⚠️  {label}" + (f" - {msg}" if msg else ""))


def _bad(label: str, msg: str = "") -> None:
    print(f"❌ {label}" + (f" - {msg}" if msg else ""))


def _detect_provider(explicit: Optional[str] = None) -> str:
    # Normalize
    v = (explicit or config.WELLNESS_PROVIDER or "auto").strip().lower()
    if v in {"garmin", "intervals"}:
        return v

    # auto: prefer Intervals when configured, else Garmin when configured
    if (os.getenv("INTERVALS_API_KEY") or config.INTERVALS_API_KEY).strip():
        return "intervals"
    if config.GARMIN_EMAIL.strip() and config.GARMIN_PASSWORD.strip():
        return "garmin"

    # fallback (least setup friction)
    return "intervals"


def main() -> None:
    print("TrailTraining doctor\n")

    config.ensure_directories()
    profile = os.getenv("TRAILTRAINING_PROFILE", "default")
    base_dir = Path(config.BASE_DIR)
    _ok("Profile", profile)
    _ok("Base dir", str(base_dir))

    issues = 0

    # ---- Strava ----
    if config.STRAVA_ID and config.STRAVA_ID != 0:
        _ok("STRAVA_CLIENT_ID set")
    else:
        _bad("STRAVA_CLIENT_ID missing", "Set STRAVA_CLIENT_ID in your profile env.")
        issues += 1

    if config.STRAVA_SECRET.strip():
        _ok("STRAVA_CLIENT_SECRET set")
    else:
        _bad("STRAVA_CLIENT_SECRET missing", "Set STRAVA_CLIENT_SECRET in your profile env.")
        issues += 1

    if config.STRAVA_REDIRECT_URI.strip():
        _ok("STRAVA_REDIRECT_URI set", config.STRAVA_REDIRECT_URI)
    else:
        _warn(
            "STRAVA_REDIRECT_URI missing",
            "Default will be used, but set it explicitly to match your Strava app.",
        )

    token_path = default_token_path()
    if token_path.exists():
        _ok("Strava token", str(token_path))
    else:
        _warn("Strava token not found", f"Run: trailtraining --profile {profile} auth-strava")

    # ---- Wellness provider ----
    provider = _detect_provider()
    _ok("Wellness provider", provider)

    if provider == "intervals":
        if (os.getenv("INTERVALS_API_KEY") or config.INTERVALS_API_KEY).strip():
            _ok("INTERVALS_API_KEY set")
        else:
            _bad("INTERVALS_API_KEY missing", "Set INTERVALS_API_KEY (or set provider to garmin).")
            issues += 1

        athlete_id = (
            os.getenv("INTERVALS_ATHLETE_ID") or config.INTERVALS_ATHLETE_ID or ""
        ).strip()
        if athlete_id:
            _ok("INTERVALS_ATHLETE_ID", athlete_id)
        else:
            _warn("INTERVALS_ATHLETE_ID not set", "Default '0' may still work (current athlete).")

    if provider == "garmin":
        if config.GARMIN_EMAIL.strip():
            _ok("GARMIN_EMAIL set")
        else:
            _bad("GARMIN_EMAIL missing")
            issues += 1

        if config.GARMIN_PASSWORD.strip():
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
                "Install GarminDb and ensure garmindb_cli is on PATH (or set GARMINGDB_CLI).",
            )
            issues += 1

    # ---- Optional OpenAI ----
    if os.getenv("OPENAI_API_KEY") or os.getenv("TRAILTRAINING_OPENAI_API_KEY"):
        _ok("OpenAI API key set (coach enabled)")
    else:
        _warn("OpenAI API key not set", "Coach won't run until you set OPENAI_API_KEY.")

    print("\nSummary:")
    if issues:
        _bad("Doctor found issues", f"{issues} blocking issue(s).")
        raise SystemExit(1)
    _ok("Doctor OK", "No blocking issues found.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
