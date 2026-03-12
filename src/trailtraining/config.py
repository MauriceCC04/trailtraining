# src/trailtraining/config.py
# Centralized configuration for directory paths and credentials

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


# Prefer TRAILTRAINING_BASE_DIR (documented in README), but keep compatibility with TRAILTRAINING_DATA_DIR.
_base = _env("TRAILTRAINING_BASE_DIR") or _env("TRAILTRAINING_DATA_DIR", "~/trailtraining-data")
BASE_DIR_PATH = Path(_base).expanduser().resolve()

# Legacy string variables used across the repo
BASE_DIR = str(BASE_DIR_PATH)

RHR_DIRECTORY = os.path.join(BASE_DIR, "RHR")
SLEEP_DIRECTORY = os.path.join(BASE_DIR, "Sleep")
FIT_DIRECTORY = os.path.join(BASE_DIR, "FitFiles")

PROCESSING_DIRECTORY = os.path.join(BASE_DIR, "processing")
PROMPTING_DIRECTORY = os.path.join(BASE_DIR, "prompting")

# Credentials (read from environment)
STRAVA_ID = int(_env("STRAVA_CLIENT_ID", "0") or "0")
STRAVA_SECRET = _env("STRAVA_CLIENT_SECRET", "")
STRAVA_REDIRECT_URI = _env("STRAVA_REDIRECT_URI", "http://127.0.0.1:5000/authorization")

GARMIN_EMAIL = _env("GARMIN_EMAIL", "")
GARMIN_PASSWORD = _env("GARMIN_PASSWORD", "")

# intervals:
# ---- Intervals.icu (wellness: sleep + HR) ----
# Personal/single-user integration (Intervals Settings → API Access)
INTERVALS_API_KEY = os.environ.get("INTERVALS_API_KEY", "")
INTERVALS_ATHLETE_ID = os.environ.get("INTERVALS_ATHLETE_ID", "0")  # "0" = current athlete

# Optional: if you later switch to OAuth (multi-user/public app)
INTERVALS_CLIENT_ID = os.environ.get("INTERVALS_CLIENT_ID", "")
INTERVALS_CLIENT_SECRET = os.environ.get("INTERVALS_CLIENT_SECRET", "")
INTERVALS_REDIRECT_URI = os.environ.get("INTERVALS_REDIRECT_URI", "")

# Optional: let your app choose the wellness provider
WELLNESS_PROVIDER = _env("TRAILTRAINING_WELLNESS_PROVIDER") or _env(
    "WELLNESS_PROVIDER", "intervals"
)


def ensure_directories() -> None:
    """Create all expected directories."""
    for d in [
        BASE_DIR,
        RHR_DIRECTORY,
        SLEEP_DIRECTORY,
        FIT_DIRECTORY,
        PROCESSING_DIRECTORY,
        PROMPTING_DIRECTORY,
    ]:
        os.makedirs(d, exist_ok=True)
