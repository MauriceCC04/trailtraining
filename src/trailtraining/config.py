# src/trailtraining/config.py
# Centralized configuration for directory paths and credentials

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


# ---- Runtime getters -----------------------------------------------------


def base_dir_path() -> Path:
    base = _env("TRAILTRAINING_BASE_DIR") or _env("TRAILTRAINING_DATA_DIR", "~/trailtraining-data")
    return Path(base).expanduser().resolve()


def base_dir() -> str:
    return str(base_dir_path())


def rhr_directory() -> str:
    return os.path.join(base_dir(), "RHR")


def sleep_directory() -> str:
    return os.path.join(base_dir(), "Sleep")


def fit_directory() -> str:
    return os.path.join(base_dir(), "FitFiles")


def processing_directory() -> str:
    return os.path.join(base_dir(), "processing")


def prompting_directory() -> str:
    return os.path.join(base_dir(), "prompting")


def strava_id() -> int:
    return int(_env("STRAVA_CLIENT_ID", "0") or "0")


def strava_secret() -> str:
    return _env("STRAVA_CLIENT_SECRET", "")


def strava_redirect_uri() -> str:
    return _env("STRAVA_REDIRECT_URI", "http://127.0.0.1:5000/authorization")


def garmin_email() -> str:
    return _env("GARMIN_EMAIL", "")


def garmin_password() -> str:
    return _env("GARMIN_PASSWORD", "")


def intervals_api_key() -> str:
    return _env("INTERVALS_API_KEY", "")


def intervals_athlete_id() -> str:
    return _env("INTERVALS_ATHLETE_ID", "0")  # "0" = current athlete


def intervals_client_id() -> str:
    return _env("INTERVALS_CLIENT_ID", "")


def intervals_client_secret() -> str:
    return _env("INTERVALS_CLIENT_SECRET", "")


def intervals_redirect_uri() -> str:
    return _env("INTERVALS_REDIRECT_URI", "")


def wellness_provider_setting() -> str:
    # Prefer the namespaced env var, but keep compatibility with WELLNESS_PROVIDER.
    # Default to "auto" so provider resolution can inspect the configured credentials.
    return _env("TRAILTRAINING_WELLNESS_PROVIDER") or _env("WELLNESS_PROVIDER", "auto")


def ensure_directories() -> None:
    """Create all expected directories using runtime configuration."""
    for d in [
        base_dir(),
        rhr_directory(),
        sleep_directory(),
        fit_directory(),
        processing_directory(),
        prompting_directory(),
    ]:
        os.makedirs(d, exist_ok=True)


# ---- Legacy snapshot constants -------------------------------------------
# Keep these for backward compatibility with the rest of the repo.
# Runtime-sensitive paths/credentials should use the getter functions above.

BASE_DIR_PATH = base_dir_path()
BASE_DIR = base_dir()

RHR_DIRECTORY = rhr_directory()
SLEEP_DIRECTORY = sleep_directory()
FIT_DIRECTORY = fit_directory()

PROCESSING_DIRECTORY = processing_directory()
PROMPTING_DIRECTORY = prompting_directory()

STRAVA_ID = strava_id()
STRAVA_SECRET = strava_secret()
STRAVA_REDIRECT_URI = strava_redirect_uri()

GARMIN_EMAIL = garmin_email()
GARMIN_PASSWORD = garmin_password()

INTERVALS_API_KEY = intervals_api_key()
INTERVALS_ATHLETE_ID = intervals_athlete_id()
INTERVALS_CLIENT_ID = intervals_client_id()
INTERVALS_CLIENT_SECRET = intervals_client_secret()
INTERVALS_REDIRECT_URI = intervals_redirect_uri()

WELLNESS_PROVIDER = wellness_provider_setting()
