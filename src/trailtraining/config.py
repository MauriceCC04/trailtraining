from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    rhr_directory: Path
    sleep_directory: Path
    fit_directory: Path
    processing_directory: Path
    prompting_directory: Path


@dataclass(frozen=True)
class RuntimeConfig:
    paths: AppPaths
    strava_id: int
    strava_secret: str
    strava_redirect_uri: str
    garmin_email: str
    garmin_password: str
    intervals_api_key: str
    intervals_athlete_id: str
    intervals_client_id: str
    intervals_client_secret: str
    intervals_redirect_uri: str
    wellness_provider: str


def _resolve_base_dir() -> Path:
    base = _env("TRAILTRAINING_BASE_DIR") or _env("TRAILTRAINING_DATA_DIR", "~/trailtraining-data")
    return Path(base).expanduser().resolve()


def _build_paths(base_dir: Path) -> AppPaths:
    return AppPaths(
        base_dir=base_dir,
        rhr_directory=base_dir / "RHR",
        sleep_directory=base_dir / "Sleep",
        fit_directory=base_dir / "FitFiles",
        processing_directory=base_dir / "processing",
        prompting_directory=base_dir / "prompting",
    )


def current() -> RuntimeConfig:
    base_dir = _resolve_base_dir()
    return RuntimeConfig(
        paths=_build_paths(base_dir),
        strava_id=int(_env("STRAVA_CLIENT_ID", "0") or "0"),
        strava_secret=_env("STRAVA_CLIENT_SECRET"),
        strava_redirect_uri=_env("STRAVA_REDIRECT_URI", "http://127.0.0.1:5000/authorization"),
        garmin_email=_env("GARMIN_EMAIL"),
        garmin_password=_env("GARMIN_PASSWORD"),
        intervals_api_key=_env("INTERVALS_API_KEY"),
        intervals_athlete_id=_env("INTERVALS_ATHLETE_ID", "0"),
        intervals_client_id=_env("INTERVALS_CLIENT_ID"),
        intervals_client_secret=_env("INTERVALS_CLIENT_SECRET"),
        intervals_redirect_uri=_env("INTERVALS_REDIRECT_URI"),
        wellness_provider=_env("TRAILTRAINING_WELLNESS_PROVIDER")
        or _env("WELLNESS_PROVIDER", "auto"),
    )


def base_dir_path() -> Path:
    return current().paths.base_dir


def base_dir() -> str:
    return str(current().paths.base_dir)


def rhr_directory() -> str:
    return str(current().paths.rhr_directory)


def sleep_directory() -> str:
    return str(current().paths.sleep_directory)


def fit_directory() -> str:
    return str(current().paths.fit_directory)


def processing_directory() -> str:
    return str(current().paths.processing_directory)


def prompting_directory() -> str:
    return str(current().paths.prompting_directory)


def strava_id() -> int:
    return current().strava_id


def strava_secret() -> str:
    return current().strava_secret


def strava_redirect_uri() -> str:
    return current().strava_redirect_uri


def garmin_email() -> str:
    return current().garmin_email


def garmin_password() -> str:
    return current().garmin_password


def intervals_api_key() -> str:
    return current().intervals_api_key


def intervals_athlete_id() -> str:
    return current().intervals_athlete_id


def intervals_client_id() -> str:
    return current().intervals_client_id


def intervals_client_secret() -> str:
    return current().intervals_client_secret


def intervals_redirect_uri() -> str:
    return current().intervals_redirect_uri


def wellness_provider_setting() -> str:
    return current().wellness_provider


def ensure_directories(runtime: Optional[RuntimeConfig] = None) -> None:
    cfg = runtime or current()
    for path in [
        cfg.paths.base_dir,
        cfg.paths.rhr_directory,
        cfg.paths.sleep_directory,
        cfg.paths.fit_directory,
        cfg.paths.processing_directory,
        cfg.paths.prompting_directory,
    ]:
        path.mkdir(parents=True, exist_ok=True)
