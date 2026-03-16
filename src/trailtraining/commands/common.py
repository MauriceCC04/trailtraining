from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


def _run(func: Callable[[], None]) -> None:
    try:
        func()
    except SystemExit:
        raise
    except Exception:
        log.exception("Unhandled error")
        sys.exit(1)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def apply_profile(profile: str) -> str:
    profile = (profile or "default").strip() or "default"
    os.environ["TRAILTRAINING_PROFILE"] = profile

    env_path = Path.home() / ".trailtraining" / "profiles" / f"{profile}.env"
    _load_env_file(env_path)

    os.environ.setdefault(
        "TRAILTRAINING_BASE_DIR",
        str(Path.home() / "trailtraining-data" / profile),
    )

    return profile
