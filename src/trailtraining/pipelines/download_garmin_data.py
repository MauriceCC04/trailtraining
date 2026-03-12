"""
Configuration script for GarminDb. Writes config to ~/.GarminDb/GarminConnectConfig.json.

Multi-user support:
- Stores a per-profile GarminDb config at:
    ~/.trailtraining/garmin/<profile>/GarminConnectConfig.json
- Then makes ~/.GarminDb/GarminConnectConfig.json point to the active profile
  via symlink (preferred) or copy (fallback).

NOTE:
GarminDb reads ONE active config file location (~/.GarminDb/GarminConnectConfig.json).
So don't run two profiles' Garmin downloads concurrently on the same machine unless
you isolate HOME or run in separate containers/OS users.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from trailtraining.config import config


def _safe_profile_name(name: str) -> str:
    name = (name or "default").strip() or "default"
    # Keep it simple: filesystem-safe-ish
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".")) or "default"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def write_config(active_home: Optional[Path] = None) -> None:
    profile = _safe_profile_name(os.getenv("TRAILTRAINING_PROFILE", "default"))

    # Per-profile stored config (always in your real home)
    stored_dir = Path.home() / ".trailtraining" / "garmin" / profile
    stored_cfg = stored_dir / "GarminConnectConfig.json"

    # Active GarminDb config location:
    # - default: your real home (~/.GarminDb/...)
    # - when active_home is provided: <active_home>/.GarminDb/...
    home_root = active_home if active_home is not None else Path.home()
    active_dir = home_root / ".GarminDb"
    active_cfg = active_dir / "GarminConnectConfig.json"
    active_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "db": {"type": "sqlite"},
        "garmin": {"domain": "garmin.com"},
        "credentials": {
            "user": config.GARMIN_EMAIL,
            "secure_password": False,
            "password": config.GARMIN_PASSWORD,  # Do not hardcode passwords
            "password_file": None,
        },
        "data": {
            "weight_start_date": "07/08/2025",
            "sleep_start_date": "12/25/2023",
            "rhr_start_date": "12/25/2023",
            "monitoring_start_date": "12/25/2023",
            "download_latest_activities": 25,
            "download_all_activities": 1000,
        },
        "directories": {
            "relative_to_home": False,
            "base_dir": config.BASE_DIR,
            "mount_dir": "/Volumes/GARMIN",
        },
        "enabled_stats": {
            "monitoring": False,
            "steps": False,
            "itime": False,
            "sleep": True,
            "rhr": True,
            "weight": False,
            "activities": False,
        },
        "course_views": {"steps": []},
        "modes": {},
        "activities": {"display": []},
        "settings": {
            "metric": False,
            "default_display_activities": ["walking", "running", "cycling"],
        },
        "checkup": {"look_back_days": 90},
    }

    # Write per-profile config
    _atomic_write_json(stored_cfg, configuration)

    # Point <HOME>/.GarminDb/GarminConnectConfig.json to the per-profile config
    # Try symlink first; fall back to copying.
    try:
        if active_cfg.exists() or active_cfg.is_symlink():
            active_cfg.unlink()
        active_cfg.symlink_to(stored_cfg)
    except Exception:
        shutil.copyfile(stored_cfg, active_cfg)


# To download and import all data from Garmin Connect, run:
#   garmindb_cli.py --all --download --import --analyze
# To incrementally update your db:
#   garmindb_cli.py --all --download --import --analyze --latest
# To backup your DB files:
#   garmindb_cli.py --backup
