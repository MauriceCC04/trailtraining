# garmin_pipeline.py
"""
Pipeline script to run all garmin data processing steps in order.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from trailtraining.data import garmin as garmin_processing
from trailtraining.pipelines import download_garmin_data


def main():
    print("Starting Garmin data processing pipeline...")

    # Profile-specific HOME for GarminDb so auth/session/cache can't leak across users
    profile = (os.getenv("TRAILTRAINING_PROFILE", "default") or "default").strip() or "default"
    gdb_home = Path.home() / ".trailtraining" / "garmin" / profile / "garmindb_home"
    gdb_home.mkdir(parents=True, exist_ok=True)

    # Write active GarminDb config under the profile-specific HOME
    download_garmin_data.write_config(active_home=gdb_home)

    print("Updating Garmin data...")
    script = (
        os.environ.get("GARMINGDB_CLI")
        or shutil.which("garmindb_cli")
        or shutil.which("garmindb_cli.py")
    )

    if not script:
        raise FileNotFoundError(
            "garmindb_cli not found. Install GarminDb and ensure the CLI is on your PATH, "
            "or set GARMINGDB_CLI=/full/path/to/garmindb_cli."
        )

    args = ["--all", "--download", "--import", "--analyze", "--latest"]

    # If we found a .py CLI, run it via Python for portability.
    cmd = ([sys.executable, script] if str(script).endswith(".py") else [script]) + args

    # IMPORTANT: run GarminDb with isolated HOME
    env = os.environ.copy()
    env["HOME"] = str(gdb_home)
    env.setdefault("XDG_CONFIG_HOME", str(gdb_home / ".config"))
    env.setdefault("XDG_CACHE_HOME", str(gdb_home / ".cache"))

    subprocess.run(cmd, check=True, env=env)

    print("processing Garmin data...")
    garmin_processing.main()


if __name__ == "__main__":
    main()
