# garmin_pipeline.py
"""
Pipeline script to run all garmin data processing steps in order.
"""
from trailtraining.data import garmin as garmin_processing
from trailtraining.pipelines import download_garmin_data

import os
import shutil
import subprocess
import sys


def main():
    print("Starting Garmin data processing pipeline...")

    # Ensure per-profile GarminDb config is active BEFORE running GarminDb.
    # (Fixes: `trailtraining --profile X fetch-garmin` not setting config.)
    download_garmin_data.write_config()

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

    subprocess.run(cmd, check=True)

    print("processing Garmin data...")
    garmin_processing.main()


if __name__ == "__main__":
    main()