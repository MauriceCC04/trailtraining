# garmin_pipeline.py
"""
Pipeline script to run all garmin data processing steps in order.
"""
from trailtraining.data import garmin as garmin_processing
import subprocess
import shutil
import os


def main():
    print("Starting Garmin data processing pipeline...")
    #update garmin data
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

    cmd = [
        script,
        "--all", "--download", "--import", "--analyze", "--latest",
    ]
    subprocess.run(cmd, check=True)
    print("processing Garmin data...")
    garmin_processing.main()

if __name__ == "__main__":
    main()
