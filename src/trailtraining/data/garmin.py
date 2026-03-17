import glob
import os
from datetime import datetime, timedelta
from typing import Any

from trailtraining import config
from trailtraining.util.state import load_json, save_json


def combine_json_files(directory: str, output_file: str) -> None:
    """
    Combine all JSON files in a directory into a single JSON file.

    Args:
        directory (str): Path to the directory containing JSON files.
        output_file (str): Path to the output JSON file.
    """
    json_files = sorted(glob.glob(os.path.join(directory, "*.json")))
    combined_data = []

    for json_file in json_files:
        combined_data.append(load_json(json_file, default=None))

    save_json(output_file, combined_data, compact=False)


def format_personal_data(input_path: str, output_path: str) -> None:
    """
    Format personal data from Garmin JSON file.
    """
    data = load_json(input_path, default={})
    if not isinstance(data, dict):
        data = {}

    for key in ["email", "locale", "timeZone", "countryCode"]:
        data.get("userInfo", {}).pop(key, None)

    for key in [
        "userId",
        "vo2Max",
        "vo2MaxCycling",
        "functionalThresholdPower",
        "criticalSwimSpeed",
        "activityClass",
    ]:
        data.get("biometricProfile", {}).pop(key, None)

    data.pop("timeZone", None)
    data.pop("locale", None)
    data.pop("birthDate", None)
    data.pop("gender", None)

    save_json(output_path, data, compact=False)


def shorten_rhr(input_path: str, output_path: str) -> None:
    """
    Filter RHR data to only include entries from the last 200 days.
    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    today = datetime.now()
    cutoff = today - timedelta(days=200)
    data = load_json(input_path, default=[])
    if not isinstance(data, list):
        data = []

    filtered = []
    for entry in data:
        metrics = (
            entry.get("allMetrics", {}).get("metricsMap", {}).get("WELLNESS_RESTING_HEART_RATE", [])
        )
        if metrics:
            date_str = metrics[0].get("calendarDate")
            if date_str:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                if date_obj >= cutoff:
                    filtered.append(entry)

    save_json(output_path, filtered, compact=False)


def shorten_sleep(input_path: str, output_path: str) -> None:
    """
    Filter sleep data to only include entries from the last 200 days.
    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    today = datetime.now()
    cutoff = today - timedelta(days=200)
    data = load_json(input_path, default=[])
    if not isinstance(data, list):
        data = []

    filtered = []
    for entry in data:
        date_str = entry.get("calendarDate")
        if date_str:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            if date_obj >= cutoff:
                filtered.append(entry)

    save_json(output_path, filtered, compact=False)


def filter_sleep(input_path: str, output_path: str) -> None:
    """
    Convert GarminDb sleep JSON (often bulky / nested) to a lightweight list of dicts.

    Notes:
    - Handles both "dailySleepDTO" nested format and newer top-level formats.
    - Avoids pandas dependency.
    - Tolerant of sleep.json being:
        * a list[dict]
        * a list[list[dict]] (flattens)
        * a dict (wraps to list)
    """
    raw = load_json(input_path, default=[])

    entries = []
    if isinstance(raw, dict):
        entries = [raw]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, list):
                entries.extend([x for x in item if isinstance(x, dict)])
            elif isinstance(item, dict):
                entries.append(item)
            else:
                continue
    else:
        raise ValueError(f"Sleep JSON must be a list or dict. Got: {type(raw)}")

    wanted_numeric = [
        "sleepTimeSeconds",
        "deepSleepSeconds",
        "lightSleepSeconds",
        "remSleepSeconds",
        "awakeSleepSeconds",
        "bodyBatteryChange",
        "restingHeartRate",
        "restlessMomentsCount",
        "avgOvernightHrv",
    ]

    def pick(entry: dict, key: str) -> Any:
        dto = entry.get("dailySleepDTO")
        if isinstance(dto, dict) and dto.get(key) is not None:
            return dto.get(key)
        return entry.get(key)

    def to_int(v: Any, default: int = -1) -> int:
        if v is None:
            return default
        try:
            return int(float(v))
        except Exception:
            return default

    out = []
    for entry in entries:
        cal = pick(entry, "calendarDate")
        cal_s = str(cal)[:10] if cal else ""

        row: dict[str, object] = {"calendarDate": cal_s}
        for k in wanted_numeric:
            row[k] = to_int(pick(entry, k))

        hrv_status = pick(entry, "hrvStatus")
        if hrv_status is not None:
            row["hrvStatus"] = str(hrv_status)

        out.append(row)

    save_json(output_path, out, compact=False)


def main() -> None:
    runtime = config.current()
    config.ensure_directories(runtime)
    paths = runtime.paths

    combine_json_files(str(paths.rhr_directory), str(paths.processing_directory / "rhr.json"))
    combine_json_files(str(paths.sleep_directory), str(paths.processing_directory / "sleep.json"))

    format_personal_data(
        str(paths.fit_directory / "personal-information.json"),
        str(paths.prompting_directory / "formatted_personal_data.json"),
    )

    shorten_rhr(
        str(paths.processing_directory / "rhr.json"),
        str(paths.prompting_directory / "shortened_rhr.json"),
    )

    filter_sleep(
        str(paths.processing_directory / "sleep.json"),
        str(paths.processing_directory / "filtered_sleep.json"),
    )

    shorten_sleep(
        str(paths.processing_directory / "filtered_sleep.json"),
        str(paths.prompting_directory / "shortened_sleep.json"),
    )
