import glob
import json
import os
from datetime import datetime, timedelta

from trailtraining import config


def combine_json_files(directory: str, output_file: str) -> None:
    """
    Combine all JSON files in a directory into a single JSON file.

    Args:
        directory (str): Path to the directory containing JSON files.
        output_file (str): Path to the output JSON file.
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))
    combined_data = []

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as file:
            data = json.load(file)
            combined_data.append(data)
    # delete the output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(combined_data, file, indent=4)


def format_personal_data(input_path, output_path) -> None:
    """
    Format personal data from Garmin JSON file.
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    # Remove specified keys from userInfo
    for key in ["email", "locale", "timeZone", "countryCode"]:
        data.get("userInfo", {}).pop(key, None)
    # Remove specified keys from biometricProfile
    for key in [
        "userId",
        "vo2Max",
        "vo2MaxCycling",
        "functionalThresholdPower",
        "criticalSwimSpeed",
        "activityClass",
    ]:
        data.get("biometricProfile", {}).pop(key, None)
    # Remove top-level timeZone
    data.pop("timeZone", None)
    data.pop("locale", None)
    data.pop("birthDate", None)
    data.pop("gender", None)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def shorten_rhr(input_path: str, output_path: str) -> None:
    """
    Filter RHR data to only include entries from the last 200 days.
    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    # delete the output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    today = datetime.now()
    cutoff = today - timedelta(days=200)
    with open(input_path, encoding="utf-8") as file:
        data = json.load(file)
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
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(filtered, file, indent=4)


def shorten_sleep(input_path: str, output_path: str) -> None:
    """
    Filter sleep data to only include entries from the last 200 days.
    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    today = datetime.now()
    # the following needs to be changed
    # there is no more "dailySleepDTO", the entry is "calendarDate"
    cutoff = today - timedelta(days=200)
    with open(input_path, encoding="utf-8") as file:
        data = json.load(file)
    # delete the output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    filtered = []
    # for entry in data:
    #    metrics = entry.get("dailySleepDTO", {})
    #    if metrics:
    #        date_str = metrics.get("calendarDate")
    #        if date_str:
    #            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    #            if date_obj >= cutoff:
    #                filtered.append(entry)
    for entry in data:
        date_str = entry.get("calendarDate")
        if date_str:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            if date_obj >= cutoff:
                filtered.append(entry)
    # Write the filtered data to the output file
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(filtered, file, indent=4)


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
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Normalize shapes
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

    def pick(entry: dict, key: str):
        dto = entry.get("dailySleepDTO")
        if isinstance(dto, dict) and dto.get(key) is not None:
            return dto.get(key)
        return entry.get(key)

    def to_int(v, default: int = -1) -> int:
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

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)


def main():
    # Combine JSON files
    combine_json_files(config.RHR_DIRECTORY, os.path.join(config.PROCESSING_DIRECTORY, "rhr.json"))
    combine_json_files(
        config.SLEEP_DIRECTORY, os.path.join(config.PROCESSING_DIRECTORY, "sleep.json")
    )

    # Format personal data
    format_personal_data(
        os.path.join(config.FIT_DIRECTORY, "personal-information.json"),
        os.path.join(config.PROMPTING_DIRECTORY, "formatted_personal_data.json"),
    )

    # Shorten RHR data
    shorten_rhr(
        os.path.join(config.PROCESSING_DIRECTORY, "rhr.json"),
        os.path.join(config.PROMPTING_DIRECTORY, "shortened_rhr.json"),
    )

    # Filter sleep data
    filter_sleep(
        os.path.join(config.PROCESSING_DIRECTORY, "sleep.json"),
        os.path.join(config.PROCESSING_DIRECTORY, "filtered_sleep.json"),
    )

    # Shorten sleep data
    shorten_sleep(
        os.path.join(config.PROCESSING_DIRECTORY, "filtered_sleep.json"),
        os.path.join(config.PROMPTING_DIRECTORY, "shortened_sleep.json"),
    )
