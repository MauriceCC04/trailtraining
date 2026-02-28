import json
import os
from trailtraining import config
from datetime import datetime, timedelta
import glob

def combine_json_files(directory: str, output_file: str) -> None:
    """
    Combine all JSON files in a directory into a single JSON file.

    Args:
        directory (str): Path to the directory containing JSON files.
        output_file (str): Path to the output JSON file.
    """
    json_files = glob.glob(os.path.join(directory, '*.json'))
    combined_data = []

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            combined_data.append(data)
    #delete the output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(combined_data, file, indent=4)

def format_personal_data(input_path, output_path) -> None:
    """
    Format personal data from Garmin JSON file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Remove specified keys from userInfo
    for key in ['email', 'locale', 'timeZone', 'countryCode']:
        data.get('userInfo', {}).pop(key, None)
    # Remove specified keys from biometricProfile
    for key in ['userId', 'vo2Max', 'vo2MaxCycling', 'functionalThresholdPower', 'criticalSwimSpeed',
                'activityClass']:
        data.get('biometricProfile', {}).pop(key, None)
    # Remove top-level timeZone
    data.pop('timeZone', None)
    data.pop('locale', None)
    data.pop('birthDate', None)
    data.pop('gender', None)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def shorten_rhr(input_path: str, output_path: str) -> None:
    """
    Filter RHR data to only include entries from the last 200 days.
    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    #delete the output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    today = datetime.now()
    cutoff = today - timedelta(days=200)
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    filtered = []
    for entry in data:
        metrics = entry.get("allMetrics", {}).get("metricsMap", {}).get("WELLNESS_RESTING_HEART_RATE", [])
        if metrics:
            date_str = metrics[0].get("calendarDate")
            if date_str:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                if date_obj >= cutoff:
                    filtered.append(entry)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(filtered, file, indent=4)

def shorten_sleep(input_path: str, output_path: str) -> None:
    """
    Filter sleep data to only include entries from the last 200 days.
    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    today = datetime.now()
    #the following needs to be changed
    #there is no more "dailySleepDTO", the entry is "calendarDate"
    cutoff = today - timedelta(days=200)
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    #delete the output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    filtered = []
    #for entry in data:
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
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(filtered, file, indent=4)



def filter_sleep(input_path: str, output_path: str) -> None:
    # Load the combined sleep JSON
    sleep_df = pd.read_json(os.path.join(config.PROCESSING_DIRECTORY, "sleep.json"))

    # Drop the bulky arrays/objects you don't need
    to_drop = [
        'wellnessSpO2SleepSummaryDTO', 'wellnessEpochSPO2DataDTOList',
        'hrvData', 'sleepBodyBattery', 'sleepStress', 'sleepHeartRate',
        'sleepMovement', 'remSleepData', 'skinTempDataExists',
        'sleepRestlessMoments', 'sleepLevels',
        'wellnessEpochRespirationDataDTOList', 'respirationVersion',
        'wellnessEpochRespirationAveragesList'
    ]
    sleep_df = sleep_df.drop(columns=[c for c in to_drop if c in sleep_df.columns], errors="ignore")

    # Helper to read from dailySleepDTO if present, else top-level
    def get_val(row, key):
        dto = row.get('dailySleepDTO')
        if isinstance(dto, dict) and key in dto:
            return dto.get(key)
        return row.get(key)

    # Extract fields robustly (works for both old and new formats)
    for key in [
        'calendarDate', 'sleepTimeSeconds', 'sleepQualityTypePK',
        'deepSleepSeconds', 'lightSleepSeconds', 'remSleepSeconds', 'awakeSleepSeconds',
        'bodyBatteryChange', 'restingHeartRate', 'restlessMomentsCount', 'avgOvernightHrv', 'hrvStatus'
    ]:
        if key not in sleep_df.columns or key == 'calendarDate':  # calendarDate might already be top-level
            sleep_df[key] = sleep_df.apply(lambda r: get_val(r, key), axis=1)

    # Drop the nested object if it exists
    sleep_df = sleep_df.drop(columns=[c for c in ['sleepQualityTypePK', 'dailySleepDTO'] if c in sleep_df.columns],
                             errors="ignore")

    # --- ONLY coerce numeric fields; do NOT touch calendarDate/hrvStatus ---
    numeric_cols = [
        'sleepTimeSeconds', 'deepSleepSeconds', 'lightSleepSeconds',
        'remSleepSeconds', 'awakeSleepSeconds', 'bodyBatteryChange',
        'restingHeartRate', 'restlessMomentsCount', 'avgOvernightHrv'
    ]
    for col in numeric_cols:
        if col in sleep_df.columns:
            sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce').fillna(-1).astype(int)

    # Keep date/text as strings
    if 'calendarDate' in sleep_df.columns:
        # ensure strings (so later strptime won't see ints)
        sleep_df['calendarDate'] = sleep_df['calendarDate'].astype(str)
    if 'hrvStatus' in sleep_df.columns:
        sleep_df['hrvStatus'] = sleep_df['hrvStatus'].astype(str)

    # Write JSON
    sleep_df.to_json(output_path, orient='records', indent=4)


def main():
    #Combine JSON files
    combine_json_files(config.RHR_DIRECTORY, os.path.join(config.PROCESSING_DIRECTORY, 'rhr.json'))
    combine_json_files(config.SLEEP_DIRECTORY, os.path.join(config.PROCESSING_DIRECTORY, 'sleep.json'))

    # Format personal data
    format_personal_data(os.path.join(config.FIT_DIRECTORY, 'personal-information.json'),
                         os.path.join(config.PROMPTING_DIRECTORY, 'formatted_personal_data.json'))

    # Shorten RHR data
    shorten_rhr(os.path.join(config.PROCESSING_DIRECTORY, 'rhr.json'),
                os.path.join(config.PROMPTING_DIRECTORY, 'shortened_rhr.json'))

    # Filter sleep data
    filter_sleep(os.path.join(config.PROCESSING_DIRECTORY, 'sleep.json'),
                 os.path.join(config.PROCESSING_DIRECTORY, 'filtered_sleep.json'))

    # Shorten sleep data
    shorten_sleep(os.path.join(config.PROCESSING_DIRECTORY, 'filtered_sleep.json'),
                  os.path.join(config.PROMPTING_DIRECTORY, 'shortened_sleep.json'))
