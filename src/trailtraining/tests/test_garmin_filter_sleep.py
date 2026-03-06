import json
from pathlib import Path
from trailtraining.data.garmin import filter_sleep

def test_garmin_filter_sleep_pure_python(tmp_path: Path):
    inp = tmp_path / "sleep.json"
    outp = tmp_path / "filtered_sleep.json"

    sample = [
        {
            "dailySleepDTO": {
                "calendarDate": "2026-02-27",
                "sleepTimeSeconds": "7200",
                "restingHeartRate": 42,
                "avgOvernightHrv": 60,
            },
            "sleepLevels": [1, 2, 3],  # bulky field (ignored by our output)
        }
    ]
    inp.write_text(json.dumps(sample), encoding="utf-8")

    filter_sleep(str(inp), str(outp))
    out = json.loads(outp.read_text(encoding="utf-8"))

    assert isinstance(out, list) and len(out) == 1
    assert out[0]["calendarDate"] == "2026-02-27"
    assert out[0]["sleepTimeSeconds"] == 7200
    assert out[0]["restingHeartRate"] == 42
    assert out[0]["avgOvernightHrv"] == 60