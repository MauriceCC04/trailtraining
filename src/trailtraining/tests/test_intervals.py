from trailtraining.pipelines.intervals import normalize_to_filtered_sleep

def test_intervals_normalize_basic():
    x = {"id": "2026-02-27", "sleepSecs": 3600, "restingHR": 45, "avgOvernightHrv": 55}
    out = normalize_to_filtered_sleep(x)
    assert out["calendarDate"] == "2026-02-27"
    assert out["sleepTimeSeconds"] == 3600
    assert out["restingHeartRate"] == 45
    assert out["avgOvernightHrv"] == 55