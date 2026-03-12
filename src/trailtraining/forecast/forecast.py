# src/trailtraining/forecast/forecast.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from trailtraining import config
from trailtraining.metrics.training_load import day_training_load_hours
from trailtraining.util.state import load_json, save_json


def _as_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _sleep_int(day_obj: dict[str, Any], key: str) -> Optional[float]:
    sleep = day_obj.get("sleep")
    if not isinstance(sleep, dict):
        return None
    v = sleep.get(key)
    # Treat -1 as missing
    if isinstance(v, (int, float)) and int(v) != -1:
        return float(v)
    return None


def _mean(xs: list[float]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None]
    if not xs2:
        return None
    return sum(xs2) / len(xs2)


def _std(xs: list[float]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None]
    if len(xs2) < 2:
        return None
    m = sum(xs2) / len(xs2)
    var = sum((x - m) ** 2 for x in xs2) / (len(xs2) - 1)
    return var**0.5


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _window_days(combined: list[dict[str, Any]], last_d: date, days: int) -> list[dict[str, Any]]:
    start = last_d - timedelta(days=days - 1)
    out: list[dict[str, Any]] = []
    for d in combined:
        ds = d.get("date")
        if not isinstance(ds, str):
            continue
        dd = _as_date(ds)
        if dd and start <= dd <= last_d:
            out.append(d)
    return out


def _compute_daily_series(combined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Returns list aligned with combined (assumed chronological), each:
      {date, training_load_hours, rhr}
    """
    out: list[dict[str, Any]] = []
    for day in combined:
        ds = day.get("date")
        if not isinstance(ds, str):
            continue
        out.append(
            {
                "date": ds[:10],
                "training_load_hours": float(day_training_load_hours(day)),
                "rhr": _sleep_int(day, "restingHeartRate"),
            }
        )
    return out


def _rolling_sum(values: list[float], window: int) -> list[Optional[float]]:
    """
    rolling sum ending at i (inclusive). For i < window-1 => None
    """
    if window <= 0:
        return [None] * len(values)
    out: list[Optional[float]] = [None] * len(values)
    acc = 0.0
    for i, v in enumerate(values):
        acc += float(v)
        if i >= window:
            acc -= float(values[i - window])
        if i >= window - 1:
            out[i] = acc
    return out


@dataclass(frozen=True)
class ForecastResult:
    date: str
    readiness_score: float
    readiness_status: str
    overreach_risk_score: float
    overreach_risk_level: str
    inputs: dict[str, Any]
    drivers: dict[str, list[str]]


def compute_readiness_and_risk(
    combined: list[dict[str, Any]],
) -> ForecastResult:
    if not combined:
        raise ValueError("combined_summary.json is empty")

    last_d = _as_date(combined[-1].get("date", ""))
    if not last_d:
        raise ValueError("Could not parse last date from combined_summary.json")

    w7 = _window_days(combined, last_d, 7)
    w28 = _window_days(combined, last_d, 28)

    rhr7_vals = [v for v in (_sleep_int(d, "restingHeartRate") for d in w7) if v is not None]
    rhr28_vals = [v for v in (_sleep_int(d, "restingHeartRate") for d in w28) if v is not None]

    rhr7 = _mean(rhr7_vals)
    rhr28 = _mean(rhr28_vals)
    rhr28_std = _std(rhr28_vals)

    # Training load (moving_time * load_factor)
    series = _compute_daily_series(combined)
    load_vals = [float(x["training_load_hours"]) for x in series]
    roll7 = _rolling_sum(load_vals, 7)

    last7_load = roll7[-1] if roll7 else None

    # Baseline load distribution: prior rolling windows (exclude the most recent window)
    prior_roll7 = [x for x in roll7[:-1] if x is not None]
    base7_mean = _mean([float(x) for x in prior_roll7]) if prior_roll7 else None
    base7_std = _std([float(x) for x in prior_roll7]) if prior_roll7 else None

    # --- Normalize deltas without hard-coded bpm thresholds ---
    drivers_readiness: list[str] = []
    drivers_risk: list[str] = []

    # RHR z-score (primary readiness input per your requirement)
    if rhr7 is not None and rhr28 is not None:
        delta_rhr = float(rhr7 - rhr28)
        if rhr28_std and rhr28_std > 0:
            z_rhr = delta_rhr / float(rhr28_std)
        else:
            # fallback: scale vs 2% of baseline (avoid dividing by tiny std)
            z_rhr = delta_rhr / max(1.0, float(rhr28) * 0.02)
    else:
        delta_rhr = None
        z_rhr = None

    # Load z-score (overreach context)
    if last7_load is not None and base7_mean is not None:
        delta_load = float(last7_load - base7_mean)
        if base7_std and base7_std > 0:
            z_load = delta_load / float(base7_std)
        else:
            # fallback: ratio vs baseline
            z_load = delta_load / max(0.25, float(base7_mean))
    else:
        delta_load = None
        z_load = None

    # --- Readiness score ---
    # Start from a neutral 70; adjust mainly by RHR z-score (your key requirement),
    # and modestly by acute load if it's above baseline.
    readiness = 70.0
    if z_rhr is not None:
        readiness -= 15.0 * float(z_rhr)  # higher-than-baseline RHR reduces readiness
        if delta_rhr is not None:
            if delta_rhr > 0:
                drivers_readiness.append(f"Resting HR is above 28d baseline by {delta_rhr:.2f} bpm")
            elif delta_rhr < 0:
                drivers_readiness.append(
                    f"Resting HR is below 28d baseline by {abs(delta_rhr):.2f} bpm"
                )

    if z_load is not None and z_load > 0:
        readiness -= 8.0 * float(z_load)
        drivers_readiness.append("Recent 7d training load is above your recent rolling baseline")

    readiness = _clamp(readiness, 0.0, 100.0)

    if readiness >= 75:
        status = "primed"
    elif readiness >= 50:
        status = "steady"
    else:
        status = "fatigued"

    # --- Overreach risk ---
    # Risk increases when BOTH load is elevated and RHR is elevated vs baseline.
    risk = 35.0
    if z_load is not None:
        risk += 18.0 * max(0.0, float(z_load))
        if delta_load is not None and delta_load > 0:
            drivers_risk.append("7d training load is elevated vs recent rolling baseline")

    if z_rhr is not None:
        risk += 22.0 * max(0.0, float(z_rhr))
        if delta_rhr is not None and delta_rhr > 0:
            drivers_risk.append("Resting HR is elevated vs 28d baseline (fatigue/stress signal)")

    risk = _clamp(risk, 0.0, 100.0)

    if risk >= 70:
        risk_level = "high"
    elif risk >= 40:
        risk_level = "moderate"
    else:
        risk_level = "low"

    inputs = {
        "as_of_date": last_d.isoformat(),
        "rhr_7d_mean_bpm": (round(rhr7, 2) if rhr7 is not None else None),
        "rhr_28d_mean_bpm": (round(rhr28, 2) if rhr28 is not None else None),
        "rhr_28d_std_bpm": (round(rhr28_std, 2) if rhr28_std is not None else None),
        "rhr_delta_bpm": (round(delta_rhr, 2) if delta_rhr is not None else None),
        "rhr_z": (round(z_rhr, 3) if z_rhr is not None else None),
        "training_load_7d_hours": (round(float(last7_load), 3) if last7_load is not None else None),
        "training_load_rolling7_mean_hours": (
            round(float(base7_mean), 3) if base7_mean is not None else None
        ),
        "training_load_rolling7_std_hours": (
            round(float(base7_std), 3) if base7_std is not None else None
        ),
        "training_load_delta_hours": (
            round(float(delta_load), 3) if delta_load is not None else None
        ),
        "training_load_z": (round(z_load, 3) if z_load is not None else None),
        "notes": [
            "training_load_hours = sum(moving_time_hours * load_factor)",
            "load_factor uses avgHR/maxHR when available; otherwise defaults to 1.0",
        ],
    }

    return ForecastResult(
        date=last_d.isoformat(),
        readiness_score=round(readiness, 1),
        readiness_status=status,
        overreach_risk_score=round(risk, 1),
        overreach_risk_level=risk_level,
        inputs=inputs,
        drivers={"readiness": drivers_readiness, "overreach_risk": drivers_risk},
    )


def run_forecasts(
    input_dir: Optional[str] = None, output_path: Optional[str] = None
) -> dict[str, Any]:
    """
    CLI entrypoint used by: trailtraining forecast
    Writes readiness_and_risk_forecast.json by default.
    """
    config.ensure_directories()

    base = (
        Path(input_dir).expanduser().resolve()
        if input_dir
        else Path(config.PROMPTING_DIRECTORY).expanduser().resolve()
    )
    summary_p = base / "combined_summary.json"
    if not summary_p.exists():
        raise FileNotFoundError(f"Missing {summary_p}")

    combined = load_json(summary_p, default=[])
    if not isinstance(combined, list):
        raise ValueError("combined_summary.json must be a list")

    fr = compute_readiness_and_risk(combined)

    outp = (
        Path(output_path).expanduser().resolve()
        if output_path
        else (base / "readiness_and_risk_forecast.json")
    )
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "result": {
            "date": fr.date,
            "readiness": {"score": fr.readiness_score, "status": fr.readiness_status},
            "overreach_risk": {"score": fr.overreach_risk_score, "level": fr.overreach_risk_level},
            "inputs": fr.inputs,
            "drivers": fr.drivers,
        },
    }
    save_json(outp, payload, compact=False)

    return {"saved": str(outp), "result": payload}
