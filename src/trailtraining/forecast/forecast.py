# src/trailtraining/forecast/forecast.py
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

from trailtraining import config
from trailtraining.contracts import (
    ForecastArtifact,
    ForecastDrivers,
    ForecastInputs,
    ForecastReadiness,
    ForecastResultArtifact,
    ForecastRisk,
)
from trailtraining.metrics.training_load import day_training_load_hours, latest_atl_ctl_tsb
from trailtraining.util.dates import _as_date
from trailtraining.util.errors import ArtifactError, DataValidationError
from trailtraining.util.state import load_json, save_json

log = logging.getLogger(__name__)

ReadinessStatus = Literal["primed", "steady", "fatigued"]
RiskLevel = Literal["low", "moderate", "high"]


# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastConfig:
    """
    All numeric weights and thresholds for readiness and overreach-risk scoring.

    Readiness starts at `readiness_baseline` and is adjusted by each available
    signal's z-score multiplied by its weight.  Risk starts at `risk_baseline`
    and increases when signals suggest overreaching.

    The z-score for each signal is:
        (recent_7d_mean - baseline_28d_mean) / baseline_28d_std
    When std is too small we fall back to a fixed percentage of the baseline
    mean as a proxy std.

    Signal directions (positive z = ...):
        RHR:   higher than baseline → fatigued     → readiness ↓, risk ↑
        Load:  higher than baseline → high demand  → readiness ↓ (if z>0), risk ↑
        Sleep: higher than baseline → well-rested  → readiness ↑, risk ↓
        HRV:   higher than baseline → well-rested  → readiness ↑, risk ↓

    All weights can be overridden without editing source via env vars of the
    form TRAILTRAINING_FORECAST_{FIELD_NAME_UPPER}.  For example:
        TRAILTRAINING_FORECAST_RHR_READINESS_WEIGHT=12
    """

    # --- starting points -------------------------------------------------------
    readiness_baseline: float = 70.0
    """Neutral readiness before any signals are applied (mid-range of 0-100)."""

    risk_baseline: float = 35.0
    """
    Baseline overreach risk before any signals are applied.
    35 puts an athlete with no recovery telemetry in the low-to-moderate zone,
    reflecting genuine uncertainty rather than assumed safety.
    """

    # --- readiness weights (per z-score unit) ----------------------------------
    rhr_readiness_weight: float = 15.0
    """
    RHR z-score contribution (subtracted — elevated RHR = worse readiness).
    1 SD above baseline drops readiness by 15 pts.
    """

    load_readiness_weight: float = 8.0
    """
    Load z-score contribution (subtracted when load is above baseline).
    Smaller than RHR weight because elevated load is expected during build phases.
    """

    sleep_readiness_weight: float = 8.0
    """
    Sleep z-score contribution (added — more sleep vs baseline = better readiness).
    Applied bidirectionally: less sleep reduces readiness, more sleep improves it.
    """

    hrv_readiness_weight: float = 6.0
    """
    HRV z-score contribution (added — higher HRV = better readiness).
    Slightly smaller weight than sleep/RHR because HRV is noisier day-to-day.
    """

    # --- risk weights (per z-score unit) --------------------------------------
    load_risk_weight: float = 18.0
    """Risk increase per unit of elevated load z-score. Only applied when z > 0."""

    rhr_risk_weight: float = 22.0
    """
    Risk increase per unit of elevated RHR z-score.
    Highest weight because RHR elevation is the most reliable single-metric
    fatigue indicator we have.
    """

    sleep_risk_weight: float = 8.0
    """Risk increase per unit of reduced sleep z-score. Only applied when z < 0."""

    hrv_risk_weight: float = 10.0
    """Risk increase per unit of reduced HRV z-score. Only applied when z < 0."""

    # --- readiness status thresholds ------------------------------------------
    primed_threshold: float = 75.0
    """Score at or above this → 'primed'."""

    steady_threshold: float = 50.0
    """Score at or above this (but below primed) → 'steady'. Below → 'fatigued'."""

    # --- risk level thresholds ------------------------------------------------
    high_risk_threshold: float = 70.0
    """Score at or above this → 'high'."""

    moderate_risk_threshold: float = 40.0
    """Score at or above this (but below high) → 'moderate'. Below → 'low'."""

    # --- z-score fallback std proxies (fraction of 28d mean) ------------------
    rhr_std_fallback_pct: float = 0.02
    """When RHR std is unavailable, use 2% of the 28d RHR mean as a std proxy."""

    load_std_fallback_pct: float = 0.25
    """When load std is unavailable, use 25% of the rolling baseline mean."""

    sleep_std_fallback_pct: float = 0.10
    """When sleep std is unavailable, use 10% of the 28d sleep mean."""

    hrv_std_fallback_pct: float = 0.10
    """When HRV std is unavailable, use 10% of the 28d HRV mean."""

    # --- data quality gate ----------------------------------------------------
    min_signal_days: int = 2
    """
    Minimum days of data in the 7-day window required to use a signal in scoring.
    Prevents a single outlier day from dominating the score.
    """

    @classmethod
    def from_env(cls) -> ForecastConfig:
        """Return a config with any fields overridden by env vars."""
        import os

        defaults = cls()

        def _f(name: str, default: float) -> float:
            raw = os.getenv(f"TRAILTRAINING_FORECAST_{name.upper()}")
            if raw is None or not raw.strip():
                return default
            try:
                return float(raw)
            except ValueError:
                log.warning("Invalid float for TRAILTRAINING_FORECAST_%s: %r", name.upper(), raw)
                return default

        def _i(name: str, default: int) -> int:
            raw = os.getenv(f"TRAILTRAINING_FORECAST_{name.upper()}")
            if raw is None or not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError:
                log.warning("Invalid int for TRAILTRAINING_FORECAST_%s: %r", name.upper(), raw)
                return default

        return cls(
            readiness_baseline=_f("readiness_baseline", defaults.readiness_baseline),
            risk_baseline=_f("risk_baseline", defaults.risk_baseline),
            rhr_readiness_weight=_f("rhr_readiness_weight", defaults.rhr_readiness_weight),
            load_readiness_weight=_f("load_readiness_weight", defaults.load_readiness_weight),
            sleep_readiness_weight=_f("sleep_readiness_weight", defaults.sleep_readiness_weight),
            hrv_readiness_weight=_f("hrv_readiness_weight", defaults.hrv_readiness_weight),
            load_risk_weight=_f("load_risk_weight", defaults.load_risk_weight),
            rhr_risk_weight=_f("rhr_risk_weight", defaults.rhr_risk_weight),
            sleep_risk_weight=_f("sleep_risk_weight", defaults.sleep_risk_weight),
            hrv_risk_weight=_f("hrv_risk_weight", defaults.hrv_risk_weight),
            primed_threshold=_f("primed_threshold", defaults.primed_threshold),
            steady_threshold=_f("steady_threshold", defaults.steady_threshold),
            high_risk_threshold=_f("high_risk_threshold", defaults.high_risk_threshold),
            moderate_risk_threshold=_f("moderate_risk_threshold", defaults.moderate_risk_threshold),
            rhr_std_fallback_pct=_f("rhr_std_fallback_pct", defaults.rhr_std_fallback_pct),
            load_std_fallback_pct=_f("load_std_fallback_pct", defaults.load_std_fallback_pct),
            sleep_std_fallback_pct=_f("sleep_std_fallback_pct", defaults.sleep_std_fallback_pct),
            hrv_std_fallback_pct=_f("hrv_std_fallback_pct", defaults.hrv_std_fallback_pct),
            min_signal_days=_i("min_signal_days", defaults.min_signal_days),
        )


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _mean(xs: list[float]) -> Optional[float]:
    valid = [x for x in xs if x is not None]
    return sum(valid) / len(valid) if valid else None


def _std(xs: list[float]) -> Optional[float]:
    valid = [x for x in xs if x is not None]
    if len(valid) < 2:
        return None
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / (len(valid) - 1))


def _z_score(
    recent: float,
    baseline: float,
    std: Optional[float],
    fallback_pct: float,
) -> float:
    """
    Compute (recent - baseline) / std.
    When std is None or near-zero, substitute fallback_pct * |baseline| as the
    denominator so the score degrades gracefully rather than dividing by zero.
    """
    denom = std if (std is not None and std > 0) else max(abs(baseline) * fallback_pct, 1e-6)
    return (recent - baseline) / denom


def normalize_readiness_status(value: str) -> ReadinessStatus:
    normalized = value.strip().lower()
    if normalized == "primed":
        return "primed"
    if normalized == "steady":
        return "steady"
    if normalized == "fatigued":
        return "fatigued"
    raise ValueError(f"Invalid readiness status: {value!r}")


def normalize_risk_level(value: str) -> RiskLevel:
    normalized = value.strip().lower()
    if normalized == "low":
        return "low"
    if normalized == "moderate":
        return "moderate"
    if normalized == "high":
        return "high"
    raise ValueError(f"Invalid risk level: {value!r}")


# ---------------------------------------------------------------------------
# Day-level data extraction helpers
# ---------------------------------------------------------------------------


def _sleep_hours(day_obj: dict[str, Any]) -> Optional[float]:
    sleep = day_obj.get("sleep")
    if not isinstance(sleep, dict):
        return None
    secs = sleep.get("sleepTimeSeconds")
    if isinstance(secs, (int, float)) and float(secs) > 0:
        return float(secs) / 3600.0
    return None


def _sleep_int(day_obj: dict[str, Any], key: str) -> Optional[float]:
    """Return a numeric sleep metric, treating -1 as missing."""
    sleep = day_obj.get("sleep")
    if not isinstance(sleep, dict):
        return None
    v = sleep.get(key)
    if isinstance(v, (int, float)) and int(v) != -1:
        return float(v)
    return None


def _recent_signal_counts(days: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "sleep": sum(1 for d in days if _sleep_hours(d) is not None),
        "resting_hr": sum(1 for d in days if _sleep_int(d, "restingHeartRate") is not None),
        "hrv": sum(1 for d in days if _sleep_int(d, "avgOvernightHrv") is not None),
    }


def _recovery_capability_from_counts(
    counts: dict[str, int], *, min_days: int = 2
) -> dict[str, str]:
    """
    Summarise which recovery signals have enough data to be trustworthy.
    Returns a human-readable label and a stable key for downstream use.
    """
    active: list[tuple[str, str]] = []
    if counts.get("sleep", 0) >= min_days:
        active.append(("sleep", "sleep"))
    if counts.get("resting_hr", 0) >= min_days:
        active.append(("resting_hr", "resting HR"))
    if counts.get("hrv", 0) >= min_days:
        active.append(("hrv", "HRV"))

    if not active:
        return {"key": "load_only", "label": "I only have training data"}

    parts = [label for _, label in active]
    key = "load_" + "_".join(k for k, _ in active)

    if len(parts) == 1:
        label = f"I have load + {parts[0]} only"
    else:
        label = "I have load + " + " + ".join(parts)

    return {"key": key, "label": label}


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------


def _window_days(combined: list[dict[str, Any]], last_d: date, days: int) -> list[dict[str, Any]]:
    start = last_d - timedelta(days=days - 1)
    return [
        d for d in combined if (dd := _as_date(str(d.get("date", "")))) and start <= dd <= last_d
    ]


# ---------------------------------------------------------------------------
# Training-load rolling series (used for baseline distribution, not just last7)
# ---------------------------------------------------------------------------


def _compute_daily_load_series(combined: list[dict[str, Any]]) -> list[float]:
    """Return per-day training load hours in the same order as combined."""
    return [float(day_training_load_hours(day)) for day in combined]


def _rolling_sum(values: list[float], window: int) -> list[Optional[float]]:
    """Rolling sum ending at index i (inclusive). Returns None when i < window-1."""
    if window <= 0:
        return [None] * len(values)
    out: list[Optional[float]] = [None] * len(values)
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        if i >= window - 1:
            out[i] = acc
    return out


# ---------------------------------------------------------------------------
# Rollups extraction (clean helper replacing the 40-line nested block)
# ---------------------------------------------------------------------------


def _extract_last7_load_from_rollups(
    rollups: dict[str, Any], last_d: date
) -> tuple[Optional[float], Optional[str]]:
    """
    Pull total_training_load_hours from combined_rollups.json windows['7'].

    Returns (load_hours, None) on success, or (None, warning_message) on any
    validation failure so the caller can fall back cleanly.
    """
    windows = rollups.get("windows")
    if not isinstance(windows, dict):
        return None, "combined_rollups.json missing a valid 'windows' object"

    w7 = windows.get("7")
    if not isinstance(w7, dict):
        return None, "combined_rollups.json missing windows['7']"

    end_date_raw = w7.get("end_date")
    end_date_str = end_date_raw[:10] if isinstance(end_date_raw, str) else ""
    if end_date_str != last_d.isoformat():
        return None, (
            f"rollups windows['7'].end_date {end_date_str!r} does not match "
            f"combined_summary.json last date {last_d.isoformat()!r}"
        )

    acts = w7.get("activities")
    if not isinstance(acts, dict):
        return None, "rollups windows['7'].activities is missing or not an object"

    v = acts.get("total_training_load_hours")
    if not isinstance(v, (int, float)):
        return (
            None,
            "rollups windows['7'].activities.total_training_load_hours is missing or invalid",
        )

    return float(v), None


# ---------------------------------------------------------------------------
# Result dataclass (public API — unchanged from original)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastResult:
    date: str
    readiness_score: float
    readiness_status: str
    overreach_risk_score: float
    overreach_risk_level: str
    inputs: dict[str, Any]
    drivers: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_readiness_and_risk(
    combined: list[dict[str, Any]],
    rollups: Optional[dict[str, Any]] = None,
    cfg: Optional[ForecastConfig] = None,
) -> ForecastResult:
    """
    Compute readiness and overreach risk from daily combined data.

    Signal priority:
        1. Training load (always available if activities exist)
        2. Resting HR (used when ≥ cfg.min_signal_days in the 7d window)
        3. Sleep duration (used when ≥ cfg.min_signal_days in the 7d window)
        4. HRV (used when ≥ cfg.min_signal_days in the 7d window)

    Each available signal adjusts the score up or down via a z-score relative
    to its own 28-day baseline.  Missing signals are surfaced in data_notes and
    the recovery_capability label rather than silently ignored.
    """
    if cfg is None:
        cfg = ForecastConfig.from_env()

    if not combined:
        raise DataValidationError(
            message="combined_summary.json is empty",
            hint="Run the earlier pipeline steps first.",
        )

    last_d = _as_date(str(combined[-1].get("date", "")))
    if not last_d:
        raise DataValidationError(
            message="Could not parse last date from combined_summary.json",
            hint="Expected the last item to have a valid ISO date string in 'date'.",
        )

    # Build time windows
    w7 = _window_days(combined, last_d, 7)
    w28 = _window_days(combined, last_d, 28)

    # ---- Signal counts and capability label --------------------------------
    signal_counts_7d = _recent_signal_counts(w7)
    capability = _recovery_capability_from_counts(signal_counts_7d, min_days=cfg.min_signal_days)

    # ---- RHR ---------------------------------------------------------------
    rhr7_vals = [v for d in w7 if (v := _sleep_int(d, "restingHeartRate")) is not None]
    rhr28_vals = [v for d in w28 if (v := _sleep_int(d, "restingHeartRate")) is not None]
    rhr7 = _mean(rhr7_vals)
    rhr28 = _mean(rhr28_vals)
    rhr28_std = _std(rhr28_vals)

    # ---- Sleep -------------------------------------------------------------
    sleep7_vals = [v for d in w7 if (v := _sleep_hours(d)) is not None]
    sleep28_vals = [v for d in w28 if (v := _sleep_hours(d)) is not None]
    sleep7 = _mean(sleep7_vals)
    sleep28 = _mean(sleep28_vals)
    sleep28_std = _std(sleep28_vals)

    # ---- HRV ---------------------------------------------------------------
    hrv7_vals = [v for d in w7 if (v := _sleep_int(d, "avgOvernightHrv")) is not None]
    hrv28_vals = [v for d in w28 if (v := _sleep_int(d, "avgOvernightHrv")) is not None]
    hrv7 = _mean(hrv7_vals)
    hrv28 = _mean(hrv28_vals)
    hrv28_std = _std(hrv28_vals)

    # ---- Training load -----------------------------------------------------
    # The daily series is always computed from combined_summary.json.
    # We use it to derive the prior rolling-window baseline (base7_mean / base7_std).
    # The rollup may override last7_load with a more authoritative figure, but the
    # baseline distribution must still come from the series so the two are comparable.
    load_vals = _compute_daily_load_series(combined)
    roll7 = _rolling_sum(load_vals, 7)

    last7_load: Optional[float] = roll7[-1] if roll7 else None
    used_rollups_last7 = False
    rollups_notes: list[str] = []

    if isinstance(rollups, dict):
        rollup_load, warning = _extract_last7_load_from_rollups(rollups, last_d)
        if warning:
            log.warning("%s — falling back to rolling sum from combined_summary.json.", warning)
            rollups_notes.append(f"Rollups fallback: {warning}")
        else:
            last7_load = rollup_load
            used_rollups_last7 = True

    # Prior rolling windows (exclude the current 7-day slice) form the baseline.
    prior_roll7 = [float(x) for x in roll7[:-1] if x is not None]
    base7_mean = _mean(prior_roll7)
    base7_std = _std(prior_roll7)
    load_model = latest_atl_ctl_tsb(combined)
    atl_load = None
    ctl_load = None
    tsb_load = None

    if isinstance(load_model, dict):
        v = load_model.get("atl_load_h")
        atl_load = float(v) if isinstance(v, (int, float)) else None

        v = load_model.get("ctl_load_h")
        ctl_load = float(v) if isinstance(v, (int, float)) else None

        v = load_model.get("tsb_load_h")
        tsb_load = float(v) if isinstance(v, (int, float)) else None

    # ---- Z-scores ----------------------------------------------------------
    # Only compute a z-score when we have enough recent data to trust it.

    z_rhr: Optional[float] = None
    delta_rhr: Optional[float] = None
    if (
        rhr7 is not None
        and rhr28 is not None
        and signal_counts_7d["resting_hr"] >= cfg.min_signal_days
    ):
        delta_rhr = rhr7 - rhr28
        z_rhr = _z_score(rhr7, rhr28, rhr28_std, cfg.rhr_std_fallback_pct)

    z_load: Optional[float] = None
    delta_load: Optional[float] = None
    if last7_load is not None and base7_mean is not None:
        delta_load = last7_load - base7_mean
        z_load = _z_score(last7_load, base7_mean, base7_std, cfg.load_std_fallback_pct)

    z_sleep: Optional[float] = None
    delta_sleep: Optional[float] = None
    if (
        sleep7 is not None
        and sleep28 is not None
        and signal_counts_7d["sleep"] >= cfg.min_signal_days
    ):
        delta_sleep = sleep7 - sleep28
        z_sleep = _z_score(sleep7, sleep28, sleep28_std, cfg.sleep_std_fallback_pct)

    z_hrv: Optional[float] = None
    delta_hrv: Optional[float] = None
    if hrv7 is not None and hrv28 is not None and signal_counts_7d["hrv"] >= cfg.min_signal_days:
        delta_hrv = hrv7 - hrv28
        z_hrv = _z_score(hrv7, hrv28, hrv28_std, cfg.hrv_std_fallback_pct)

    # ---- Readiness score ---------------------------------------------------
    # Start from neutral baseline; each signal adjusts up or down.
    drivers_readiness: list[str] = []
    readiness = cfg.readiness_baseline

    if z_rhr is not None:
        # High RHR → fatigued → subtract
        readiness -= cfg.rhr_readiness_weight * z_rhr
        if delta_rhr is not None:
            if delta_rhr > 0:
                drivers_readiness.append(f"Resting HR is {delta_rhr:.1f} bpm above 28d baseline")
            elif delta_rhr < 0:
                drivers_readiness.append(
                    f"Resting HR is {abs(delta_rhr):.1f} bpm below 28d baseline"
                )

    if z_load is not None and z_load > 0:
        # Elevated load only penalises readiness; below-baseline load is neutral
        readiness -= cfg.load_readiness_weight * z_load
        drivers_readiness.append("Recent 7d training load is above your rolling baseline")

    if z_sleep is not None:
        # More sleep → better readiness; less sleep → worse (bidirectional)
        readiness += cfg.sleep_readiness_weight * z_sleep
        if delta_sleep is not None:
            if delta_sleep < 0:
                drivers_readiness.append(f"Sleep is {abs(delta_sleep):.1f}h below 28d baseline")
            elif delta_sleep > 0:
                drivers_readiness.append(f"Sleep is {delta_sleep:.1f}h above 28d baseline")

    if z_hrv is not None:
        # Higher HRV → better readiness (bidirectional)
        readiness += cfg.hrv_readiness_weight * z_hrv
        if delta_hrv is not None:
            if delta_hrv < 0:
                drivers_readiness.append(f"HRV is {abs(delta_hrv):.1f}ms below 28d baseline")
            elif delta_hrv > 0:
                drivers_readiness.append(f"HRV is {delta_hrv:.1f}ms above 28d baseline")

    readiness = _clamp(readiness, 0.0, 100.0)

    if readiness >= cfg.primed_threshold:
        status: ReadinessStatus = "primed"
    elif readiness >= cfg.steady_threshold:
        status = "steady"
    else:
        status = "fatigued"

    # ---- Overreach risk score ----------------------------------------------
    # Each signal only pushes risk up when it is elevated in the bad direction.
    # (Elevated RHR, reduced sleep, reduced HRV, and elevated load all increase risk.)
    drivers_risk: list[str] = []
    risk = cfg.risk_baseline

    if z_load is not None and z_load > 0:
        risk += cfg.load_risk_weight * z_load
        if delta_load is not None:
            drivers_risk.append("7d training load is elevated vs recent rolling baseline")

    if z_rhr is not None and z_rhr > 0:
        risk += cfg.rhr_risk_weight * z_rhr
        if delta_rhr is not None:
            drivers_risk.append(
                f"Resting HR is {delta_rhr:.1f} bpm above 28d baseline (fatigue/stress signal)"
            )

    if z_sleep is not None and z_sleep < 0:
        # Sleep below baseline → risk up
        risk += cfg.sleep_risk_weight * abs(z_sleep)
        if delta_sleep is not None:
            drivers_risk.append(
                f"Sleep is {abs(delta_sleep):.1f}h below 28d baseline (reduced recovery)"
            )

    if z_hrv is not None and z_hrv < 0:
        # HRV below baseline → risk up
        risk += cfg.hrv_risk_weight * abs(z_hrv)
        if delta_hrv is not None:
            drivers_risk.append(
                f"HRV is {abs(delta_hrv):.1f}ms below 28d baseline (fatigue signal)"
            )

    risk = _clamp(risk, 0.0, 100.0)

    if risk >= cfg.high_risk_threshold:
        risk_level: RiskLevel = "high"
    elif risk >= cfg.moderate_risk_threshold:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # ---- Build notes -------------------------------------------------------
    notes: list[str] = [
        "training_load_hours = sum(moving_time_hours * load_factor)",
        "load_factor uses avgHR/maxHR when available; otherwise defaults to 1.0",
        f"Recovery telemetry capability: {capability['label']}.",
        (
            f"Recent 7d usable recovery days: sleep={signal_counts_7d['sleep']}, "
            f"resting_hr={signal_counts_7d['resting_hr']}, hrv={signal_counts_7d['hrv']}."
        ),
    ]
    if used_rollups_last7:
        notes.append(
            "Used combined_rollups.json windows['7'].activities.total_training_load_hours "
            "as the authoritative last-7d training load."
        )
    notes.extend(rollups_notes)

    signals_used = []
    if z_rhr is not None:
        signals_used.append("resting_hr")
    if z_sleep is not None:
        signals_used.append("sleep")
    if z_hrv is not None:
        signals_used.append("hrv")
    if z_load is not None:
        signals_used.append("load")
    if atl_load is not None and ctl_load is not None and tsb_load is not None:
        notes.append(
            "ATL/CTL/TSB computed from daily training_load_hours "
            "(EWMA time constants: ATL=7d, CTL=42d; TSB=CTL-ATL)."
        )
    if signals_used:
        notes.append(f"Signals contributing to this score: {', '.join(signals_used)}.")
    else:
        notes.append(
            "No individual signals had enough data to contribute; score reflects baseline uncertainty only."
        )

    # ---- Assemble inputs dict ----------------------------------------------
    def _r(v: Optional[float], n: int = 2) -> Optional[float]:
        return round(v, n) if v is not None else None

    inputs: dict[str, Any] = {
        "as_of_date": last_d.isoformat(),
        # RHR
        "rhr_7d_mean_bpm": _r(rhr7),
        "rhr_28d_mean_bpm": _r(rhr28),
        "rhr_28d_std_bpm": _r(rhr28_std),
        "rhr_delta_bpm": _r(delta_rhr),
        "rhr_z": _r(z_rhr, 3),
        # Load
        "training_load_7d_hours": _r(last7_load, 3),
        "training_load_rolling7_mean_hours": _r(base7_mean, 3),
        "training_load_rolling7_std_hours": _r(base7_std, 3),
        "training_load_delta_hours": _r(delta_load, 3),
        "training_load_z": _r(z_load, 3),
        "atl_load_hours": _r(atl_load, 3),
        "ctl_load_hours": _r(ctl_load, 3),
        "tsb_load_hours": _r(tsb_load, 3),
        # Sleep
        "sleep_7d_mean_hours": _r(sleep7),
        "sleep_28d_mean_hours": _r(sleep28),
        "sleep_28d_std_hours": _r(sleep28_std),
        "sleep_delta_hours": _r(delta_sleep),
        "sleep_z": _r(z_sleep, 3),
        # HRV
        "hrv_7d_mean_ms": _r(hrv7),
        "hrv_28d_mean_ms": _r(hrv28),
        "hrv_28d_std_ms": _r(hrv28_std),
        "hrv_delta_ms": _r(delta_hrv),
        "hrv_z": _r(z_hrv, 3),
        # Capability
        "recovery_capability_key": capability["key"],
        "recovery_capability_label": capability["label"],
        "sleep_days_7d": signal_counts_7d["sleep"],
        "resting_hr_days_7d": signal_counts_7d["resting_hr"],
        "hrv_days_7d": signal_counts_7d["hrv"],
        "notes": notes,
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


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def run_forecasts(
    input_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    cfg: Optional[ForecastConfig] = None,
) -> dict[str, Any]:
    """
    Used by: trailtraining forecast

    Loads combined_summary.json (and combined_rollups.json if present),
    runs compute_readiness_and_risk, validates the result against the
    ForecastArtifact contract, and writes readiness_and_risk_forecast.json.

    Returns {"saved": str(path), "result": dict}.
    """
    runtime = config.current()
    config.ensure_directories(runtime)

    base = (
        Path(input_dir).expanduser().resolve() if input_dir else runtime.paths.prompting_directory
    )
    summary_p = base / "combined_summary.json"
    if not summary_p.exists():
        raise FileNotFoundError(f"Missing required artifact: {summary_p}")

    combined = load_json(summary_p, default=[])
    if not isinstance(combined, list):
        raise ArtifactError(
            message="combined_summary.json must be a list",
            hint=f"Got {type(combined).__name__} instead.",
        )

    rollups: Optional[dict[str, Any]] = None
    rollups_p = base / "combined_rollups.json"
    if rollups_p.exists():
        raw = load_json(rollups_p, default=None)
        if isinstance(raw, dict):
            rollups = raw
        elif raw is not None:
            log.warning(
                "combined_rollups.json is a %s, not an object — ignoring.",
                type(raw).__name__,
            )

    resolved_cfg = cfg or ForecastConfig.from_env()
    fr = compute_readiness_and_risk(combined, rollups=rollups, cfg=resolved_cfg)

    outp = (
        Path(output_path).expanduser().resolve()
        if output_path
        else base / "readiness_and_risk_forecast.json"
    )

    payload = ForecastArtifact(
        generated_at=datetime.utcnow().isoformat() + "Z",
        result=ForecastResultArtifact(
            date=fr.date,
            readiness=ForecastReadiness(
                score=fr.readiness_score,
                status=normalize_readiness_status(fr.readiness_status),
            ),
            overreach_risk=ForecastRisk(
                score=fr.overreach_risk_score,
                level=normalize_risk_level(fr.overreach_risk_level),
            ),
            inputs=ForecastInputs.model_validate(fr.inputs),
            drivers=ForecastDrivers.model_validate(fr.drivers),
        ),
    )

    save_json(outp, payload.model_dump(mode="json"), compact=False)
    return {"saved": str(outp), "result": payload.model_dump(mode="json")}
