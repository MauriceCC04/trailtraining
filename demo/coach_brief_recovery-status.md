## Coach Brief — Recovery Status (as of 2026-03-10)

### Snapshot (Yesterday + last 7 days)
**Yesterday (2026-03-10):**
- Ride 48.63 km, **118 min**, 82 m+, avgHR 147 (moderate aerobic load)

**Last 7 days (2026-03-04..2026-03-10):**
- Load totals (rollups):
  - **Distance:** 182.816 km *(signal: `load.last7.distance_km`)*
  - **Moving time:** **10.175 h** *(signal: `load.last7.moving_time_hours`)*
  - **Elevation:** 1,931 m+ *(signal: `load.last7.elevation_m`)*
  - **Training load:** 22.666 “load_h” *(signal: `load.last7.training_load_hours`)*
  - **Sessions:** 7 *(signal: `load.last7.activity_count`)*
- Key load feature: one very large vertical trail run on 2026-03-07 (21.02 km, **1621 m+**, 196 min, avgHR 157) stacked into an otherwise high-frequency week (7 sessions / 7 days).
- Recovery data availability:
  - **Sleep days with data (last 7): 0** *(signal: `load.last7.sleep_days_with_data`)*
  - So we do **not** have last-7 sleep duration/HRV/RHR to confirm how well you’re absorbing the work *(signals: `recovery.last7.sleep_hours_mean`, `recovery.last7.hrv_mean`, `recovery.last7.rhr_mean` are null)*.
- Baseline context (28 days):
  - 28-day training load: 80.455 load_h *(signal: `load.baseline28.training_load_hours`)*
  - 28-day sleep mean: **8.7 h** on days with data *(signal: `recovery.last28.sleep_hours_mean`)*

### Readiness
**Status: fatigued** *(authoritative signal: `forecast.readiness.status`; score `forecast.readiness.score` = 49.1)*

**What’s driving it (from available data):**
- **Overreach risk is high** *(signals: `forecast.overreach_risk.level`, `forecast.overreach_risk.score` = 81.9)* and explicitly attributed to **elevated 7-day training load vs your recent rolling baseline** (per the deterministic forecast inputs/drivers).
- You’ve had **7 consecutive days with activities** (no clear rest day in the last-7 window) plus a **big vertical trail stimulus** mid-week—good fitness work, but it raises the odds that fatigue is accumulating.
- **Major limitation:** we can’t cross-check with physiology (sleep/HRV/RHR) because sleep is missing for the entire last week. With load elevated and recovery signals unavailable, the smart call is to treat today as **fatigue-management mode**.

### Recovery actions (today + tonight)
**Today (priority = absorb the work):**
- **Keep intensity low** (easy spinning/walk only) or take a full rest day if you feel any heaviness/residual soreness from the 3/7 trail run.
- **Mobility (10–15 min):** calves/ankles + hips + thoracic rotation; add **light eccentric calf work** only if legs feel good (no “digging a deeper hole”).
- **Fueling:** emphasize “recovery carbs + protein” across the day (especially within 2 hours of any session), and don’t under-eat after the high-volume week.

**Tonight (most important lever right now):**
- Aim to re-establish your baseline sleep pattern (your 28-day mean is **~8.7 h** on recorded days — *signal: `recovery.last28.sleep_hours_mean`*). Keep caffeine earlier, and front-load fluids/electrolytes so you’re not waking up dehydrated or to use the bathroom repeatedly.

### Warnings / flags
- **Fatigue stacking risk:** training frequency is high (7 sessions in 7 days), and the deterministic system flags **high overreach risk** *(signals: `forecast.overreach_risk.level`, `forecast.overreach_risk.score`)*.
- **Missing recovery telemetry:** 0 sleep-recorded days in the last 7 *(signal: `load.last7.sleep_days_with_data`)* means we’re flying blind on HRV/RHR and sleep duration right when load is flagged as elevated.

### Data notes (how this was assessed)
- Used rollups for last-7 and last-28 load totals *(signals under `load.last7.*` and `load.baseline28.*`)*.
- Sleep/HRV/RHR trends **cannot** be computed for the last 7 days because sleep is absent (null signals).
- **Discrepancy note:** the deterministic forecast references a higher 7-day training load in its internal inputs (32.125 load_h) than the rollup total (22.666 load_h). I treated the rollups as the “reported totals” for snapshot metrics, while still honoring the forecast’s **authoritative** readiness/overreach classification.
