## Coach Brief — recovery-status (as of 2026-03-12)

### Snapshot (Yesterday + last 7 days)
**Yesterday (2026-03-12)**
- **Ride** 51.44 km, **112 min**, 109 m+, **avgHR 164** (Lunch Ride)

**Last 7 days (2026-03-06..2026-03-12) — load**
- **6 sessions** *(signal: `load.last7.activity_count` = 6)*
- **9.53 h moving time** *(signal: `load.last7.moving_time_hours` = 9.533 h)*
- **181.9 km** *(signal: `load.last7.distance_km` = 181.909 km)*
- **1949 m climbing** *(signal: `load.last7.elevation_m` = 1949 m)*
- **Training load = 21.30 load-h** *(signal: `load.last7.training_load_hours` = 21.301 load_h)*
  - Mix: **5 Rides + 1 TrailRun** (the TrailRun accounts for most of the climbing)

**Recovery data availability (last 7 days)**
- **No sleep metrics captured** *(signal: `load.last7.sleep_days_with_data` = 0; `recovery.last7.sleep_hours_mean` = null; `recovery.last7.hrv_mean` = null; `recovery.last7.rhr_mean` = null)*
- Baseline context: last 28 days shows **sleep mean ~8.7 h when data exists** *(signal: `recovery.last28.sleep_hours_mean` = 8.7 h; 17 days with sleep data in 28d rollup)*

---

### Readiness
**Status: steady** *(authoritative signal: `forecast.readiness.status` = steady; score `forecast.readiness.score` = 56.2)*

**Why “steady” (what’s driving it)**
- **Load is elevated vs your recent rolling baseline**, which increases recovery demand *(signals: `load.last7.training_load_hours` = 21.301; deterministic driver notes + overreach risk below)*.
- **We can’t confirm recovery is keeping up** because there’s **zero sleep/HRV/RHR data in the last 7 days** *(signals: `load.last7.sleep_days_with_data` = 0; `recovery.last7.*` all null)*.
- **Overreach risk is moderate** right now *(signals: `forecast.overreach_risk.level` = moderate; `forecast.overreach_risk.score` = 66.0)*—this aligns with a week containing both higher cardiovascular strain rides (e.g., yesterday avgHR 164) and a big-vert trail run earlier in the week.

---

### Recovery actions for today/tonight (priority order)
1. **Sleep (highest leverage)**
   - Aim for a **full night** consistent with your recent baseline pattern (your 28-day mean when recorded is **~8.7 h**, signal: `recovery.last28.sleep_hours_mean`).
   - If schedule allows: **20–30 min nap** or quiet lying-down reset to reduce accumulated stress from the elevated 7-day load.

2. **Fueling**
   - Given the elevated 7-day training load *(signal: `load.last7.training_load_hours` = 21.301)*, prioritize **carb + protein** across the day:
     - After training (or as a general recovery meal): **carb-forward meal + 25–40 g protein**.
   - If training again today: **carbs before + during**, even for rides, to avoid digging a deeper hole.

3. **Hydration**
   - Replace fluids proactively (especially after higher-HR rides like yesterday). Add **electrolytes** if you’re sweating heavily or riding indoors.

4. **Mobility / tissue work (10–20 min)**
   - **Calves/ankles + hips + T-spine** (trail + cycling posture combo).
   - Keep it easy: the goal is to **downshift**, not create more soreness.

---

### Warnings / flags
- **Moderate overreach risk** *(signals: `forecast.overreach_risk.level` = moderate; `forecast.overreach_risk.score` = 66.0)*: with load elevated vs your recent rolling baseline, watch for “hidden fatigue” signs (heavy legs, unusually high effort for normal pace/power, irritability, poor sleep).
- **No recovery telemetry for 7 straight days** *(signal: `load.last7.sleep_days_with_data` = 0)*: that makes it easy to accidentally stack fatigue because we can’t validate whether sleep/HRV/RHR are trending well.
- Subjective note worth respecting: your 3/7 trail run title suggests you were managing pain (“Ibuprofen…”). Without more data I can’t assess it, but it’s a **risk flag**: avoid forcing intensity if anything feels structurally “sharp” or worsening.

---

### Data notes (what I used / what’s missing)
- Used **combined_rollups windows["7"] vs windows["28"]** as authoritative for load totals and baseline comparisons.
- Treated sleep metrics as **missing** for the last 7 days because **sleep days with data = 0** *(signal: `load.last7.sleep_days_with_data`)*; therefore **no HRV/RHR/sleep trend** can be computed for the current week *(signals: `recovery.last7.sleep_hours_mean`, `recovery.last7.hrv_mean`, `recovery.last7.rhr_mean` are null)*.
- Readiness status is set to **deterministic forecast**: *(signal: `forecast.readiness.status` = steady)*.

If you want, share whether today is a planned training day or a recovery day—then I’ll translate this recovery status into a specific session recommendation that respects your ramp and hard-day constraints.
