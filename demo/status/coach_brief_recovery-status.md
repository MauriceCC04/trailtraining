## Coach Brief — Recovery Status (as of 2026-03-14)

### Snapshot (Yesterday + last 7 days)
**Yesterday (2026-03-13)**
- Ride: **51.12 km**, **107 min**, **120 m+**, avgHR **159** (“Lunch Ride”)

**Last 7 days (2026-03-07..2026-03-13) — load**
- Volume: **233.024 km** *(signal: `load.last7.distance_km`)*
- Moving time: **11.324 h** *(signal: `load.last7.moving_time_hours`)*
- Elevation: **2069 m** *(signal: `load.last7.elevation_m`)*
- Training load: **25.394 load_h** *(signal: `load.last7.training_load_hours`)*
- Sessions: **7** *(signal: `load.last7.activity_count`)*
  Mix: **6 Rides + 1 TrailRun** (the TrailRun contributed **1621 m+** on 2026-03-07)

**Baseline context (last 28 days: 2026-02-14..2026-03-13)**
- Training load: **82.268 load_h** *(signal: `load.baseline28.training_load_hours`)*
  → Your **last-7 load is elevated vs your recent rolling baseline**, aligning with the forecast driver and overreach flag.

**Recovery data availability**
- Sleep/HRV/RHR: **no sleep data at all in the last 7 days** *(signal: `load.last7.sleep_days_with_data` = 0; signals `recovery.last7.*` are null)*
- 28-day sleep mean is **8.67 h** when data exists *(signal: `recovery.last28.sleep_hours_mean`),* but we **can’t verify** whether you’re currently matching that.

---

### Readiness: **steady** *(authoritative; signal: `forecast.readiness.status`)*
**Why “steady” (what’s driving it)**
- **Load is up vs baseline/rolling norm**, which increases the likelihood you’re carrying fatigue even if you feel okay:
  - Last 7 training load **25.394 load_h** *(signal: `load.last7.training_load_hours`)* vs 28d context **82.268 load_h** *(signal: `load.baseline28.training_load_hours`)* and the deterministic model notes the 7d load is elevated.
- **Overreach risk = moderate** *(signals: `forecast.overreach_risk.level`, `forecast.overreach_risk.score` = 66.0)*.
- **Key limiter:** recovery signals are **missing** right when we need them most (sleep days = 0), so we can’t confirm you’re absorbing the work well via sleep duration, HRV, or resting HR trends.

Net: fitness work is happening, but you’re in a “trust but verify” zone—steady readiness with moderate overreach risk due to load and missing recovery telemetry.

---

### Today’s Plan (recovery-forward; keeps you inside constraints)
Goal: **insert a true low-stress day** to reduce stacking and create at least one rest/very-easy day in the current 7-day chunk.

**Option A (preferred): Rest day**
- **0:00 training**, short walk only if you want.

**Option B: Easy flush (if you’re restless)**
- **30–60 min easy spin** (nose-breathing / conversational, very light resistance), flat route.
- **is_hard_day = false** (no tempo, no hills, no “see how legs feel” surges).

This helps address that you’ve effectively had **7 sessions in the last 7 days** (no obvious rest day) while load is elevated.

---

### Recovery actions (today/tonight)
**Sleep**
- Make tonight a priority: **aim for a full, unbroken night** (use your 28d average of **~8.67 h** as a reference target; signal: `recovery.last28.sleep_hours_mean`).
- If you can, get back to recording sleep—your readiness accuracy improves dramatically once we have 3–5 recent nights.

**Fueling / hydration**
- Because load has been high *(signal: `load.last7.training_load_hours`)*:
  - Eat to replenish today even if resting: **protein across 3–4 meals** + **carbs with meals** (especially if training resumes tomorrow).
  - Hydrate steadily; include electrolytes if you’ve been doing high-HR midday rides.

**Mobility / tissue work (10–15 min)**
- Ankles/calves, hips, gentle thoracic mobility.
- Light leg flush (easy foam roll) only—no aggressive deep work if you’re feeling beat up.

---

### Risks & Flags
- **Moderate overreach risk** with elevated 7-day training load *(signals: `forecast.overreach_risk.level`, `load.last7.training_load_hours`)*.
- **No recovery telemetry in the last 7 days** *(signal: `load.last7.sleep_days_with_data` = 0)* → we’re “flying blind” on HRV/RHR/sleep trends.
- The 2026-03-07 trail run title suggests you may have been managing discomfort; treat that as a cue to **avoid intensity** until you’ve had at least one easy/rest day plus normal-feeling legs.

---

### Data notes
- Used **rollups (authoritative)** for 7d/28d totals and load comparison.
- **Sleep metrics** (sleepTimeSeconds/HRV/RHR) were **not available for 2026-03-07..2026-03-13**, so no 7-day recovery trend could be computed *(signals `recovery.last7.*` null; `load.last7.sleep_days_with_data` = 0)*.
- Activity intensity inference is limited: we have avgHR on multiple rides, but without consistent maxHR and without workout structure, I’m not labeling any day as a “hard day” unless you tell me it was intervals/tempo/hill reps.

If you want, share whether yesterday’s and 3/12’s rides (avgHR 159–164) were steady endurance or hard efforts—this changes how aggressively we prioritize rest vs easy volume tomorrow.
