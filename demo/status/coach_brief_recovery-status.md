## Coach Brief — Recovery Status (as of 2026-03-14)

### Snapshot (Yesterday + last 7 days)
**Yesterday (2026-03-13)**
- **Ride:** 51.12 km, **107 min**, 120 m+, **avgHR 159** (moderately hard aerobic day)

**Last 7 days (2026-03-07..2026-03-13) — current load**
- **Volume/Load:** 233.0 km, **11.32 h moving**, 2069 m+, **25.39 training_load_h**, **7 activities**
  (signals: `load.last7.distance_km`, `load.last7.moving_time_hours`, `load.last7.elevation_m`, `load.last7.training_load_hours`, `load.last7.activity_count`)
- **Sport mix:** 6 rides + 1 big trail run (21.0 km / **1621 m+** / 3h16)
- **Recovery data availability:** **0 sleep nights with data** in the last 7 days (signal: `load.last7.sleep_days_with_data`)

**Baseline (last 28 days)**
- 82.27 training_load_h over 28d (signal: `load.baseline28.training_load_hours`) ⇒ ~**20.6 training_load_h per week** on average
- Sleep baseline when available: **8.67 h mean** over last 28d (signal: `recovery.last28.sleep_hours_mean`)

**What stands out**
- Your **last-7 load is elevated vs your 28-day “typical week”** (25.39 vs ~20.6 training_load_h/week), and it’s been **dense** (7 sessions in 7 days) without an obvious unload day.
- The last 2 days show **higher avgHR rides (159, 164)**, which often adds more stress than the same duration at lower HR.

---

### Readiness
**Status: `steady`** (authoritative; signal: `forecast.readiness.status`, score `forecast.readiness.score` = 56.2)

**Rationale (data-based)**
- Deterministic model flags **moderate overreach risk** (signals: `forecast.overreach_risk.level` = moderate, `forecast.overreach_risk.score` = 66.0), driven by **7d training load elevated vs recent rolling baseline** (as provided in the deterministic drivers).
- **We cannot confirm recovery** via sleep/HRV/RHR trends this week because **sleep data is missing** (signals: `recovery.last7.sleep_hours_mean`, `recovery.last7.hrv_mean`, `recovery.last7.rhr_mean` are null). So “steady” here is essentially: *load is up, but we lack physiological recovery markers to upgrade/downgrade confidently.*

---

### Recovery actions for today/tonight (to reduce overreach risk)
**1) Training choice today (recovery-forward)**
- Make today a **rest day** *or* **30–60 min very easy (Z1) spin/walk**, flat, nose-breathing easy.
- Purpose: **absorb** the last 7 days and reduce the chance that the next quality/long day becomes a “forced” off day.

**2) Mobility / tissue care (10–20 min)**
- Easy hips + calves + ankles sequence, plus light thoracic rotation.
- If you’re carrying soreness from the 1621 m+ trail run, keep this **gentle**, not aggressive stretching.

**3) Fueling & hydration (simple targets)**
- **Today:** prioritize regular meals with **protein at each meal** (recovery/repair) and **carbs** proportional to whether you train.
- **If you do any session:** take carbs during if >60 min, and get a carb+protein meal within ~2 hours after.
- Hydration: aim for **clear/pale urine by mid-day**, and add electrolytes if you’re training or it’s warm.

**4) Sleep**
- You have a strong baseline when it’s tracked (28d mean **8.67 h**, signal: `recovery.last28.sleep_hours_mean`). Use that as your target: **plan for ~8.5–9 h time-in-bed tonight**.
- Because we’re missing sleep telemetry this week, be extra strict with sleep routine (same bedtime, dark/cool room).

---

### Warnings / flags
- **Load density:** 7 sessions in 7 days (signal: `load.last7.activity_count`), plus a high-vert trail run—this is a classic setup for “quiet fatigue” even if you feel okay.
- **Moderate overreach risk is already flagged** (signals: `forecast.overreach_risk.level`, `forecast.overreach_risk.score`). The safest move is to **insert a true easy/rest day now**.
- **Recovery-data blind spot:** last 7 days has **no sleep/RHR/HRV data**, so if motivation drops, legs feel flat/heavy, or HR is unusually high for easy effort, treat that as actionable feedback and back off.
- **Soft tissue/injury note:** the trail run title mentions pain management (“Ibuprofen…”). Not a diagnosis—just a flag to be conservative with intensity and downhill pounding until you’re consistently pain-free.

---

### Data notes (what I used / what’s missing)
- Load comes from rollups (authoritative): `load.last7.*` vs `load.baseline28.*`.
- Recovery signals **could not be trended** this week: `recovery.last7.sleep_hours_mean`, `recovery.last7.hrv_mean`, `recovery.last7.rhr_mean` are null; rollup shows `load.last7.sleep_days_with_data` = 0.
- Minor mismatch: the deterministic readiness inputs referenced **training_load_7d_hours = 21.301 as of 2026-03-12**, while the current 7-day rollup ending 2026-03-13 reports **25.394**—this is expected due to different “as-of” dates/windows rather than an error.

If you want, share how your legs feel today (fresh/flat/sore) and whether you plan to run or ride—without sleep telemetry this week, your subjective check-in is the best tie-breaker for how conservative we should be.
