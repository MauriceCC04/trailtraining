## 7-day meal plan (2026-03-14 → 2026-03-20) built off your current load & recovery signals

**Load context:** Last 7d was **11.324 h moving time** and **25.394 training-load hours** with **7 sessions** (mostly rides + 1 big vert trail run) — elevated vs your recent rolling baseline and paired with **moderate overreach risk** and **readiness = steady** (signals: `load.last7.moving_time_hours`, `load.last7.training_load_hours`, `load.last7.activity_count`, `forecast.readiness.status`, `forecast.overreach_risk.level`).
**Recovery context:** **No sleep data recorded in the last 7d** (signals: `load.last7.sleep_days_with_data`, `recovery.last7.sleep_hours_mean`), so we’ll bias nutrition toward consistent energy availability, high protein distribution, and hydration/electrolytes to support recovery even when sleep metrics aren’t visible.

### Macro emphasis key (per day)
- **Higher-carb day:** best for long sessions / intensity / big climbing.
- **Moderate-carb day:** best for typical 60–120 min aerobic rides/runs.
- **Lighter day:** best for rest or very easy short sessions (still prioritize protein + micronutrients).

> If your actual training differs on a given day: **match the day’s macro emphasis to your session** (higher-carb for longer/harder; lighter for rest).

---

## Day 1 — Sat 2026-03-14 (Moderate-carb; “reset/rebuild” day)
**Breakfast:**
- Oats cooked in milk (or soy) + banana/berries + chia
- 2 eggs (or tofu scramble) for extra protein

**Snack (mid-morning):**
- Greek yogurt + honey + granola *or* smoothie (milk/soy + fruit + oats)

**Lunch:**
- Rice bowl: rice + chicken/tempeh + roasted veg + olive oil + salsa/beans

**Snack (mid-afternoon):**
- Bagel or toast + peanut butter + fruit

**Dinner:**
- Salmon (or lentils) + potatoes + big salad (olive oil + vinegar)
- Optional bread if you feel flat

**Before bed (optional):**
- Cottage cheese / yogurt + berries (easy protein to support overnight recovery)

---

## Day 2 — Sun 2026-03-15 (Higher-carb; “long/quality session” support)
**Pre-session (60–120 min before):**
- Bagel + jam + yogurt *or* oatmeal + banana
- Coffee/tea if you use it

**During training (if >90 min):**
- Aim **30–60 g carbs/hour** (drink mix + gels/chews/banana)
- Add electrolytes (see hydration section)

**Post-session (within ~60 min):**
- Chocolate milk / recovery shake + banana
- Then a real meal within 2 hours

**Lunch (post):**
- Burrito/bowl: tortillas or rice + beans + chicken/tofu + guac + veg

**Snack:**
- Pretzels + hummus *or* cereal + milk

**Dinner:**
- Pasta (or couscous) + lean meat/soy + tomato sauce + spinach
- Fruit for dessert

---

## Day 3 — Mon 2026-03-16 (Moderate-carb; “steady aerobic”)
**Breakfast:**
- Overnight oats + berries + nuts *or* avocado toast + eggs + fruit

**Snack:**
- Apple + cheese *or* trail mix

**Lunch:**
- Sandwich/wrap: turkey/tuna/tofu + cheese + lots of veg
- Side: yogurt or fruit

**Snack (pre-ride if training later):**
- Granola bar + banana

**Dinner:**
- Stir-fry: rice + veggies + shrimp/chicken/tofu (easy to scale portions)

---

## Day 4 — Tue 2026-03-17 (Lighter day; “rest / very easy”)
**Breakfast:**
- Omelet/tofu scramble + veggies + toast (1 slice)
- Fruit

**Snack:**
- Carrots + hummus *or* yogurt

**Lunch:**
- Big salad + quinoa + beans + feta (or chicken) + olive oil dressing
- Whole-grain roll if needed

**Snack:**
- Nuts + fruit

**Dinner:**
- Chili (beans + veg + lean meat/soy)
- Small serving rice or cornbread if hungry

**Notes:** Keep carbs present but not the centerpiece; still hit protein well.

---

## Day 5 — Wed 2026-03-18 (Higher-carb; “key workout” support)
**Breakfast (or pre-session meal):**
- Oatmeal + banana + honey + milk/soy
- Add 1 scoop protein in smoothie or a side of eggs if breakfast-only

**During training (if >75–90 min):**
- **30–60 g carbs/hour**, fluids + electrolytes

**Post-session:**
- Rice cakes + jam + protein shake *or* yogurt + cereal + fruit

**Lunch:**
- Noodle soup + extra rice + chicken/tofu + veg (easy on the gut)

**Snack:**
- Bagel + cream cheese *or* granola + yogurt

**Dinner:**
- Rice or potatoes + lean protein + cooked veg
- Optional dessert: fruit + yogurt (good if appetite is high)

---

## Day 6 — Thu 2026-03-19 (Moderate-carb; “consistency” day)
**Breakfast:**
- Pancakes/waffles (or oats) + fruit + yogurt

**Snack:**
- Banana + nuts

**Lunch:**
- Mediterranean plate: pita + hummus + chicken/falafel + couscous + salad

**Snack:**
- Protein bar + fruit

**Dinner:**
- Tacos: tortillas + beans + meat/tofu + salsa + cheese + slaw

---

## Day 7 — Fri 2026-03-20 (Moderate → Higher-carb if you go long on weekend)
**Breakfast:**
- Oats + fruit + yogurt *or* breakfast sandwich + fruit

**Lunch:**
- Rice bowl or pasta salad with plenty of carbs + protein

**Snack (prep for weekend):**
- Pretzels + sports drink *or* bagel + honey
- Hydrate well (see below)

**Dinner:**
- “Carb-forward” plate if a long session is coming: pasta/rice + protein + veg
- Keep fiber moderate if you train early next morning

---

# Hydration & timing (daily + training)
**Daily baseline:**
- Start the day with **~500 ml water**.
- With meals: drink enough that urine is *generally pale yellow*.

**Training hydration (especially given your recent steady-high load; `load.last7.training_load_hours`):**
- For most sessions: **500–750 ml fluid/hour** (more if hot/sweaty).
- **Electrolytes:** target **~300–600 mg sodium/hour** during longer rides/runs (adjust to sweat rate and conditions).
- **Post-session:** 500 ml water + electrolytes if you finished salty/white-crusted or with high sweat loss.

**Recovery rhythm (important since last-7 sleep data is missing; `load.last7.sleep_days_with_data`):**
- Don’t “save” calories for night—fuel earlier to avoid late-day hunger spikes.
- Include a **protein-containing bedtime snack** on higher-load days if you’re consistently hungry at night.

---

# Simple portion guidance (no biometrics available)
Because we don’t have weight/sex/age data in your provided profile JSON, keep portions adjustable:
- **Protein:** include a solid serving **at each meal** (roughly 25–40 g/meal for many athletes).
- **Carbs:** scale by day type
  - Higher-carb days: carbs at **every** meal + during session
  - Moderate days: carbs at 3–4 touchpoints/day
  - Lighter/rest: carbs mainly around breakfast/lunch, keep dinner balanced
- **Fats:** keep moderate; slightly lower right before/after hard sessions (easier digestion).

---

## Data notes (what I used / what’s missing)
- Used rollups to classify load: `load.last7.moving_time_hours`, `load.last7.training_load_hours`, and baseline comparison via `load.baseline28.training_load_hours`.
- Readiness is set from your authoritative forecast: `forecast.readiness.status = steady`, with `forecast.overreach_risk.level = moderate`.
- Sleep guidance is conservative because **last 7 days contain no sleep entries** (`load.last7.sleep_days_with_data = 0`, `recovery.last7.sleep_hours_mean = null`). Baseline sleep over 28d shows **~8.67 h mean** (`recovery.last28.sleep_hours_mean`), but we can’t confirm the most recent week.

If you tell me what your next 7 days of training *actually* look like (long day(s), rest day, any intensity), I’ll re-label each day’s macro emphasis precisely to the sessions rather than using the “typical week” structure above.
