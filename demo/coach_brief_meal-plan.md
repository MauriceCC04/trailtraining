## 7‑day meal plan (aligned to your current load + recovery picture)

### Load & recovery context (from your data)
- **Last 7 days load is high vs your baseline**:
  - `load.last7.training_load_hours` = **21.301** (vs rolling 7d mean 7.437; z=1.725 in forecast inputs)
  - `load.last7.moving_time_hours` = **9.533 h**, `load.last7.distance_km` = **181.909 km**, `load.last7.elevation_m` = **1949 m**
- **Readiness is “steady”**: `forecast.readiness.status` = **steady** (score `forecast.readiness.score` = 56.2)
- **Overreach risk is moderate**: `forecast.overreach_risk.level` = **moderate** (score 66.0), driven by elevated 7d load.
- **Sleep data is missing for the last 7 days**: `load.last7.sleep_days_with_data` = **0** (so we can’t tailor nutrition to recent sleep/HRV/RHR fluctuations—this plan leans recovery-supportive by default).

### How to use this plan
Because you didn’t provide an upcoming training schedule, I’m giving you a **7‑day template** matched to your current pattern (lots of riding + occasional long/vert trail).
Assign days as:
- **2 High‑carb days** (your longest/most intense sessions)
- **3 Moderate‑carb days** (typical 60–120 min aerobic)
- **1 Light day** (very easy/recovery)
- **1 Rest day** (no training)

If you swap your training days around, **swap the matching day’s meals** with it.

---

# Day-by-day plan (breakfast / lunch / dinner / snacks)
Portions: use a simple plate method since biometrics aren’t provided.
- **High‑carb day**: 1–2 fists carbs per meal + carb-focused snack(s)
- **Moderate**: ~1 fist carbs per meal
- **Light/Rest**: 0.5–1 fist carbs per meal (keep protein/veg strong)

---

## Day 1 — Moderate‑carb (typical aerobic day)
**Breakfast:** Greek yogurt bowl + oats/muesli + banana + honey + nuts
**Lunch:** Chicken (or tofu) rice bowl: rice + beans + salsa + avocado + veggies
**Dinner:** Salmon (or tempeh) + potatoes + big salad + olive oil
**Snacks:**
- Fruit + handful of trail mix
- Pre‑session (if needed): toast + jam
- Post‑session (within 60 min): chocolate milk or yogurt + fruit

**Macro emphasis:** Moderate carb, steady protein.

---

## Day 2 — High‑carb (your longest ride/trail/vert day)
**Breakfast (2–3 h pre):** Big oats: oats + milk + banana + raisins + maple + pinch of salt
**During session (if >90 min):**
- Aim **carbs each hour** (sports drink/gels/banana) + fluids (see hydration section)
**Lunch (post):** Large sandwich/wrap (turkey/egg/tofu) + extra bread/tortilla + fruit
**Dinner:** Pasta night: pasta + meat sauce (or lentil sauce) + veggies + parmesan
**Snacks:**
- Immediately post: smoothie (milk/alt + banana + berries) + whey/soy protein
- Later: cereal + milk, or rice cakes + nut butter + honey

**Macro emphasis:** High carb (this is where you “pay back” the work).

---

## Day 3 — Light‑carb (easy spin/jog or shorter recovery day)
**Breakfast:** 2–3 eggs (or tofu scramble) + toast + fruit
**Lunch:** Big salad + quinoa (smaller portion) + tuna/chickpeas + olive oil
**Dinner:** Stir-fry: lots of veg + protein + **smaller** rice portion
**Snacks:**
- Cottage cheese (or soy yogurt) + berries
- Dark chocolate + nuts (small)

**Macro emphasis:** Lighter carbs, high micronutrients; keep protein consistent.

---

## Day 4 — Moderate‑carb
**Breakfast:** Overnight oats + chia + berries
**Lunch:** Leftover stir-fry + rice (normal portion)
**Dinner:** Tacos: tortillas + beef/beans + cheese + salsa + slaw
**Snacks:**
- Pre: banana
- Post: yogurt drink + granola

**Macro emphasis:** Moderate carb.

---

## Day 5 — High‑carb (2nd key day: longer endurance or harder workout)
**Breakfast (pre):** Bagel + peanut butter + banana + yogurt
**During (if long):** carbs + electrolytes (see hydration)
**Lunch (post):** Rice + eggs (or tofu) + soy sauce + veggies + fruit
**Dinner:** Curry + rice: chicken/chickpeas + coconut curry + rice + naan (optional)
**Snacks:**
- Recovery: 3:1-ish carb:protein (e.g., smoothie + pretzels)
- Evening: cereal or oatmeal if hungry

**Macro emphasis:** High carb; prioritize recovery timing.

---

## Day 6 — Moderate‑carb (steady aerobic)
**Breakfast:** Pancakes/waffles + berries + yogurt (easy way to get carbs in)
**Lunch:** Lentil soup + bread + side salad
**Dinner:** Pizza (homemade or store) + big salad (don’t skip veg)
**Snacks:**
- Fruit + jerky (or edamame)
- Hummus + crackers

**Macro emphasis:** Moderate carb, higher total calories if you’re stacking volume.

---

## Day 7 — Rest day (lowest carb, highest “rebuild” focus)
**Breakfast:** Omelet (or tofu scramble) + veggies + small toast
**Lunch:** Protein-heavy bowl: greens + beans + chicken/tofu + seeds + olive oil; small grain portion
**Dinner:** Roast/air-fry: veggies + protein + small potato or bread
**Snacks:**
- Yogurt/cottage cheese + fruit
- Nuts + herbal tea

**Macro emphasis:** Lower carb, high protein, high produce, high hydration.

---

# Hydration guidance + timing (simple, athlete-friendly)
**Daily baseline:**
- Drink regularly through the day; include **electrolytes** if you’re training most days (you’ve had **6 activities/7d**: `load.last7.activity_count` = 6).

**Pre-session (60–120 min):**
- 500–700 ml water; add electrolytes if you tend to sweat a lot.

**During training:**
- **~500–750 ml/hour**, more if hot.
- If **>90 min**: add electrolytes + carbohydrate. For long days, aim for a steady carb drip (sports drink + gels/chews/banana).

**Post-session (first 60 min):**
- Fluids + a carb/protein hit (examples above).
- Then a full meal within ~2–3 hours.

---

# Recovery-supportive nutrition priorities (given moderate overreach risk)
Because load is elevated (`load.last7.training_load_hours` high vs baseline; `forecast.overreach_risk.level` = moderate) and sleep data is missing (`load.last7.sleep_days_with_data` = 0), default to:
- **Protein at every meal** (and a protein-rich snack on harder days).
- **Carbs concentrated around training** (especially on High‑carb days).
- **More colorful produce + fats** on Rest/Light days (anti-“empty calories” approach).
- **Evening routine**: finish the last big meal 2–3 h before bed; if hungry, choose an easy snack (yogurt + fruit, cereal + milk).

---

## Data notes / limitations
- No body weight/sex/age provided (`formatted_personal_data.json` empty), so portions are **general** (plate-based, not grams/kg).
- Last 7 days have **no sleep entries** (`load.last7.sleep_days_with_data` = 0), so I can’t adjust day-to-day fueling based on recent sleep duration/HRV/RHR trends; I used **load + forecast** signals (`forecast.readiness.status`, `forecast.overreach_risk.level`) to bias the plan toward recovery.

If you tell me what your *next 7 days of training* roughly look like (which days are long/hard/rest), I’ll map this template to specific dates and tighten the carb targets around each key session.
