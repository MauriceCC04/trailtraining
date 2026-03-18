# trailtraining

![CI](https://github.com/MauriceCC04/trailtraining/actions/workflows/ci.yml/badge.svg)

A local-first Python CLI that turns Strava/Garmin training data into auditable coaching artifacts — structured plans, deterministic evaluation, and iterative revision.

Most AI coaching features are generic and unverifiable. This project is an attempt to build something better: every output is grounded in local data, checked against explicit constraints, and revisable from its own evaluation report.

→ **No API keys needed to see what it produces.** Check out [`demo/`](demo) and [`docs/engineering.md`](docs/engineering.md).

---

## How it works

```
Strava / Garmin / Intervals.icu
        │
        ▼
   local ingestion
        │
        ▼
  combine → combined_rollups.json + formatted_personal_data.json
        │
        ▼
  deterministic forecast (readiness + overreach risk)
        │
        ▼
  coach --prompt training-plan [--plan-days 7|14|21|28]
        │
        ├──► coach_brief_training-plan.json / .txt
        │
        ▼
  eval-coach [--soft-eval] [--soft-eval-runs N] [--skip-synthesis]
        │
        ├──► eval_report.json
        │
        ▼
  revise-plan [--auto-reeval]
        │
        ├──► revised-plan.json / .txt
        ├──► revised-plan-reeval.json     ← delta score (with --auto-reeval)
        │
        ▼
  plan-to-ics  ──► training-plan.ics → Calendar.app
```

The pipeline runs in one direction: ingest → forecast → generate → evaluate → revise → export. Generated plans are treated as first drafts, not final answers.

---

## Engineering decisions worth noting

**Deterministic constraints before generation.** Ramp rate, hard-day spacing, and rest structure are enforced mathematically on the output — not left to the model's judgment. The model sees the constraints in its context; if it still violates them, guardrails correct in-place.

**Two-stage evaluation.** `eval-coach` runs deterministic checks (ramp %, consecutive hard days, citation coverage). With `--soft-eval`, a second model acts as an independent judge using a rubric-driven schema — strengths, concerns, marker-level evidence. The soft evaluator should be a different model family than the generator to avoid shared blind spots.

**Per-rubric batch evaluation.** The soft evaluator runs one LLM call per rubric group rather than one monolithic call across all 18+ markers. This eliminates inter-marker anchoring bias — early marker scores no longer pull later ones toward them. `goal_alignment` and `plan_coherence` are batched together (load and goal logic are related); `caution_proportionality` is isolated because it is the hardest to score consistently. Rubric scores are always derived from marker averages, never asked for separately.

**Parallel batch execution.** Rubric batch calls run concurrently by default via a thread pool, cutting soft-eval wall-clock time roughly in half. Disable with `--no-parallel-batches` for debugging or tight rate limits.

**Synthesis skip for fast iteration.** `--skip-synthesis` skips the final narrative LLM call during soft evaluation. Strengths, concerns, and improvements are derived locally from marker scores instead. This saves one LLM call per eval cycle when you only need scores — for example, when piping directly into `revise-plan`.

**Observation-before-score.** Each marker result requires an `observation` field — a plain-language description of what the model sees in the plan — before it may assign a score. This makes scoring decisions auditable and catches lazy scoring where a model would anchor on a number without reading the plan.

**Revision is part of the pipeline.** `revise-plan` takes the original plan and its eval report and produces a revised artifact. With `--auto-reeval`, the revised plan is immediately re-evaluated against the deterministic constraint engine, a `delta_score` is computed, and a warning is printed if the revision made things worse. Generate → critique → revise → re-check is the intended loop, not a nice-to-have.

**Inter-rater reliability measurement.** `--soft-eval-runs N` runs the soft evaluator N times with temperature > 0 and reports per-marker score variance. Markers with standard deviation above 0.5 on a 1–5 scale are flagged in the report — high variance signals a rubric definition that is too ambiguous to score consistently and may need tightening.

**Lifestyle constraints separated from race goals.** Schedule and access constraints (e.g. "weekdays road only, one mountain day on weekends") are stored in `TRAILTRAINING_LIFESTYLE_NOTES` separately from `TRAILTRAINING_PRIMARY_GOAL`. The coach sees both during generation and plans accordingly. The soft evaluator also sees the constraints, so it scores the plan against the best plan possible *given* those constraints — not against an unconstrained ideal. Without this separation, a plan perfectly designed for your life could score 70–80 because the evaluator penalizes weekday road runs as "not trail-specific enough".

**Structured contracts throughout.** Artifacts are validated with strict Pydantic models. The LLM is prompted with a JSON schema and the output is validated on the way out — malformed responses trigger a repair pass before anything is saved.

**Graceful degradation.** The pipeline runs on activity-only data and improves when sleep, HRV, or resting HR are available. Missing recovery telemetry is surfaced explicitly in the output rather than silently omitted.

---

## Setup

**Requirements:** Python 3.9+, a Strava API application, one wellness source (GarminDB or Intervals.icu), an OpenRouter API key.

```bash
git clone https://github.com/MauriceCC04/trailtraining.git
cd trailtraining
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Profiles load from `~/.trailtraining/profiles/<profile>.env`:

```bash
STRAVA_CLIENT_ID="..."
STRAVA_CLIENT_SECRET="..."
STRAVA_REDIRECT_URI="http://127.0.0.1:5000/authorization"

# Intervals.icu (or use Garmin equivalents)
INTERVALS_API_KEY="..."
INTERVALS_ATHLETE_ID="0"

OPENROUTER_API_KEY="sk-or-v1-..."
TRAILTRAINING_LLM_MODEL="openai/gpt-4.1-mini"
TRAILTRAINING_SOFT_EVAL_MODEL="anthropic/claude-haiku-4-5"

# Coaching preferences
TRAILTRAINING_PRIMARY_GOAL="21k trail race, 1500m elevation on September 15 2026"
TRAILTRAINING_LIFESTYLE_NOTES="Weekdays: road runs or cycling only. One long mountain day on Saturday or Sunday."
TRAILTRAINING_PLAN_DAYS="7"   # 7, 14, 21, or 28
```

**`TRAILTRAINING_PRIMARY_GOAL`** accepts any free-form description of your target. If it contains a recognisable date (e.g. `July 30 2026`, `2026-09-12`, `in April`), the CLI automatically computes weeks-to-race and injects a recommended training phase (base / build / peak / taper) into every generated plan.

**`TRAILTRAINING_LIFESTYLE_NOTES`** accepts schedule or access constraints that affect session placement. These are stored in the plan artifact (`meta.lifestyle_notes`), injected into the generation prompt so the coach respects them, and passed to the soft evaluator so it doesn't penalize constrained choices. Keep this separate from your race goal — the goal describes *what* you're training for, the lifestyle notes describe *when and where* you can train.

---

## Model selection

The pipeline uses two models: one for generation (`TRAILTRAINING_LLM_MODEL`) and one for evaluation (`TRAILTRAINING_SOFT_EVAL_MODEL`). These should be different model families to avoid the evaluator sharing the generator's blind spots.

As of March 2026:
**Generation model** — must support structured JSON output via OpenRouter's Responses API. In practice, this means **OpenAI models only** (`openai/gpt-4.1-mini`, `openai/gpt-5.4`, etc.). Non-OpenAI models (Gemini, Claude, DeepSeek) fall back to unstructured text through OpenRouter and produce malformed schemas that require expensive repair passes — or fail entirely on 28-day plans.

**Evaluation model** — less constrained because eval batch schemas are small and simple. Claude Haiku 4.5 (`anthropic/claude-haiku-4.5`) works reliably and is fast/cheap.

**Recommended configurations:**

| Config | Generator             | Evaluator | Cost/cycle | Notes |
|---|-----------------------|---|-|---|
| **Reliable + cheap** | `openai/gpt-5.4-mini` | `anthropic/claude-haiku-4.5` | Best default. Structured output works on first call. |
| **Highest quality** | `openai/gpt-5.4`      | `anthropic/claude-haiku-4.5` | Better plan quality, same eval. |
| **Fast iteration** | `openai/gpt-5.4-mini` | `anthropic/claude-haiku-4.5` + `--skip-synthesis` | Skips narrative synthesis LLM call. |

---

## Typical workflow

```bash
trailtraining --profile alice doctor          # check setup
trailtraining --profile alice auth-strava     # OAuth flow
trailtraining --profile alice run-all         # ingest + combine
trailtraining --profile alice forecast        # readiness + overreach risk
trailtraining --profile alice coach --prompt training-plan
trailtraining --profile alice eval-coach --soft-eval
trailtraining --profile alice revise-plan --auto-reeval
trailtraining --profile alice plan-to-ics     # export to calendar
```

Other prompts: `recovery-status`, `meal-plan`, `session-review`.

### 28-day plan

```bash
trailtraining --profile alice coach --prompt training-plan --plan-days 28
# or set TRAILTRAINING_PLAN_DAYS=28 in your .env to make it the default
```

The plan is split into phased training weeks (build → build → peak → recovery). Hard-day and rest-day constraints are enforced per rolling 7-day window across all weeks. `weekly_totals` in the artifact reflects week 1 so ramp-rate validation remains accurate.

### Soft evaluation with inter-rater reliability

```bash
# Single evaluation pass (default, parallel batches)
trailtraining --profile alice eval-coach --soft-eval

# Run 2 times and flag markers with high score variance
trailtraining --profile alice eval-coach --soft-eval --soft-eval-runs 2

# Fast mode: skip synthesis, just get scores
trailtraining --profile alice eval-coach --soft-eval --skip-synthesis

# Debug mode: sequential batches
trailtraining --profile alice eval-coach --soft-eval --no-parallel-batches
```

With `--soft-eval-runs N`, the evaluator runs N independent passes with temperature > 0. The report includes per-marker score variance; any marker with std > 0.5 on a 1–5 scale is printed as a warning. High variance means the rubric definition is ambiguous — it is useful for calibrating rubrics during development or before deploying a new prompt to production.

### Revision with automatic re-evaluation

```bash
# Revise and immediately check whether the revision helped
trailtraining --profile alice revise-plan --auto-reeval
```

With `--auto-reeval`, the revised plan is re-evaluated straight away against the deterministic constraint engine. A delta report is written to `revised-plan-reeval.json` with `original_score`, `revised_score`, `delta_score`, and any remaining violations. If the delta is negative the CLI prints a warning so you know to inspect the report before accepting the revision.

### Lifestyle constraints

```bash
# Via .env (recommended)
TRAILTRAINING_LIFESTYLE_NOTES="Weekdays: road runs or cycling only. One long mountain day on Saturday or Sunday."

# Or per-command override
trailtraining --profile alice coach --prompt training-plan \
  --lifestyle-notes "Weekdays: road only. Weekend mountain."
```

The `--lifestyle-notes` flag is available on `coach`, `eval-coach`, and `revise-plan`. The notes are recorded in the plan artifact at `meta.lifestyle_notes` and carried through the full pipeline. The soft evaluator uses them to avoid penalizing schedule-constrained choices — for example, it won't flag weekday road runs under `trail_specificity` or cycling sessions under `non_competing_focus` when those are forced by the athlete's schedule.

### Calendar export

```bash
trailtraining --profile alice plan-to-ics            # reads the most recent plan, opens Calendar.app
trailtraining --profile alice plan-to-ics --no-open  # write .ics only, skip Calendar.app
trailtraining --profile alice plan-to-ics --start-hour 6 --output ~/Desktop/plan.ics
```

`plan-to-ics` picks the most recently modified plan (`revised-plan.json` or `coach_brief_training-plan.json`), converts it to a standards-compliant `.ics` file, and on macOS opens it with Calendar.app so you can accept the import. Training sessions become timed events (default 07:00); rest days become all-day events.

---

## Output layout

```
~/trailtraining-data/<profile>/prompting/
├── combined_rollups.json
├── formatted_personal_data.json
├── readiness_and_risk_forecast.json
├── coach_brief_training-plan.json / .txt
├── eval_report.json
├── revised-plan.json / .txt
├── revised-plan-reeval.json               ← delta score (--auto-reeval only)
├── training-plan.ics                      ← calendar export
└── coach_brief_<recovery-status|meal-plan|session-review>.md
```

---

## Soft evaluation rubrics

The soft evaluator grades plans across five rubrics (weights for trail running shown):

| Rubric | Weight | What it checks |
|---|---|---|
| `goal_alignment` | 30% | Plan targets trail running specifically, not generic fitness |
| `plan_coherence` | 25% | Hard/easy spacing, session type matches purpose, totals arithmetic, week-level fatigue logic |
| `explanation_quality` | 20% | Reasoning is specific and non-generic, day purposes are useful |
| `caution_proportionality` | 15% | Warnings match context; missing data is acknowledged and acted on |
| `actionability` | 10% | Sessions are concrete enough to execute day-to-day |

Each rubric is scored independently in its own LLM call (4 batch calls total, run in parallel by default). Rubric scores are derived from the average of their marker scores — they are never asked for directly, which prevents the model from anchoring a rubric score and then reverse-engineering marker evidence to match it.

Key markers include `week_coherence` (scored last within `plan_coherence`, after all per-session markers are filled), `weekly_totals_arithmetic` (checks that session durations match the stated weekly total), `missing_data_acknowledgment`, and `missing_data_behavioral_response` (checks that the plan actually adjusts conservatively when telemetry is absent, not just mentions it).

Two markers carry explicit **failure conditions** that trigger a score of 1 regardless of other evidence:

- `load_progression_logic` — planned week 1 volume is more than 15% above the last 7 days without explicit justification.
- `non_competing_focus` — more than one non-running session per week without a stated rationale connecting it to the trail goal.

When `TRAILTRAINING_LIFESTYLE_NOTES` is set, the evaluator receives the constraints alongside the rubrics and adjusts scoring accordingly — road sessions forced by schedule constraints are not penalized under `trail_specificity`, and cycling sessions are not flagged under `non_competing_focus` when the schedule makes them the sensible weekday option.

---

## Stack

Python 3.9–3.12 · Pydantic v2 · OpenAI SDK (OpenRouter) · Flask (OAuth callback) · pytest · ruff · mypy

---

## Limitations

- Requires user-managed credentials and local setup to run
- Load modeling is intentionally simple (moving time × intensity proxy)
- Generation with structured JSON output requires OpenAI models via OpenRouter; non-OpenAI models lack reliable structured output support through the Responses API
- Not medical software — outputs should be reviewed with common sense

## License

MIT
