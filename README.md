# trailtraining

![CI](https://github.com/MauriceCC04/trailtraining/actions/workflows/ci.yml/badge.svg)

A local-first Python CLI that turns Strava/Garmin training data into auditable coaching artifacts — structured plans, deterministic evaluation, readiness forecasting, and iterative revision.

Most AI coaching features are generic and unverifiable. This project is an attempt to build something better: every output is grounded in local data, checked against explicit constraints, and revisable from its own evaluation report.

→ **No API keys needed to see what it produces.** Check out [`demo/`](demo) and [`docs/engineering.md`](docs/engineering.md).

---

## How it works

```text
Strava / Garmin / Intervals.icu
        │
        ▼
   local ingestion
        │
        ▼
  combine → combined_summary.json
          → combined_rollups.json (7d / 28d windows + load model state)
          → formatted_personal_data.json
        │
        ▼
  deterministic forecast
    - readiness + overreach risk
    - rolling 7d load vs prior rolling baseline
    - ATL / CTL / TSB state from daily training load
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
        ├──► revised-plan-reeval.json
        │
        ▼
  plan-to-ics  ──► training-plan.ics → Calendar.app
````

The pipeline runs in one direction: ingest → forecast → generate → evaluate → revise → export. Generated plans are treated as first drafts, not final answers.

---

## What the load model does

The base daily load signal is intentionally simple and robust:

```text
training_load_hours = moving_time_hours × load_factor
```

Where `load_factor` is derived from average HR / max HR when available, and falls back to `1.0` otherwise.

That raw daily load is then used in two places:

1. **Forecast scoring**
   * readiness / overreach risk use recent 7-day load against a prior rolling 7-day baseline
   * this is the primary scoring input today

2. **Load model state**
   * **ATL** = short-term fatigue proxy
   * **CTL** = longer-term fitness/base proxy
   * **TSB** = freshness proxy, computed as `CTL - ATL`

ATL and CTL are modeled as exponentially weighted moving averages of daily `training_load_hours`, with default time constants:
* **ATL:** 7 days
* **CTL:** 42 days
* **TSB:** `CTL - ATL`

This means the project now supports both:

* a simple, interpretable rolling-baseline forecast score
* a classic endurance load model state for downstream analysis and prompting

---

## Engineering decisions worth noting

**Two-stage evaluation.** `eval-coach` runs deterministic checks (ramp %, consecutive hard days, citation coverage). With `--soft-eval`, a second model acts as an independent judge using a rubric-driven schema — strengths, concerns, marker-level evidence.

**Per-rubric batch evaluation.** The soft evaluator runs one LLM call per rubric group rather than one monolithic call across all markers. This reduces inter-marker anchoring bias and makes marker-level scoring more reliable.

**Parallel batch execution.** Rubric batch calls run concurrently by default via a thread pool, cutting soft-eval wall-clock time materially. Disable with `--no-parallel-batches` for debugging or tight rate limits.

**Synthesis skip for fast iteration.** `--skip-synthesis` skips the final narrative LLM call during soft evaluation. Strengths, concerns, and improvements are derived locally from marker scores instead.

**Observation-before-score.** Each marker result requires an `observation` field before it may assign a score. This makes scoring decisions auditable and discourages lazy scoring.

**Revision is part of the pipeline.** `revise-plan` takes the original plan and its eval report and produces a revised artifact. With `--auto-reeval`, the revised plan is immediately re-evaluated and a delta score is computed.

**Lifestyle constraints separated from race goals.** Schedule and access constraints are stored separately from the primary goal. The coach sees both during generation, and the evaluator scores the plan against the best plan possible given those constraints.

---

## Setup

**Requirements:** Python 3.9+, a Strava API application, one wellness source (GarminDB or Intervals.icu), and an OpenRouter API key for generation/evaluation.

```bash
git clone https://github.com/MauriceCC04/trailtraining.git
cd trailtraining
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
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
TRAILTRAINING_SOFT_EVAL_MODEL="anthropic/claude-haiku-4.5"

# Coaching preferences
TRAILTRAINING_PRIMARY_GOAL="21k trail race, 1500m elevation on September 15 2026"
TRAILTRAINING_LIFESTYLE_NOTES="Weekdays: road runs or cycling only. One long mountain day on Saturday or Sunday."
TRAILTRAINING_PLAN_DAYS="7"   # 7, 14, 21, or 28
```

**`TRAILTRAINING_PRIMARY_GOAL`** accepts any free-form description of your target. If it contains a recognizable date, the CLI computes weeks-to-race and injects a recommended training phase into generation.

**`TRAILTRAINING_LIFESTYLE_NOTES`** accepts schedule or access constraints that affect session placement. Keep this separate from the race goal — the goal describes *what* you're training for, the lifestyle notes describe *when and where* you can train.

---

## Typical workflow

```bash
trailtraining --profile alice doctor
trailtraining --profile alice auth-strava
trailtraining --profile alice run-all
trailtraining --profile alice forecast
trailtraining --profile alice coach --prompt training-plan
trailtraining --profile alice eval-coach --soft-eval
trailtraining --profile alice revise-plan --auto-reeval
trailtraining --profile alice plan-to-ics
```

Other prompts: `recovery-status`, `meal-plan`, `session-review`.

---

## Forecast outputs

The forecast artifact is deterministic and validated. It currently includes:

* readiness score + status
* overreach risk score + level
* recent 7-day load
* prior rolling 7-day load baseline mean / std
* resting HR, sleep, HRV 7d vs 28d comparisons
* recovery telemetry coverage notes
* ATL / CTL / TSB load model state

This means you can inspect both:

* **why** the forecast scored readiness/risk the way it did
* **what** the athlete’s current load state looks like in a more classic endurance-training sense

---

## Signals and citations

The signal registry is the bridge between deterministic metrics and generated coaching text.

Signals include:

* `load.last7.*`
* `load.baseline28.*`
* `recovery.last7.*`
* `recovery.last28.sleep_hours_mean`
* `load.model.atl_hours`
* `load.model.ctl_hours`
* `load.model.tsb_hours`

The coach is expected to cite these signal IDs in generated plans and rationale so outputs stay auditable.

---

## 28-day plans

```bash
trailtraining --profile alice coach --prompt training-plan --plan-days 28
```

The plan is split into phased training weeks. Hard-day and rest-day constraints are enforced across rolling windows, not just single calendar weeks. `weekly_totals` reflects week 1 so ramp-rate validation remains accurate.

---

## Soft evaluation

```bash
trailtraining --profile alice eval-coach --soft-eval
trailtraining --profile alice eval-coach --soft-eval --soft-eval-runs 2
trailtraining --profile alice eval-coach --soft-eval --skip-synthesis
trailtraining --profile alice eval-coach --soft-eval --no-parallel-batches
```

With `--soft-eval-runs N`, the evaluator runs N independent passes and reports per-marker score variance. High variance means the rubric definition is ambiguous and probably needs tightening.

---

## Revision with automatic re-evaluation

```bash
trailtraining --profile alice revise-plan --auto-reeval
```

With `--auto-reeval`, the revised plan is re-evaluated immediately and a delta report is written with:

* `original_score`
* `revised_score`
* `delta_score`
* remaining violations

---

## Calendar export

```bash
trailtraining --profile alice plan-to-ics
trailtraining --profile alice plan-to-ics --no-open
trailtraining --profile alice plan-to-ics --start-hour 6 --output ~/Desktop/plan.ics
```

`plan-to-ics` picks the most recently modified plan, converts it to a standards-compliant `.ics` file, and on macOS can open it with Calendar.app.

---

## Output layout

```text
~/trailtraining-data/<profile>/prompting/
├── combined_summary.json
├── combined_rollups.json
├── formatted_personal_data.json
├── readiness_and_risk_forecast.json
├── coach_brief_training-plan.json / .txt
├── eval_report.json
├── revised-plan.json / .txt
├── revised-plan-reeval.json
├── training-plan.ics
└── coach_brief_<recovery-status|meal-plan|session-review>.md
```

`combined_rollups.json` now serves two roles:

* fixed-window rollups (`7`, `28`)
* current load-model state (`load_model`) when available

---

## Testing

```bash
pytest -q
mypy src/trailtraining
ruff check .
```
---

## Stack

Python 3.9–3.12 · Pydantic v2 · OpenAI SDK (OpenRouter) · Flask (OAuth callback) · pytest · ruff · mypy

---

## Limitations

* Requires user-managed credentials and local setup to run
* Base daily load is intentionally simple: `moving_time × intensity proxy`
* ATL / CTL / TSB are only as good as that underlying daily load signal
* Readiness/risk scoring still relies mainly on rolling-baseline heuristics rather than full Banister-style performance modeling
* Generation with structured JSON output works best with OpenAI models via OpenRouter
* Not medical software — outputs should be reviewed with common sense

## License

MIT
