# trailtraining

![CI](https://github.com/MauriceCC04/trailtraining/actions/workflows/ci.yml/badge.svg)

A local-first Python CLI that turns Strava, Garmin, and Intervals.icu data into auditable coaching artifacts: structured plans, deterministic evaluation, readiness forecasting, soft rubric-based review, iterative revision, and calendar export.

Most AI coaching tools are generic and unverifiable. `trailtraining` is an attempt to build something better: every output is grounded in local data, checked against explicit constraints, and revisable from its own evaluation report.

→ **No API keys needed to inspect the artifact shapes.** Check out [`demo/`](demo)

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
          → combined_rollups.json
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
        ├──► revised-plan-comparison.json
        ├──► selected-plan.json / .txt
        ├──► revised-plan-reeval.json
        │
        ▼
  plan-to-ics  ──► training-plan.ics → Calendar.app
```

The pipeline is directional:

**ingest → forecast → generate → evaluate → revise/select → export**

Generated plans are treated as first drafts, not final answers.

---

## One-shot workflow

If you want the whole planning loop in one command:

```bash
trailtraining --profile alice run-training-cycle
```

That unified command runs the equivalent of:

```bash
trailtraining --profile alice run-all-intervals
trailtraining --profile alice forecast
trailtraining --profile alice coach --prompt training-plan --plan-days 28
trailtraining --profile alice eval-coach --soft-eval --soft-eval-runs 2
trailtraining --profile alice revise-plan --auto-reeval
trailtraining --profile alice eval-coach --input ~/trailtraining-data/alice/prompting/revised-plan.json --soft-eval
```

It also preserves separate original and revised evaluation artifacts so you can compare them directly.

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
   * this is the main deterministic scoring input today

2. **Load model state**

   * **ATL** = short-term fatigue proxy
   * **CTL** = longer-term fitness/base proxy
   * **TSB** = freshness proxy, computed as `CTL - ATL`

ATL and CTL are modeled as exponentially weighted moving averages of daily `training_load_hours`, with default time constants:

* **ATL:** 7 days
* **CTL:** 42 days
* **TSB:** `CTL - ATL`

This means the project supports both:

* a simple, interpretable rolling-baseline forecast score
* a classic endurance load-model state for downstream analysis and prompting

---

## Engineering decisions worth noting

**Deterministic evaluation comes first.** `eval-coach` checks hard constraints such as ramp %, hard-day spacing, rest-day minimums, and citation coverage.

**Soft evaluation is optional but structured.** With `--soft-eval`, a second model scores the plan using a rubric-driven schema: strengths, concerns, marker-level evidence, and suggested improvements.

**Per-rubric batch evaluation.** The soft evaluator runs one LLM call per rubric group rather than one monolithic call across all markers. This reduces inter-marker anchoring bias and makes marker-level scoring easier to inspect.

**Parallel batch execution.** Rubric batch calls run concurrently by default. Disable with `--no-parallel-batches` for debugging or tight rate limits.

**Observation-before-score.** Each marker result requires an `observation` field before a score can be assigned. This makes scoring decisions more auditable.

**Revision is a revise-and-select step.** `revise-plan` generates a revised candidate from the original plan plus its eval report, then records pairwise comparison metadata. The final saved artifact may preserve the original plan if the pairwise judge prefers it. The comparison metadata and companion artifacts are written separately so that outcome is inspectable.

**Lifestyle constraints are separate from race goals.** Schedule and access constraints are stored separately from the primary goal. The coach sees both during generation, and the evaluator scores the plan against the best plan possible given those constraints.

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

**`TRAILTRAINING_LIFESTYLE_NOTES`** accepts schedule or access constraints that affect session placement. Keep this separate from the race goal: the goal describes *what* you're training for, the lifestyle notes describe *when and where* you can train.

---

## Typical workflows

### Manual workflow

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

### Intervals-backed end-to-end workflow

```bash
trailtraining --profile alice run-training-cycle
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
* ATL / CTL / TSB load-model state

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

Generated plans are expected to cite these signal IDs in structured fields and rationale so outputs stay auditable.

---

## 28-day plans

```bash
trailtraining --profile alice coach --prompt training-plan --plan-days 28
```

Multi-week plans are split into phased weeks. Hard-day and rest-day constraints are enforced across rolling windows, not just single calendar weeks. `weekly_totals` reflects **week 1**, not the full 28-day sum, so ramp-rate validation remains meaningful.

---

## Soft evaluation

```bash
trailtraining --profile alice eval-coach --soft-eval
trailtraining --profile alice eval-coach --soft-eval --soft-eval-runs 2
trailtraining --profile alice eval-coach --soft-eval --skip-synthesis
trailtraining --profile alice eval-coach --soft-eval --no-parallel-batches
```

With `--soft-eval-runs N`, the evaluator runs `N` independent passes and reports per-marker score variance. High variance means the rubric definition is ambiguous and probably needs tightening.

---

## Revision and automatic re-evaluation

```bash
trailtraining --profile alice revise-plan --auto-reeval
```

`revise-plan` uses the original training plan plus its evaluation report to generate a revised candidate. It also writes comparison metadata so you can inspect whether the revised candidate or the original was preferred in pairwise judging.

With `--auto-reeval`, a re-evaluation delta file is written with:

* `original_score`
* `revised_score`
* `delta_score`
* remaining violations
* grade / score metadata used by the current re-eval step

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
├── eval_report.original.json          # when using the unified cycle
├── eval_report.revised.json           # when using the unified cycle
├── revised-plan.json / .txt
├── revised-plan-comparison.json
├── selected-plan.json / .txt
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
* Revision quality still depends on the quality of both the first-pass plan and the evaluation report
* Not medical software — outputs should be reviewed with common sense

---

## License

MIT
