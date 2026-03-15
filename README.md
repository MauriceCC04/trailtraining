# trailtraining

![CI](https://github.com/MauriceCC04/trailtraining/actions/workflows/ci.yml/badge.svg)

`trailtraining` is a local-first Python CLI for turning training data and optional recovery telemetry into inspectable coaching artifacts.

It ingests activity history, merges it with optional wellness signals, computes deterministic readiness and overreach context, generates structured coaching outputs, evaluates those outputs against explicit guardrails, and can now revise a generated training plan using its own evaluation report.

The project exists because most fitness products are good at collecting data and weak at turning that data into something actionable. And most "AI coaching" features still feel generic, ungrounded, and hard to trust.

`trailtraining` is an attempt to build something better:

- local artifacts instead of opaque black boxes
- structured plans instead of vague summaries
- deterministic forecasting before generation
- explicit evaluation after generation
- iterative plan revision instead of blind trust in the first output

This is not a chatbot wrapper around Strava. It is an auditable training-planning pipeline.

## What changed recently

The current version now supports a full revision loop:

1. generate `coach_brief_training-plan.json`
2. evaluate it with `eval-coach`
3. optionally add a second-model qualitative assessment via `--soft-eval`
4. revise the original plan into `revised-plan.json`
5. render `revised-plan.txt`
6. re-evaluate the revised plan

It also hardens soft evaluation so imperfect judge-model outputs can still produce useful reports by repairing missing marker results, deriving rubric scores from markers when needed, and backfilling qualitative feedback lists.
- **Independent critique is better than self-grading.** The revision loop is designed so that, when possible, a different model family critiques and revises the original plan. In practice, that means GPT can generate the first-pass plan while Claude evaluates and revises it.

## Engineering highlights

This project is built around four ideas.

### 1. Context quality determines output quality

The LLM is not asked to invent advice from nowhere. The pipeline first assembles local context:

- recent activity rollups
- optional recovery telemetry
- deterministic readiness and overreach signals
- explicit athlete goal and style

Generation happens only after those signals exist.

### 2. LLM outputs should be evaluated, not trusted

Generated plans are checked after generation.

`eval-coach` scores plans against deterministic constraints such as:

- ramp rate versus recent load
- hard-day spacing
- rest structure
- internal consistency between days and weekly totals
- grounding and plan quality markers

With `--soft-eval`, a second model acts as a judge and produces a structured qualitative assessment with rubric scores, marker-level evidence, strengths, concerns, and suggested improvements.

### 3. Revision is part of the pipeline

Generation is no longer the endpoint.

`revise-plan` takes:

- `coach_brief_training-plan.json`
- `eval_report.json`

and produces:

- `revised-plan.json`
- `revised-plan.txt`

This makes the system iterative: generate → critique → revise → re-check.

### 4. Structured contracts prevent drift

Artifacts are validated with strict Pydantic models and matching JSON-schema-driven LLM calls.

That keeps the system inspectable and makes downstream steps easier to debug and trust.

## Quick look: sample artifacts

You do **not** need API keys to understand what the project produces.

Start with:

- [demo/](demo)
- [docs/engineering.md](docs/engineering.md)

Representative demo artifacts include:

- `demo/rollups/combined_rollups.json`
- `demo/plans/coach_brief_training-plan.json`
- `demo/plans/coach_brief_training-plan.txt`
- `demo/status/coach_brief_recovery-status.md`
- `demo/status/coach_brief_meal-plan.md`

If you add revision-loop demo assets, this list should also include:

- `demo/eval/eval_report.json`
- `demo/plans/revised-plan.json`
- `demo/plans/revised-plan.txt`

![Example generated training plan](docs/images/training-plan-output.png)

## Pipeline at a glance

```text
Strava ───────────────┐
                      │
GarminDB /            ├──► local ingestion ───► combine ───► combined_summary.json
Intervals.icu ────────┘                                 │
                                                        ▼
                                            deterministic forecast
                                     (readiness + overreach context)
                                                        │
                                                        ▼
                                           coach --prompt training-plan
                                                        │
                                                        ▼
                                   coach_brief_training-plan.json/.txt
                                                        │
                                                        ▼
                                     eval-coach (deterministic checks)
                                                        │
                                  ┌─────────────────────┴─────────────────────┐
                                  │                                           │
                                  ▼                                           ▼
                      eval_report.json                            eval-coach --soft-eval
                 (score / grade / violations)                  (second-model qualitative judge)
                                  │                                           │
                                  └─────────────────────┬─────────────────────┘
                                                        ▼
                                              revise-plan command
                           (original plan + eval_report.json -> revised artifact)
                                                        │
                                                        ▼
                                         revised-plan.json / revised-plan.txt
                                                        │
                                                        ▼
                                     eval-coach --input revised-plan.json
```

## What the project does

`trailtraining` can:

- pull activity history from Strava
- optionally pull recovery telemetry from GarminDB or Intervals.icu
- combine local activity and recovery artifacts
- compute simple recent-load, readiness, and overreach-risk signals
- generate structured coaching outputs
- evaluate generated training plans against explicit constraints
- run optional soft evaluation with a second model as judge
- revise a generated training plan from its evaluation report
- render machine-readable and human-readable plan artifacts
- support isolated multi-profile setups with `--profile`

## Why this project exists

Most fitness platforms are good at storing data and bad at using it.

You can collect heart rate, sleep, HRV, workout history, training load, and pace for months, but the end result is usually one of these:

- dashboards without concrete decisions
- generic AI summaries
- suggestions that ignore recent load
- outputs that sound polished but are impossible to audit

`trailtraining` is meant to be more useful in practice:

1. collect real training data
2. compute local signals
3. generate a structured output from those signals
4. evaluate that output
5. revise it when needed

That makes it closer to a decision-support system than a generic AI wrapper.

## Handling incomplete data

The pipeline is designed to degrade gracefully when recent recovery telemetry is sparse.

It can run on activity-only data and improves when recent sleep, resting HR, or HRV data are available. When recovery data are missing, forecasts and generated plans should be interpreted more conservatively.

## Plan evaluation and soft evaluation

Generated plans are not treated as correct just because they sound plausible.

### Deterministic evaluation

`eval-coach` checks plans against explicit constraints such as:

- excessive ramp versus recent load
- poor spacing of hard sessions
- insufficient rest
- inconsistencies between day-level sessions and weekly totals
- weak signal grounding

It writes `eval_report.json` with score, grade, violation details, and summary statistics.

### Soft evaluation

With `--soft-eval`, a second model judges plan quality using a rubric-driven schema.

The soft assessment can include:

- overall qualitative score and grade
- rubric scores
- marker-level evidence
- strengths
- concerns
- suggested improvements

The implementation is deliberately robust: if the judge response is incomplete, the system can recover missing marker results, derive rubric scores from markers, and backfill qualitative lists.

## Revision loop

The new `revise-plan` command uses the original plan and its evaluation report to generate a revised version of the same structured artifact.

Inputs:

- `coach_brief_training-plan.json`
- `eval_report.json`

Outputs:

- `revised-plan.json`
- `revised-plan.txt`

The reviser is asked to:

- preserve strong parts of the original plan
- fix issues identified by deterministic and soft evaluation
- maintain schema compatibility
- keep the result grounded in the original signals and citations

## What you can do without setup

Without creating any credentials, you can:

- inspect the sample outputs in `demo/`
- read the command surface in this README
- review the engineering notes in `docs/engineering.md`
- understand the full generation / evaluation / revision loop

## What requires setup

To run the full pipeline on your own data, you need:

- Python 3.9+
- a Strava API application
- one wellness source:
  - GarminDB, or
  - Intervals.icu API access
- an OpenRouter API key for generation, soft evaluation, and revision

## Installation

```bash
git clone https://github.com/MauriceCC04/trailtraining.git
cd trailtraining

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[analysis]"
```

Verify installation:

```bash
trailtraining -h
```

## Configuration

Profiles load environment variables from:

```bash
~/.trailtraining/profiles/<profile>.env
```

Example:

```bash
mkdir -p ~/.trailtraining/profiles
nano ~/.trailtraining/profiles/alice.env
```

Minimal example:

```bash
STRAVA_CLIENT_ID="..."
STRAVA_CLIENT_SECRET="..."
STRAVA_REDIRECT_URI="http://127.0.0.1:5000/authorization"

# Choose one wellness source

# Garmin
GARMIN_EMAIL="alice@example.com"
GARMIN_PASSWORD="..."

# or Intervals.icu
# INTERVALS_API_KEY="..."
# INTERVALS_ATHLETE_ID="0"

OPENROUTER_API_KEY="sk-or-v1-..."
TRAILTRAINING_LLM_MODEL="openai/gpt-5.2"
TRAILTRAINING_SOFT_EVAL_MODEL="anthropic/claude-sonnet-4"
```

By default, per-profile data is stored under:

```bash
~/trailtraining-data/<profile>
```

## Typical workflow

```bash
# 1. Check setup
trailtraining --profile alice doctor

# 2. Authorize data sources
trailtraining --profile alice auth-strava

# 3. Ingest and combine data
trailtraining --profile alice run-all

# 4. Forecast deterministic readiness / risk
trailtraining --profile alice forecast

# 5. Generate first-pass training plan
trailtraining --profile alice coach --prompt training-plan

# 6. Evaluate it with deterministic + soft checks
trailtraining --profile alice eval-coach --soft-eval

# 7. Revise the plan from the eval report
trailtraining --profile alice revise-plan

# 8. Re-evaluate the revised plan
trailtraining --profile alice eval-coach \
  --input ~/trailtraining-data/alice/prompting/revised-plan.json \
  --soft-eval
```

## Output layout

Typical outputs live under:

```text
~/trailtraining-data/<profile>/
├── processing/
└── prompting/
```

Common prompting artifacts:

```text
prompting/
├── combined_summary.json
├── combined_rollups.json
├── readiness_and_risk_forecast.json
├── coach_brief_training-plan.json
├── coach_brief_training-plan.txt
├── eval_report.json
├── revised-plan.json
└── revised-plan.txt
```

These artifacts are intended to answer questions like:

- How recovered do I look right now?
- Am I trending toward overreach?
- What kind of week makes sense from here?
- Does the generated plan violate obvious constraints?
- Does the soft judge find the plan specific and coherent?
- Did the revised plan improve after critique?

## Repo layout

```text
.
├── .github/workflows/
├── demo/
│   ├── README.md
│   ├── rollups/
│   ├── plans/
│   └── status/
├── docs/
│   ├── engineering.md
│   └── images/
├── src/trailtraining/
├── tests/
├── README.md
└── pyproject.toml
```

## Command reference

Core commands:

```bash
trailtraining --profile alice doctor
trailtraining --profile alice auth-strava
trailtraining --profile alice fetch-strava
trailtraining --profile alice fetch-garmin
trailtraining --profile alice fetch-intervals
trailtraining --profile alice combine
trailtraining --profile alice run-all
trailtraining --profile alice run-all-intervals
trailtraining --profile alice forecast
trailtraining --profile alice coach --prompt training-plan
trailtraining --profile alice eval-coach --input <path>
trailtraining --profile alice eval-coach --soft-eval
trailtraining --profile alice revise-plan
```

Useful examples:

```bash
trailtraining --profile alice run-all --clean
trailtraining --profile alice fetch-intervals --oldest 2025-01-01 --newest 2025-03-01
trailtraining --profile alice coach --prompt training-plan --style trailrunning
trailtraining --profile alice coach --prompt training-plan --style triathlon
trailtraining --profile alice eval-coach --soft-eval
trailtraining --profile alice revise-plan --output ~/tmp/revised-plan.json
```

## Development

Install dev dependencies:

```bash
pip install -e ".[dev]"
pre-commit install
```

Run checks:

```bash
pytest
ruff check .
mypy src tests
```

## Current limitations

- full runs require user-managed credentials and local setup
- data quality depends on upstream providers
- recovery telemetry may be incomplete
- load modeling is intentionally simple
- soft-eval first-pass judge output can still be messy, even though recovery logic now makes it more robust
- this is an engineering-focused tool, not a polished consumer product

## Safety

This is a personal training-data and planning tool.

It is **not** medical software, and generated outputs should not be treated as medical advice. Any recommendation should be reviewed with common sense and adjusted for injury status, recovery, and individual context.

## License

MIT
