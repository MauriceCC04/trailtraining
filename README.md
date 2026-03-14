# trailtraining

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A Python CLI for building a clean training dataset from **Strava** plus **GarminDB** or **Intervals.icu**, then generating downstream artifacts such as readiness forecasts, structured training-plan drafts, and constraint-based evaluations of LLM-generated coaching output.

I originally built this because I train as a trail runner and wanted a more reliable way to combine activity and wellness data than switching between vendor dashboards. The project evolved into a reusable, multi-profile CLI with testing, packaging, and reproducible local outputs.

---

## Why this project matters

Most training apps are good at storage and visualization, but weak at:

- combining activity data with wellness and recovery signals in one place
- producing exportable, analysis-ready artifacts
- supporting custom planning logic and evaluation rules
- using LLMs in a way that is structured, testable, and producing useful outputs

`trailtraining` addresses that by turning fragmented provider data into a consistent local pipeline.

---

## What it does

### Data pipeline

- Pulls **activities** from Strava
- Pulls **wellness / recovery** data from either:
  - **Intervals.icu**
  - **GarminDB**
- Normalizes and combines the data into a consistent summary format
- Produces rollups and downstream artifacts for analysis and planning

### Forecasting layer

- Computes a deterministic readiness and overreach-risk forecast from recent training and recovery signals
- Writes reproducible JSON artifacts to disk

### Coaching layer

- Generates structured outputs such as:
  - training plans
  - recovery status summaries
  - meal-plan style suggestions
- Evaluates generated training plans against simple safety and consistency rules, such as:
  - excessive weekly ramp rate
  - too many consecutive hard days
  - insufficient recovery spacing

### Multi-profile support

- Supports separate `--profile` configurations on the same machine
- Keeps credentials, tokens, and outputs isolated per user

---

## Repository structure

```text
.
├── .github/workflows/      # CI
├── demo/                   # sample outputs for previewing the project without credentials
├── src/trailtraining/      # package source code
├── tests/                  # test suite
├── README.md
├── pyproject.toml
└── requirements.txt
````

---

## Demo

You cannot run the full pipeline without provider credentials, but you can still inspect representative outputs.

The `demo/` folder contains examples such as:

* combined training summaries
* forecast outputs
* example coach generations
* evaluation artifacts

This is the fastest way to understand what the pipeline produces before configuring APIs.

---

## Requirements

* Python **3.9+**
* A **Strava API application**
* One wellness provider:

  * **Intervals.icu** API access, or
  * **GarminDB** installed locally
* An **OpenAI API key** for `coach` features

---

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
# development tools
pip install -e ".[dev]"

# analysis utilities
pip install -e ".[analysis]"
```

Verify installation:

```bash
trailtraining -h
```

### Notes

* Garmin workflows also require **GarminDB** to be installed and available locally
* OpenAI access is only required for `coach` commands

---

## Configuration

Profiles make it easy to run the CLI for multiple users on the same machine.

When you run:

```bash
trailtraining --profile alice run-all
```

the CLI:

* loads `~/.trailtraining/profiles/alice.env` if it exists
* sets `TRAILTRAINING_PROFILE=alice`
* isolates outputs under a default base directory for that profile

By default, profile data lives under:

```text
~/trailtraining-data/<profile>
```

### Example profile setup

```bash
mkdir -p ~/.trailtraining/profiles
nano ~/.trailtraining/profiles/alice.env
```

Example `~/.trailtraining/profiles/alice.env`:

```bash
# --- Strava ---
STRAVA_CLIENT_ID="..."
STRAVA_CLIENT_SECRET="..."
STRAVA_REDIRECT_URI="http://127.0.0.1:5000/authorization"

# --- Choose ONE wellness provider ---

# Option A: Garmin
GARMIN_EMAIL="alice@example.com"
GARMIN_PASSWORD="..."

# Option B: Intervals.icu
# INTERVALS_API_KEY="..."
# INTERVALS_ATHLETE_ID="0"

# Optional provider override
# TRAILTRAINING_WELLNESS_PROVIDER="auto"

# Optional custom data directory
# Use ~/... rather than $HOME/... here
# TRAILTRAINING_BASE_DIR="~/trailtraining-data/alice"

# Optional logging
# TRAILTRAINING_LOG_LEVEL="INFO"

# LLM coach
OPENAI_API_KEY="..."
TRAILTRAINING_LLM_MODEL="gpt-5.2"
TRAILTRAINING_REASONING_EFFORT="medium"
TRAILTRAINING_VERBOSITY="medium"
TRAILTRAINING_COACH_DAYS="60"
TRAILTRAINING_COACH_STYLE="trailrunning"
```

---

## Quickstart

### 1) Check configuration

```bash
trailtraining --profile alice doctor
```

### 2) Authorize Strava once per profile

```bash
trailtraining --profile alice auth-strava
```

### 3) Run the full pipeline

Auto-detect the wellness provider:

```bash
trailtraining --profile alice run-all
```

Or force a specific provider:

```bash
trailtraining --profile alice run-all --wellness-provider intervals
trailtraining --profile alice run-all --wellness-provider garmin
```

### 4) Compute deterministic readiness and overreach risk

```bash
trailtraining --profile alice forecast
```

### 5) Generate a structured training plan

```bash
trailtraining --profile alice coach --prompt training-plan
```

Other prompts:

```bash
trailtraining --profile alice coach --prompt recovery-status
trailtraining --profile alice coach --prompt meal-plan
```

### 6) Evaluate a generated plan

```bash
trailtraining --profile alice eval-coach \
  --input ~/trailtraining-data/alice/prompting/coach_brief_training-plan.json
```

---

## Command overview

### Core commands

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
```

### Useful options

```bash
trailtraining --profile alice run-all --clean
trailtraining --profile alice run-all --clean-processing
trailtraining --profile alice run-all --clean-prompting
trailtraining --profile alice fetch-intervals --oldest 2025-01-01 --newest 2025-03-01
trailtraining --profile alice coach --prompt training-plan --style trailrunning
trailtraining --profile alice coach --prompt training-plan --style triathlon
```

---

## Typical workflow

A standard run looks like this:

1. Pull wellness data
2. Authenticate and pull activities from Strava
3. Normalize and merge records
4. Save derived summaries and rollups
5. Run deterministic forecast logic
6. Generate a structured plan or coaching artifact
7. Evaluate the generated plan against safety rules

This separation is intentional:

* the data pipeline is deterministic and reproducible
* the forecasting layer is deterministic and inspectable
* the coaching layer is the user-facing AI feature, while remaining cleanly separated from the core pipeline

---

## Outputs

Depending on commands and provider setup, the project can produce artifacts such as:

* merged daily summaries
* rollup JSON files
* deterministic forecast outputs
* structured coaching outputs
* evaluation reports on generated plans

Typical output location for a profile:

```text
~/trailtraining-data/alice/
├── processing/
└── prompting/
```

The `demo/` folder shows representative examples.

---

## Engineering choices

Highlights:

* packaged Python CLI
* isolated multi-profile execution
* local-first workflow
* provider abstraction for Garmin vs Intervals
* reproducible artifacts written to disk
* CI and development tooling for code quality
* structured outputs and rule-based evaluation for LLM-generated plans

---

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
pre-commit install
```

Run checks locally:

```bash
pytest
ruff check .
mypy src
```

---

## Limitations

* Requires user-owned API credentials and local setup
* Data quality depends on provider exports and APIs
* LLM-generated coaching outputs are experimental and should not be treated as medical advice
* Some provider-specific behavior may require manual troubleshooting
* Garmin workflows rely on **GarminDB** because Garmin does not offer an open public API for this use case

---

## Safety note

This project is a personal training-data and planning tool. It is not medical software, and generated outputs should not be treated as medical advice.

---

## License

MIT
