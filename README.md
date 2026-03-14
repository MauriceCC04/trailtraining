# trailtraining

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A Python CLI for building a clean training dataset from **Strava** plus **GarminDB** or **Intervals.icu**, then generating useful downstream artifacts such as readiness forecasts, structured plan drafts, and constraint-based evaluations of LLM-generated coaching output.

I originally built this because I train as a trailrunner and wanted a more reliable way to combine activity and wellness data than switching between vendor dashboards. The project then evolved into a reusable, multi-profile CLI with testing, packaging, and reproducible outputs.

---

## Why this project matters

Most training apps are good at storage and visualization, but weak at:

- Integrating LLMs to provide a truly useful result.
- combining activity data with wellness/recovery signals in one place
- producing exportable, analysis-ready artifacts
- supporting custom planning logic and evaluation rules

`trailtraining` addresses that by turning fragmented provider data into a consistent local pipeline.

---

## What it does

### Data pipeline
- Pulls **wellness / recovery** data from either:
  - **Intervals.icu**
  - **GarminDB**
- Pulls **activities** from Strava
- Normalizes and combines the data into a consistent summary format
- Produces rollups and downstream artifacts for analysis and planning

### Coaching layer
- Generates structured outputs such as:
  - training plan JSON
  - recovery status
  - meal-plan style suggestions
- Evaluates generated plans against simple safety / consistency rules, such as:
  - excessive weekly ramp rate
  - too many consecutive hard days
  - insufficient recovery spacing

### Multi-user support
- Supports separate `--profile` configurations on the same machine
- Keeps credentials, tokens, and outputs isolated per user

---

## Repository structure


├── .github/workflows/      # CI
├── demo/                   # sample outputs for previewing the project without credentials
├── src/trailtraining/      # package source code
├── README.md
├── pyproject.toml
└── requirements.txt

Demo
You cannot run the full pipeline without provider credentials, but you can still inspect representative outputs.
The demo/ folder contains examples such as:
	•	combined training summaries
	•	forecast outputs
	•	example coach generations
	•	evaluation artifacts
This is the fastest way to understand what the pipeline produces before configuring APIs.

Quickstart
# 1) install (see Installation)
```bash
trailtraining --profile alice doctor
```

# 2) authorize Strava once per profile
```bash
trailtraining --profile alice auth-strava
```
# 3) run the full pipeline
```bash
trailtraining --profile alice run-all
```
# 4) compute deterministic readiness + overreach risk
```bash
trailtraining --profile alice forecast
```
# 5) generate a structured training plan
```bash
trailtraining --profile alice coach --prompt training-plan
```

# 6) evaluate the generated plan
```bash
trailtraining eval-coach --input ~/trailtraining-data/alice/prompting/coach_brief_training-plan.json
```

Installation
Requirements
	•	Python 3.9+
	•	A Strava API application
	•	One wellness provider:
	◦	Intervals.icu API access, or
	◦	GarminDB installed locally
	•	OpenAI API key for coach features
Install
```bash
git clone https://github.com/MauriceCC04/trailtraining.git
cd trailtraining

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
# Intervals users
```bash
pip install -e .
````
# Garmin users
```bash
pip install -e ".[garmin]"
For development tools:
pip install -e ".[dev]"
Verify installation:
trailtraining -h
```
Configuration
Profiles make it easy to run the CLI for multiple users on the same machine.
When you run:
```bash
trailtraining --profile alice run-all
```
the CLI uses profile-specific environment variables, token storage, and output directories.
Example profile setup
```bash
mkdir -p ~/.trailtraining/profiles
nano ~/.trailtraining/profiles/alice.env
Example:
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

# Optional
# TRAILTRAINING_WELLNESS_PROVIDER="auto"
# TRAILTRAINING_BASE_DIR="$HOME/trailtraining-data/alice"
# TRAILTRAINING_LOG_LEVEL="INFO"

# LLM coach
OPENAI_API_KEY="..."
TRAILTRAINING_LLM_MODEL="gpt-5.2"
TRAILTRAINING_REASONING_EFFORT="medium"
```
Typical workflow
A standard run looks like this:
	1	Pull wellness data
	2	Authenticate and pull from strava
	3	Normalize and merge records
	4	Save derived summaries and rollups
	5	Run deterministic forecast logic
	6	Generate a structured plan
	7	Evaluate the generated plan against safety rules
This separation is intentional:
	•	the data pipeline is deterministic and reproducible
	•	the coaching layer is the core product experience, while remaining cleanly separated from the deterministic pipeline

Outputs
Depending on commands and provider setup, the project can produce artifacts such as:
	•	merged daily summaries
	•	rollup JSON files
	•	deterministic forecast outputs
	•	structured coaching outputs
	•	evaluation reports on generated plans
The demo/ folder shows representative examples.

Engineering choices
Highlights:
	•	packaged Python CLI
	•	isolated multi-profile execution
	•	local-first workflow
	•	provider abstraction for Garmin vs Intervals
	•	reproducible artifacts written to disk
	•	CI and development tooling for code quality

Testing and development
Install development dependencies:
```bash
pip install -e ".[dev]"
pre-commit install
Run checks locally:
pytest
ruff check .
mypy src
```
Limitations
	•	Requires user-owned API credentials and local setup
	•	Data quality depends on provider exports / APIs
	•	LLM-generated coaching outputs are experimental and should not be treated as medical advice
	•	Some provider-specific behavior may require manual troubleshooting
	•	Reliance on 3rd party API (GarminDB) due to restrictions from Garmin
