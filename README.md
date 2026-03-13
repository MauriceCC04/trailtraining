[![CI](https://github.com/MauriceCC04/trailtraining/actions/workflows/ci.yml/badge.svg)](https://github.com/MauriceCC04/trailtraining/actions/workflows/ci.yml)

# trailtraining

A Python CLI that pulls your **training + wellness** data (**Strava + (GarminDb or Intervals.icu)**), combines it into a clean dataset, and optionally runs an **LLM “coach”** to generate a structured training plan and evaluate it against safety constraints.

**Designed for multi-user setups** on the same machine via `--profile` (separate tokens + separate data folders per user).

---

## What it does

**Pipeline**
- Fetch **wellness** from **GarminDb** *or* **Intervals.icu**
- Fetch **activities** from **Strava**
- Combine/normalize into a consistent JSON summary + rollups

**Optional coaching**
- Generate:
  - training plan (**structured JSON**)
  - recovery status
  - meal plan
- Evaluate training plans for basic safety/consistency rules (e.g., ramp rate, consecutive hard days)

---

## Demo (no credentials)

You can’t run the full pipeline without provider credentials, but you can still preview outputs.

➡️ **Sample outputs live in [`demo/`](demo/)** (combined summaries, forecasts, and example coach outputs).

---

## Quickstart

```bash
# 1) install (see Installation)
trailtraining --profile alice doctor

# 2) authorize Strava once per profile
trailtraining --profile alice auth-strava

# 3) run the full pipeline (auto-selects Intervals vs Garmin)
trailtraining --profile alice run-all

# 4) compute deterministic readiness + overreach risk (recommended)
trailtraining --profile alice forecast

# 5) generate a structured training plan (JSON)
trailtraining --profile alice coach --prompt training-plan

# 6) evaluate the plan against constraints
trailtraining eval-coach --input ~/trailtraining-data/alice/prompting/coach_brief_training-plan.json
````

---

## Prerequisites

* **Python 3.9+**
* **Strava API application** (Client ID + Client Secret) **per user/profile**
* You must create your own strava API application for each user/profile to get unique credentials. Sharing credentials between users is not recommended.
* Choose **one** wellness provider:

  * **Intervals.icu** API access (API key + athlete ID), **or**
  * **GarminDb** installed + CLI available (`garmindb_cli` or `garmindb_cli.py`)

Optional:

* **OpenAI API key** for the coach feature

---

## Installation (macOS / Linux)

```bash
git clone https://github.com/MauriceCC04/trailtraining.git
cd trailtraining

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Intervals users:
pip install -e .

# Garmin users:
pip install -e ".[garmin]"
```

Dev tooling (tests + ruff + pre-commit):

```bash
pip install -e ".[dev]"
```

Verify:

```bash
trailtraining -h
```

---

## Profiles (multi-user support)

Profiles make it easy to run for multiple people on the same machine.

When you run:

```bash
trailtraining --profile alice run-all
```

the CLI will:

1. Load environment variables from `~/.trailtraining/profiles/alice.env` (if it exists)
2. Use an isolated default data directory: `~/trailtraining-data/alice/` (overrideable)
3. Store Strava tokens at: `~/trailtraining-data/alice/tokens/strava_token.json`
4. If using GarminDb, manage a per-profile config under `~/.trailtraining/garmin/alice/`

> The profile env file does **not** override variables already set in your shell (shell vars win).

### Create a profile env file

```bash
mkdir -p ~/.trailtraining/profiles
nano ~/.trailtraining/profiles/alice.env
```

Template:

```bash
# --- Strava (PER USER) ---
STRAVA_CLIENT_ID="..."
STRAVA_CLIENT_SECRET="..."
STRAVA_REDIRECT_URI="http://127.0.0.1:5000/authorization"

# --- Choose ONE wellness provider ---

# Option A: Garmin
GARMIN_EMAIL="alice@example.com"
GARMIN_PASSWORD="..."

# Option B: Intervals.icu
# INTERVALS_API_KEY="..."
# INTERVALS_ATHLETE_ID="0"   # "0" = current athlete

# Optional: force provider selection (otherwise auto-detect)
# TRAILTRAINING_WELLNESS_PROVIDER="auto"   # auto|garmin|intervals

# Optional: override where this profile stores all data
# TRAILTRAINING_BASE_DIR="$HOME/trailtraining-data/alice"

# Optional: logging verbosity
# TRAILTRAINING_LOG_LEVEL="INFO"  # CRITICAL|ERROR|WARNING|INFO|DEBUG

# --- Optional LLM coach ---
OPENAI_API_KEY="..."
TRAILTRAINING_LLM_MODEL="gpt-5.2"
TRAILTRAINING_REASONING_EFFORT="medium"   # none|low|medium|high|xhigh
TRAILTRAINING_VERBOSITY="medium"          # low|medium|high
TRAILTRAINING_COACH_STYLE="trailrunning"  # trailrunning|triathlon
```

Repeat for `bob.env`, etc.

### Credential hygiene

* Do **not** commit `.env` files or token JSONs.
* All secrets should live in environment variables (ideally in the profile env file above).

---

## Doctor (recommended)

Run this first whenever you’re setting up a profile:

```bash
trailtraining --profile alice doctor
```

Checks:

* Strava env vars + token presence
* Wellness provider credentials
* GarminDb CLI availability (if using Garmin)
* OpenAI key presence (optional)

---

## Strava setup (required, per user)

### 1) Create a Strava API application (per user)

For each Strava account (Alice, Bob, …):

1. Log into Strava
2. Create a Strava API application
3. Copy `STRAVA_CLIENT_ID` + `STRAVA_CLIENT_SECRET`
4. Set redirect URI to: `http://127.0.0.1:5000/authorization`

Put those values in the corresponding `~/.trailtraining/profiles/<name>.env`.

### 2) Authorize once per profile

```bash
trailtraining --profile alice auth-strava
trailtraining --profile bob auth-strava
```

Tip: use an Incognito/Private window when switching between Strava accounts.

---

## Wellness provider: Intervals.icu (fastest)

Set in your profile env:

```bash
INTERVALS_API_KEY="..."
INTERVALS_ATHLETE_ID="0"
```

Fetch wellness only:

```bash
trailtraining --profile alice fetch-intervals
```

Optional date range:

```bash
trailtraining --profile alice fetch-intervals --oldest "2023-01-01" --newest "2026-02-27"
```

---

## Wellness provider: GarminDb

Install GarminDb per its documentation and ensure the CLI is available:

* `garmindb_cli` or `garmindb_cli.py`

If needed:

```bash
export GARMINGDB_CLI="/full/path/to/garmindb_cli"
```

Fetch wellness only:

```bash
trailtraining --profile alice fetch-garmin
```

### Garmin concurrency warning

GarminDb reads one “active” config at `~/.GarminDb/GarminConnectConfig.json`. `trailtraining` switches it per profile.
Don’t run two Garmin profiles at the same time on the same machine unless you isolate HOME (container / separate OS user).

### GarminDb schema version mismatch

If GarminDb updates and you see “DB version mismatch… rebuild DB”, rebuild per-profile (see the Troubleshooting section below).

---

## Running the pipeline

### Run everything

```bash
trailtraining --profile alice run-all
```

Provider selection order:

1. `--wellness-provider auto|garmin|intervals`
2. `TRAILTRAINING_WELLNESS_PROVIDER` (or legacy `WELLNESS_PROVIDER`)
3. Auto-detect:

   * Intervals if `INTERVALS_API_KEY` is set
   * else Garmin if Garmin creds are set
   * else defaults to Intervals

Force a provider:

```bash
trailtraining --profile alice run-all --wellness-provider intervals
trailtraining --profile alice run-all --wellness-provider garmin
```

### Manual steps (debugging)

```bash
# 1) wellness
trailtraining --profile alice fetch-intervals
# or:
trailtraining --profile alice fetch-garmin

# 2) Strava
trailtraining --profile alice fetch-strava

# 3) combine into prompting/combined_summary.json (+ rollups)
trailtraining --profile alice combine
```

### Cleaning options

```bash
trailtraining --profile alice run-all --clean
trailtraining --profile alice run-all --clean-processing
trailtraining --profile alice run-all --clean-prompting
```

Notes:

* `processing/` is preserved by default so Strava can remain incremental.
* Use `--clean-processing` to force a full Strava refetch.

---

## Forecasting (readiness + overreach risk)

After `run-all` (or after `combined_summary.json` exists):

```bash
trailtraining --profile alice forecast
```

Defaults:

* Input: `<base>/prompting/combined_summary.json`
* Output: `<base>/prompting/readiness_and_risk_forecast.json`

Override:

```bash
trailtraining --profile alice forecast --input /path/to/prompting/
trailtraining --profile alice forecast --output /tmp/readiness_and_risk_forecast.json
```

---

## LLM coach (optional)

Run prompts:

```bash
trailtraining --profile alice coach --prompt training-plan
trailtraining --profile alice coach --prompt recovery-status
trailtraining --profile alice coach --prompt meal-plan
```

Files written under `<base>/prompting/`:

* `coach_brief_training-plan.json` (structured)
* `coach_brief_recovery-status.md`
* `coach_brief_meal-plan.md`

Override style:

```bash
trailtraining --profile alice coach --prompt training-plan --style triathlon
trailtraining --profile alice coach --prompt training-plan --style trailrunning
```

---

## Coach evaluation harness

Evaluate a `training-plan` JSON against safety/consistency constraints:

```bash
trailtraining eval-coach --input ~/trailtraining-data/alice/prompting/coach_brief_training-plan.json
```

Override thresholds:

```bash
trailtraining eval-coach \
  --input ~/trailtraining-data/alice/prompting/coach_brief_training-plan.json \
  --max-ramp-pct 10 \
  --max-consecutive-hard 2
```

Write outputs:

```bash
trailtraining eval-coach --input ... --output ~/violations.json
trailtraining eval-coach --input ... --report ~/full_report.json
```

---

## Outputs

Within each profile’s `TRAILTRAINING_BASE_DIR` (default: `~/trailtraining-data/<profile>/`):

* `processing/`
  Intermediate state (including incremental Strava metadata). Usually keep this.
* `prompting/`
  Combined outputs used by forecasting + coach:

  * `combined_summary.json`
  * `combined_rollups.json`
  * `readiness_and_risk_forecast.json`
  * coach outputs (plan/status/meal files)
  * `formatted_personal_data.json`

Strava token:

* `<base>/tokens/strava_token.json`

---

## Troubleshooting

### Increase logging

```bash
trailtraining --profile alice --log-level DEBUG run-all
# or:
export TRAILTRAINING_LOG_LEVEL="DEBUG"
```

### GarminDb schema rebuild (version mismatch)

Fast rebuild (no full re-download):

```bash
PROFILE=alice
GDB_HOME="$HOME/.trailtraining/garmin/$PROFILE/garmindb_home"
CLI="${GARMINGDB_CLI:-$PWD/.venv/bin/garmindb_cli.py}"

HOME="$GDB_HOME" XDG_CONFIG_HOME="$GDB_HOME/.config" XDG_CACHE_HOME="$GDB_HOME/.cache" \
  .venv/bin/python3 "$CLI" --rebuild_db

trailtraining --profile "$PROFILE" fetch-garmin
```

Full rebuild (slower, most reliable) is possible if needed.

---

## Development

```bash
pytest -q
ruff check .
ruff format .
pre-commit install
pre-commit run --all-files
```

---

## CLI reference

```bash
trailtraining -h
trailtraining doctor -h
trailtraining auth-strava -h
trailtraining fetch-strava -h
trailtraining fetch-garmin -h
trailtraining fetch-intervals -h
trailtraining combine -h
trailtraining run-all -h
trailtraining run-all-intervals -h
trailtraining forecast -h
trailtraining coach -h
trailtraining eval-coach -h
```
