[![CI](https://github.com/MauriceCC04/trailtraining/actions/workflows/ci.yml/badge.svg)](https://github.com/MauriceCC04/trailtraining/actions/workflows/ci.yml)

### Trailrunning/Triathlon Training Project — User Guide

This project pulls your **training + wellness** data (**GarminDb or Intervals.icu** + **Strava**), combines it into a clean dataset, and optionally runs an LLM “coach”.

The coach can generate:
- training plans (structured JSON output)
- recovery status
- meal plans

It supports **multi-user profiles** (separate Strava tokens + separate data folders per user) via `--profile`.

---

## Quickstart

```bash
# install (see Installation)
trailtraining --profile alice doctor

# authorize Strava once per profile
trailtraining --profile alice auth-strava

# run pipeline (auto-picks Intervals vs Garmin)
trailtraining --profile alice run-all

# generate deterministic readiness + overreach risk (recommended)
trailtraining --profile alice forecast

# generate a structured training plan (JSON)
trailtraining --profile alice coach --prompt training-plan

# evaluate the plan against constraints (ramp + hard-day streak)
trailtraining eval-coach --input ~/trailtraining-data/alice/prompting/coach_brief_training-plan.json
```

---

## Prerequisites

* **Python 3.9+**
* A **Strava API application** (Client ID + Client Secret) **per user/profile** (see Strava setup)
* One wellness provider:

  * **GarminDb** installed and its CLI available (`garmindb_cli` or `garmindb_cli.py`), OR
  * **Intervals.icu** API access (API key + athlete ID)

Optional:

* **OpenAI API key** for the LLM coach feature

---

## Installation (macOS / Linux)

```bash
cd /.../trailtraining

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
#for intervals users:
pip install -e .
#for garmin users:
pip install -e ".[garmin]"
```

For development (tests + ruff + pre-commit):

```bash
pip install -e ".[dev]"
```

Verify the CLI:

```bash
trailtraining -h
```

---

## Profiles (multi-user support)

Profiles make it easy to run the tool for multiple people on the same machine.

### How profiles work

When you run:

```bash
trailtraining --profile alice run-all
```

the CLI will:

1. Load environment variables from:

   * `~/.trailtraining/profiles/alice.env` (if it exists)

2. Use an isolated default data directory (unless you override it):

   * `~/trailtraining-data/alice/`

3. Store Strava tokens per profile:

   * `~/trailtraining-data/alice/tokens/strava_token.json`

4. For Garmin users, write a per-profile GarminDb config:

   * `~/.trailtraining/garmin/alice/GarminConnectConfig.json`
     and make GarminDb’s active config (`~/.GarminDb/GarminConnectConfig.json`) point to the active profile config.

> The profile env file does **not** override variables you already set in your shell. Shell env vars win.

### Create profile files

Create the profile folder:

```bash
mkdir -p ~/.trailtraining/profiles
```

Create one env file per user, e.g. `~/.trailtraining/profiles/alice.env`:

```bash
# --- Required for Strava (PER USER) ---
STRAVA_CLIENT_ID="..."
STRAVA_CLIENT_SECRET="..."
STRAVA_REDIRECT_URI="http://127.0.0.1:5000/authorization"

# --- Choose ONE wellness provider ---

# Option A: Garmin
GARMIN_EMAIL="alice@example.com"
GARMIN_PASSWORD="..."

# Option B: Intervals.icu (instead of Garmin)
# INTERVALS_API_KEY="..."
# INTERVALS_ATHLETE_ID="0"   # "0" = current athlete

# Optional: force provider selection (otherwise run-all auto-detects)
# TRAILTRAINING_WELLNESS_PROVIDER="auto"   # auto|garmin|intervals
# WELLNESS_PROVIDER="intervals"            # back-compat

# Optional override: where this profile stores all data
# TRAILTRAINING_BASE_DIR="$HOME/trailtraining-data/alice"

# Optional: logging verbosity
# TRAILTRAINING_LOG_LEVEL="INFO"  # CRITICAL|ERROR|WARNING|INFO|DEBUG

# --- Optional LLM coach ---
# Prefer OPENAI_API_KEY; TRAILTRAINING_OPENAI_API_KEY also works
OPENAI_API_KEY="..."
# TRAILTRAINING_OPENAI_API_KEY="..."

TRAILTRAINING_LLM_MODEL="gpt-5.2"
TRAILTRAINING_REASONING_EFFORT="medium"   # none|low|medium|high|xhigh
TRAILTRAINING_VERBOSITY="medium"          # low|medium|high
TRAILTRAINING_COACH_STYLE="trailrunning"  # trailrunning|triathlon
```

Repeat for `bob.env`, etc.

> **Garmin concurrency warning:** GarminDb reads one “active” config at `~/.GarminDb/GarminConnectConfig.json`. The pipeline switches it per profile. Don’t run two Garmin profiles at the same time on the same machine unless you isolate HOME (container / separate OS user).

---

## Doctor (recommended)

Before doing anything else:

```bash
trailtraining --profile alice doctor
```

It checks:

* Strava env vars
* whether a Strava token exists
* wellness provider credentials
* GarminDb CLI availability (if using Garmin)
* OpenAI key presence (optional)

---

## Strava setup (required, per user)

### Create the Strava API application (for each user)

For each user (Alice, Bob, …):

1. Log into Strava with that user.
2. Create a Strava API application.
3. Copy:

   * `STRAVA_CLIENT_ID`
   * `STRAVA_CLIENT_SECRET`
4. Set the redirect URI to:

   * `http://127.0.0.1:5000/authorization`

Put those values in that user’s profile env file.

### Authorize each Strava account once per profile

```bash
trailtraining --profile alice auth-strava
trailtraining --profile bob auth-strava
```

Tip: use an **Incognito/Private** window (or log out/in) when switching between accounts so you authorize the correct Strava user.

---

## Wellness provider: GarminDb (if using Garmin)

Install GarminDb according to its docs, and make sure the CLI is on your PATH:

* `garmindb_cli` or `garmindb_cli.py`

If it isn’t on your PATH, set:

```bash
export GARMINGDB_CLI="/full/path/to/garmindb_cli"
```

You do **not** need to manually create `~/.GarminDb/GarminConnectConfig.json`.
The pipeline writes a per-profile config and activates it automatically.

Additionally:
### GarminDb schema updates (version mismatch)

If GarminDb was updated and you see an error like:

  "DB: <name> version mismatch... Please rebuild the <name> DB"

Rebuilds are **per profile**, because trailtraining isolates GarminDb HOME at:
  ~/.trailtraining/garmin/<profile>/garmindb_home
and writes per-profile config automatically.

#### Option 1: Rebuild DB schema (fast, no full re-download)

```bash
PROFILE=alice
GDB_HOME="$HOME/.trailtraining/garmin/$PROFILE/garmindb_home"
CLI="${GARMINGDB_CLI:-$PWD/.venv/bin/garmindb_cli.py}"

HOME="$GDB_HOME" XDG_CONFIG_HOME="$GDB_HOME/.config" XDG_CACHE_HOME="$GDB_HOME/.cache" \
  .venv/bin/python3 "$CLI" --rebuild_db

# then rerun:
trailtraining --profile "$PROFILE" fetch-garmin
```
#### Option 2: Full rebuild (slow, but most reliable)
```bash
BASE_DIR="${TRAILTRAINING_BASE_DIR:-$HOME/trailtraining-data/$PROFILE}"
BACKUP="$BASE_DIR/db_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"

find "$BASE_DIR" "$GDB_HOME" -maxdepth 8 -type f \( -iname "garmin*.db" -o -iname "garmin*.sqlite" \) \
  -exec mv -v {} "$BACKUP"/ \; 2>/dev/null

HOME="$GDB_HOME" XDG_CONFIG_HOME="$GDB_HOME/.config" XDG_CACHE_HOME="$GDB_HOME/.cache" \
  .venv/bin/python3 "$CLI" --all --download --import --analyze
```
Fetch wellness only:

```bash
trailtraining --profile alice fetch-garmin
```

---

## Wellness provider: Intervals.icu (faster)

If you use Intervals.icu, set these in your profile env instead of Garmin credentials:

```bash
INTERVALS_API_KEY="..."
INTERVALS_ATHLETE_ID="0"
```

Optional date range (env):

```bash
TRAILTRAINING_WELLNESS_OLDEST="2023-01-01"
TRAILTRAINING_WELLNESS_NEWEST="2026-02-27"
```

Fetch wellness only:

```bash
trailtraining --profile alice fetch-intervals --oldest "2023-01-01" --newest "2026-02-27"
```

Notes:

* `--oldest` defaults to a lookback window if omitted.
* `--newest` defaults to “today” if omitted.

---

## Running the full pipeline

### Run everything (auto-selects Intervals vs Garmin)

```bash
trailtraining --profile alice run-all
```

Provider selection order:

1. `--wellness-provider auto|garmin|intervals`
2. `TRAILTRAINING_WELLNESS_PROVIDER` or `WELLNESS_PROVIDER`
3. Auto-detect:

   * Intervals if `INTERVALS_API_KEY` is set
   * else Garmin if Garmin creds are set
   * else defaults to Intervals

Force a provider:

```bash
trailtraining --profile alice run-all --wellness-provider intervals
trailtraining --profile alice run-all --wellness-provider garmin
```

### Manual pipeline (debugging)

If you want to run steps one-by-one:

```bash
# 1) wellness
trailtraining --profile alice fetch-intervals --oldest "2023-01-01" --newest "2026-02-27"
# OR:
trailtraining --profile alice fetch-garmin

# 2) Strava
trailtraining --profile alice fetch-strava

# 3) Combine into prompting/combined_summary.json (+ rollups)
trailtraining --profile alice combine
```

### Intervals-only pipeline (legacy alias)

```bash
trailtraining --profile alice run-all-intervals
```

### Cleaning options

```bash
trailtraining --profile alice run-all --clean
trailtraining --profile alice run-all --clean-processing
trailtraining --profile alice run-all --clean-prompting
```

Notes:

* By default, `processing/` is preserved so Strava can remain **incremental**.
* Use `--clean-processing` if you want to force a full Strava refetch.

---

## Forecasting (readiness + overreach risk)

After `run-all` (or after you’ve produced `prompting/combined_summary.json`), you can generate a deterministic
**readiness** and **overreach risk** assessment:

```bash
trailtraining --profile alice forecast
```

### Inputs / outputs

Defaults:

* **Input**: the profile’s `prompting/combined_summary.json`
* **Output**: `prompting/readiness_and_risk_forecast.json`

Point at a specific prompting directory (the directory that contains `combined_summary.json`):

```bash
trailtraining --profile alice forecast --input /path/to/prompting/
```

Write the JSON somewhere else:

```bash
trailtraining --profile alice forecast --output /tmp/readiness_and_risk_forecast.json
```

### What’s inside `readiness_and_risk_forecast.json`

The file contains:

* `generated_at` (UTC timestamp)
* `result.date`
* `result.readiness.score` (0–100) and `result.readiness.status` (`primed` | `steady` | `fatigued`)
* `result.overreach_risk.score` (0–100) and `result.overreach_risk.level` (`low` | `moderate` | `high`)
* `result.inputs` (underlying aggregates, like recent RHR and training load)
* `result.drivers` (human-readable reasons for the scores)

### How the coach uses it

If `readiness_and_risk_forecast.json` exists in the prompting directory, the coach loads it and treats it as
**authoritative** for readiness/risk signals. If it doesn’t exist, the coach will attempt (best-effort) to compute
and save it.

---

## LLM coach (optional)

Set in your profile env:

```bash
OPENAI_API_KEY="..."
TRAILTRAINING_LLM_MODEL="gpt-5.2"
TRAILTRAINING_REASONING_EFFORT="medium"   # none|low|medium|high|xhigh
TRAILTRAINING_VERBOSITY="medium"          # low|medium|high
TRAILTRAINING_COACH_STYLE="trailrunning"  # trailrunning|triathlon
```

For `training-plan`, the coach:

* uses **retrieved history** (weekly summaries for the last N weeks)
* uses a **signal registry** and must cite `signal_ids` that justify recommendations
* returns **structured JSON** (machine readable)

Control retrieval / prompt size (env):

```bash
TRAILTRAINING_COACH_RETRIEVAL_WEEKS="8"   # weekly history window
TRAILTRAINING_COACH_DETAIL_DAYS="14"      # number of recent daily blocks included
TRAILTRAINING_COACH_DAYS="60"             # how many combined_summary days to consider
TRAILTRAINING_COACH_MAX_CHARS="200000"    # max prompt text length budget
```

### Run prompts

```bash
trailtraining --profile alice coach --prompt training-plan
trailtraining --profile alice coach --prompt recovery-status
trailtraining --profile alice coach --prompt meal-plan
```

* `training-plan` saves: `coach_brief_training-plan.json`
* `recovery-status` saves: `coach_brief_recovery-status.md`
* `meal-plan` saves: `coach_brief_meal-plan.md`

Override sport style on the CLI:

```bash
trailtraining --profile alice coach --prompt training-plan --style triathlon
trailtraining --profile bob   coach --prompt training-plan --style trailrunning
```

Point at a specific prompting directory:

```bash
trailtraining --profile alice coach --prompt recovery-status --input /path/to/prompting/
```

---

## Coach evaluation harness (recommended)

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

Env defaults:

```bash
TRAILTRAINING_MAX_RAMP_PCT="10"
TRAILTRAINING_MAX_CONSEC_HARD="2"
```

Optional: write violations to a JSON file:

```bash
trailtraining eval-coach \
  --input ~/trailtraining-data/alice/prompting/coach_brief_training-plan.json \
  --output ~/violations.json
```

Optional: write the full scoring report:

```bash
trailtraining eval-coach \
  --input ~/trailtraining-data/alice/prompting/coach_brief_training-plan.json \
  --report ~/full_report.json
```

---

## Outputs and folders

Within each profile’s `TRAILTRAINING_BASE_DIR` (default `~/trailtraining-data/<profile>/`):

* `processing/`
  Intermediate state (including Strava incremental metadata). Usually keep this.
* `prompting/`
  Combined JSON outputs used by the coach:

  * `combined_summary.json`
  * `combined_rollups.json`
  * `readiness_and_risk_forecast.json`
  * `coach_brief_training-plan.json` (structured)
  * `coach_brief_training-plan.txt`
  * `coach_brief_recovery-status.md`
  * `coach_brief_meal-plan.md`
  * `formatted_personal_data.json`
Additionally, when Garmin is chosen as the wellness provider, the following are also stored:
  * `shortened_rhr.json`
  * `shortened_sleep.json`

Strava tokens are stored at:

* `<base>/tokens/strava_token.json`

---
Sample outputs are stored in \demo

## Troubleshooting

Increase logging verbosity:

```bash
trailtraining --profile alice --log-level DEBUG run-all
```

Or set:

```bash
export TRAILTRAINING_LOG_LEVEL="DEBUG"
```

Run doctor:

```bash
trailtraining --profile alice doctor
```

---

## Development

Run tests:

```bash
pytest -q
```

Lint / format:

```bash
ruff check .
ruff format .
```

Pre-commit (optional):

```bash
pre-commit install
pre-commit run --all-files
```

---

## Helpful commands (all CLI commands)

```bash
# global help
trailtraining -h

# each subcommand help
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

