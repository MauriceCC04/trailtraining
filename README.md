### Trailrunning Training Project — User Guide

This project pulls your **training + wellness** data (**GarminDb or Intervals.icu** + **Strava**), combines it into a clean dataset, and optionally runs an LLM “coach” to generate:

- training plans
- recovery status
- meal plans

It supports **multi-user profiles** (separate Strava tokens + separate data folders per user) via `--profile`.

---

## Quickstart

```
# install (see Installation)
trailtraining --profile alice doctor

# authorize Strava once per profile
trailtraining --profile alice auth-strava

# run pipeline (auto-picks Intervals vs Garmin)
trailtraining --profile alice run-all
````

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
cd /.../Trailrunning-Training-Project

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
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
# Each user creates their own Strava API application and uses their own Client ID/Secret here.
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
OPENAI_API_KEY="..."
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

### Why per user?

In your current setup, you found that **each Strava account needs its own Strava API app credentials**. That means:

* Alice creates a Strava API application while logged into Alice’s Strava account → puts those values in `alice.env`
* Bob creates a Strava API application while logged into Bob’s Strava account → puts those values in `bob.env`

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

## LLM coach (optional)

Set in your profile env:

```bash
OPENAI_API_KEY="..."
TRAILTRAINING_LLM_MODEL="gpt-5.2"
TRAILTRAINING_REASONING_EFFORT="medium"   # none|low|medium|high|xhigh
TRAILTRAINING_VERBOSITY="medium"          # low|medium|high
TRAILTRAINING_COACH_STYLE="trailrunning"  # trailrunning|triathlon
```

Run prompts:

```bash
trailtraining --profile alice coach --prompt training-plan
trailtraining --profile alice coach --prompt recovery-status
trailtraining --profile alice coach --prompt meal-plan
```

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

## Outputs and folders

Within each profile’s `TRAILTRAINING_BASE_DIR` (default `~/trailtraining-data/<profile>/`):

* `processing/`
  Intermediate state (including Strava incremental metadata). Usually keep this.
* `prompting/`
  Combined JSON outputs used by the coach.

Strava tokens are stored at:

* `<base>/tokens/strava_token.json`

---

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

## Helpful commands

```bash
trailtraining -h
trailtraining doctor -h

trailtraining run-all -h
trailtraining fetch-intervals -h
trailtraining run-all-intervals -h

trailtraining coach -h
```