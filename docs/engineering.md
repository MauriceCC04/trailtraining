# Engineering notes

This document explains the main design choices behind `trailtraining` and how the current pipeline works.

## Overview

`trailtraining` is a local-first pipeline for turning wearable and training-platform data into structured training guidance.

At a high level, the system:

1. pulls activity and recovery data from external sources,
2. combines them into a local unified dataset,
3. derives a simple training-load metric and recovery context,
4. computes readiness and overreach-related signals,
5. uses that context to generate structured coaching outputs,
6. evaluates generated plans against rule-based constraints.

The goal is not to build a generic fitness chatbot. The goal is to make collected data more useful for actual training decisions in a way that remains inspectable.

## Design goals

The project is built around a few core principles:

### 1. Local and inspectable
Fitness and recovery data are sensitive, and generated plans should be understandable. The pipeline keeps intermediate artifacts local and makes them easy to inspect.

### 2. Useful over flashy
Most training products already visualize data. The harder problem is turning that data into something actionable. The system is designed to produce outputs that help answer practical questions such as:

- How hard has the recent week actually been?
- Does the current training load look elevated versus baseline?
- Does recovery look stable or uncertain?
- What kind of week is reasonable from here?

### 3. Structured outputs over vague summaries
Rather than producing generic text, the pipeline tries to generate structured plans with explicit rationale, weekly totals, day-by-day sessions, and cautions.

### 4. LLM outputs should be constrained and reviewable
A generated training plan should not be treated as correct just because it sounds confident. The project therefore includes rule-based evaluation to make outputs easier to review and less likely to drift into obviously poor recommendations.

## System overview

The top-level workflow is:

- collect activity data from Strava,
- collect recovery / wellness data from GarminDB or Intervals.icu,
- merge them into local summary artifacts,
- derive training-load and recovery context,
- forecast readiness / overreach-related state,
- generate coaching outputs,
- evaluate generated plans.

The README contains the higher-level system diagram. This page focuses on the engineering logic behind that pipeline.

## Data sources

The current pipeline is built around two types of input:

### Activity data
Activity history comes from Strava and provides the core training record used for recent-load calculations and context building.

### Recovery / wellness data
Recovery information comes from either:

- GarminDB, or
- Intervals.icu

These sources are used to enrich the local context with recovery-related signals such as resting-heart-rate trends and related recent telemetry.

## Local artifacts

A core design choice in `trailtraining` is to write intermediate outputs locally rather than hide the entire pipeline behind a single opaque command.

This has a few benefits:

- easier debugging,
- easier inspection of intermediate state,
- simpler iteration on forecasting and prompting,
- clearer separation between ingestion, forecasting, and generation.

Representative outputs are included in the `demo/` directory.

## Training-load metric

The forecasting layer is built around a simple derived training-load metric.

The current implementation computes daily training load in **training-load hours**, using activity duration and a lightweight intensity adjustment.

Conceptually:

```text
training_load_hours = moving_time_hours × load_factor
```

Where:

* `moving_time_hours` is the activity duration in hours,
* `load_factor` increases or decreases the load estimate based on relative effort.

When heart-rate information is available, the load factor is derived from intensity information such as average HR relative to max HR. When that is not available, the calculation falls back to a neutral default rather than pretending to know more than the data supports.

This is intentionally simple. The point is not to claim lab-grade physiological modeling. The point is to create a practical recent-load signal that is good enough to support planning and guardrail logic.

## Forecasting logic

The forecast layer combines recent training load and recent recovery signals.

The main ideas are:

### 1. Recent load

The system computes a recent 7-day training-load total and compares it to a rolling baseline from prior windows.

This creates a simple notion of:

* acute recent load,
* historical baseline load,
* whether the athlete is currently above or near their usual level.

### 2. Resting-heart-rate context

The system also compares short-window and longer-window resting-heart-rate behavior.

Conceptually, it looks at:

* recent 7-day resting HR,
* longer 28-day resting HR baseline,
* variability in the longer baseline.

This helps detect when recent recovery-related signals look elevated relative to normal.

### 3. Readiness and overreach context

Those inputs are then combined into two practical outputs:

* a readiness-oriented score / status,
* an overreach-risk-oriented score / level.

These are not intended as medical judgments or precise physiological truths. They are operational signals used to shape the generated plan.

In practice, this means:

* elevated recent load can reduce readiness and increase risk,
* elevated recent resting HR versus baseline can also reduce readiness and increase risk,
* missing telemetry should reduce confidence and encourage more conservative interpretation.

## Why use a simple training-load metric?

The training-load logic is deliberately lightweight.

That is a design choice, not an omission.

There are three reasons for this:

### Practicality

The project is meant to work with real consumer fitness data, which is often incomplete or inconsistent.

### Transparency

A simpler metric is easier to inspect, explain, and debug than a more complex black-box score.

### Extensibility

The current metric provides a clean baseline that can later be extended with more nuanced intensity handling, sport-specific adjustments, or richer physiology-informed scoring.

## Coaching generation

Once the pipeline has:

* combined activity and recovery data,
* computed recent load context,
* derived forecast outputs,

that context is used to assemble a structured coaching prompt.

The generated outputs are intended to be more useful than generic “AI insights.” They may include:

* weekly training plans,
* recovery summaries,
* meal-planning support,
* risk / caution notes,
* rationale tied to recent load and recovery state.

The important design point is that the model is not asked to invent advice from nowhere. It is asked to operate on a structured local context that already includes computed signals.

## Rule-based evaluation

One of the most important parts of the project is that generation is not the end of the pipeline.

Generated plans can be checked against explicit rules and constraints.

This matters because LLM-generated plans can sound plausible while still being poor in practice. A plan should be reviewable not only by reading the prose, but also by checking whether it violates obvious guardrails.

Examples of the kinds of concerns this layer is meant to catch include:

* too much intensity in a risky week,
* poor spacing of hard sessions,
* excessive load ramp,
* insufficient regard for uncertain recovery state.

This makes the system more useful as a decision-support tool rather than just a text generator.

## Why local-first matters

A lot of current “AI fitness” features are shallow because they skip most of the actual systems work.

They often:

* do not expose the data context clearly,
* do not explain why the recommendation was made,
* do not distinguish between strong and weak signal quality,
* do not apply meaningful constraints.

`trailtraining` takes the opposite approach:

* keep data and artifacts local,
* derive explicit context first,
* generate second,
* evaluate third.

That ordering is important.

## Reliability and limitations

The project is intentionally practical, but it has real limitations.

### Data quality

Everything downstream depends on the quality and completeness of upstream activity and recovery data.

### Missing telemetry

If recent recovery metrics are sparse or missing, confidence should be lower. The system should be interpreted more conservatively in those cases.

### Simplified load modeling

The current training-load metric is intentionally simple. It is useful as an operational signal, but it is not a complete model of fatigue, fitness, or injury risk.

### Generated plans remain assistive

The outputs are meant to support decision-making, not replace judgment.

## Why this project exists

The motivation behind `trailtraining` is simple:

* people already collect large amounts of training and wearable data,
* most platforms do very little with it that is genuinely useful,
* the first wave of LLM integrations in fitness products has often been generic and weak.

This project is an attempt to build something better for real training use: a system that turns collected data into structured, inspectable, and more actionable guidance.

## Future directions

There are several obvious directions for extension:

* better sport-specific load modeling,
* richer recovery features,
* more formal evaluation of generated plans,
* clearer confidence handling when telemetry is incomplete,
* stronger benchmarking between generated outputs and hand-written planning heuristics.

But even in its current form, the core idea is already useful: unify the data, derive recent-load context, generate a plan from that context, and evaluate the result.

## Repository layout

A simplified view of the repo:

```text
.
├── demo/
├── docs/
│   ├── engineering.md
│   └── images/
├── src/trailtraining/
├── tests/
├── README.md
└── pyproject.toml
```

## Closing note

The project should be read as an engineering system, not just a prompt wrapper.

The key contribution is the pipeline:

* local ingestion,
* unified artifacts,
* recent-load modeling,
* recovery-aware forecasting,
* structured generation,
* rule-based evaluation.

That is what makes `trailtraining` more than a generic “AI coaching” demo.
