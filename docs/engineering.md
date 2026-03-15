# Engineering notes

This document explains the architecture and design choices behind `trailtraining`.

The current system is no longer just a data-to-plan generator. It is now a local artifact pipeline that can:

1. ingest activity and optional recovery data
2. compute deterministic training context
3. generate structured coaching artifacts
4. evaluate those artifacts deterministically
5. optionally evaluate them qualitatively with a second model
6. revise the original plan from the evaluation report
7. re-evaluate the revised artifact

The project should be read as an engineering system, not just a prompt wrapper.

## System goal

`trailtraining` is a local-first training-planning pipeline for turning wearable and platform data into structured, inspectable, and reviewable guidance.

The goal is not to build a generic fitness chatbot.

The goal is to make collected data more useful for real training decisions by building a pipeline where:

- inputs are explicit
- intermediate artifacts are saved locally
- deterministic signals are computed before generation
- generated outputs are evaluated after generation
- plans can be revised from critique rather than accepted blindly

## Design principles

### 1. Local and inspectable

Fitness data are sensitive and generated plans should be easy to inspect.

`trailtraining` therefore writes intermediate artifacts locally rather than hiding everything behind a single opaque command. This improves:

- debugging
- iteration speed
- transparency
- prompt and forecast development
- trust in downstream outputs

### 2. Useful over flashy

Most fitness products already visualize data well enough.

The harder problem is turning that data into something actionable. The system is designed to help answer practical questions such as:

- How hard has the recent week actually been?
- Am I above recent baseline load?
- Does recovery look stable, elevated-risk, or uncertain?
- What kind of week makes sense from here?
- Does the generated plan actually hold up under review?
- Does a revised plan improve after critique?

### 3. Structured artifacts over vague prose

The pipeline prefers structured outputs with explicit fields over generic text.

A training plan artifact includes:

- metadata and primary goal
- recent snapshots
- readiness state
- weekly totals
- day-by-day sessions
- recovery actions
- risks
- data notes
- citations

That structure makes both evaluation and revision easier.

### 4. Generation should be constrained and reviewed

A plan is not trusted just because it sounds confident.

This project explicitly separates:

- deterministic forecasting
- LLM generation
- deterministic evaluation
- qualitative judging
- revision from critique

That separation is one of the most important architectural decisions in the repo.

## End-to-end pipeline

The current pipeline is:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                                INPUT LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Strava activity history                                                   │
│  GarminDB wellness data (optional)                                         │
│  Intervals.icu wellness data (optional)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGEST + LOCAL ARTIFACTS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  fetch-strava / fetch-garmin / fetch-intervals                             │
│  combine                                                                    │
│  outputs: combined_summary.json, combined_rollups.json                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETERMINISTIC SIGNAL LAYER                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  forecast                                                                   │
│  computes readiness + overreach context from recent load and recovery data │
│  output: readiness_and_risk_forecast.json                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GENERATION LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  coach --prompt training-plan                                              │
│  output: coach_brief_training-plan.json / .txt                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETERMINISTIC EVALUATION LAYER                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  eval-coach                                                                 │
│  constraint checks, scoring, violation reporting                            │
│  output: eval_report.json                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                           ▼                     ▼
┌────────────────────────────────────┐  ┌────────────────────────────────────┐
│  score / grade / violations         │  │     SOFT EVALUATION LAYER         │
│  deterministic quality signals      │  ├────────────────────────────────────┤
│                                      │  │  eval-coach --soft-eval           │
│                                      │  │  second-model judge               │
│                                      │  │  rubric scores + markers          │
└────────────────────────────────────┘  │  strengths / concerns / fixes      │
                                        └────────────────────────────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               REVISION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  revise-plan                                                                │
│  inputs: original training plan + eval_report.json                          │
│  outputs: revised-plan.json / revised-plan.txt                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RE-EVALUATION / ITERATION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  eval-coach --input revised-plan.json --soft-eval                          │
│  compare revised artifact against original                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

The key point is that generation is no longer the endpoint. The system now supports an explicit critique-and-revision loop.

## Data sources

The pipeline currently works with two main categories of input.

### Activity data

Activity history comes from Strava and acts as the core recent-load record.

This is the main source for:

- activity counts
- moving time
- distance
- elevation
- training-load rollups

### Recovery and wellness data

Recovery information can come from either:

- GarminDB
- Intervals.icu

These sources enrich the local context with recovery-related features such as resting-heart-rate trends, sleep coverage, and related telemetry.

The pipeline is intentionally tolerant of missing data. It can run in activity-only mode and becomes more informative when recent wellness data are available.

## Local artifact model

A core architectural decision in `trailtraining` is to persist intermediate state as local artifacts.

That choice creates several benefits:

- easier debugging and inspection
- easier experimentation with forecasting and prompting
- explicit boundaries between pipeline stages
- straightforward re-use of artifacts in later steps

Typical artifacts include:

```text
processing/
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

This artifact model is what makes evaluation and revision practical. The reviser does not need to regenerate the entire context from scratch; it can operate on saved plan and eval artifacts.

## Deterministic forecasting

Before any plan is generated, the system computes recent-load and recovery-aware signals.

The forecasting layer is deliberately lightweight and transparent.

### Training-load metric

The current implementation computes a simple training-load signal in training-load hours.

Conceptually:

```text
training_load_hours = moving_time_hours × load_factor
```

Where:

- `moving_time_hours` is duration in hours
- `load_factor` adjusts for relative effort when that information is available

This metric is intentionally simple. The goal is not physiological perfection. The goal is a practical, inspectable signal that is good enough to support planning and guardrails.

### Forecast outputs

The forecast layer compares recent windows to longer baselines and emits practical outputs such as:

- readiness status / score
- overreach-risk level / score
- notes about missing telemetry or uncertainty

These are operational signals used to shape prompting and later evaluation.

## Generation layer

Once local context exists, the generation layer assembles a structured prompt and produces coaching outputs.

Examples include:

- weekly training plans
- recovery summaries
- meal-planning support

The important engineering decision is that the model is asked to operate on a structured local context rather than freeform user prose alone.

For training plans, the generated artifact is a schema-constrained JSON document with a sibling human-readable `.txt` rendering.

## Deterministic evaluation layer

Generation is not treated as success.

`eval-coach` checks the generated plan against explicit deterministic constraints and produces `eval_report.json`.

Examples of what this layer is designed to catch:

- excessive ramp versus recent load
- poor spacing of hard sessions
- insufficient rest structure
- inconsistency between day details and weekly totals
- weak grounding or citation problems

This is the first review gate.

## Soft evaluation layer

The newer soft-eval path adds a second model as a qualitative judge.

This model is not generating a new plan. It is evaluating the generated plan against explicit rubrics and markers.

The soft assessment can include:

- summary
- confidence
- rubric scores
- marker-level evidence
- strengths
- concerns
- suggested improvements

### Why a second model?

Deterministic checks are good at catching structural issues.

They are less suited for questions like:

- does the plan actually feel coherent as a week?
- are the explanations specific or generic?
- does the week clearly support the athlete goal?
- is the caution tone proportionate?
- is the plan genuinely actionable?

That is where the soft judge helps.

### Why the judge/reviser uses a different model

The revision loop intentionally uses a different model family from the initial generator when possible. In this setup, GPT is used to generate the first-pass training plan, while Claude is used as the second-stage judge/reviser.

The goal is not to claim that one model is universally better. The goal is to reduce same-model anchoring and self-grading bias. A separate model is less likely to simply mirror the generator’s original reasoning, which makes critique, scoring, and revision more independent.

This does not guarantee correctness, but it is a stronger evaluation pattern than having the same model both produce and judge the plan.

### Robustness against malformed judge output

The latest version hardens this stage significantly.

The judge model does not always return a perfectly complete first-pass payload. The implementation now recovers by:

- running a marker-only repair pass if marker results are missing
- accepting incomplete top-level feedback lists and backfilling them locally
- deriving rubric scores from marker scores when rubric-level outputs are malformed
- detecting internally inconsistent outputs, such as narrative-only responses with all-zero scores

This is an important design improvement because it turns judge-model brittleness into a recoverable systems problem instead of a pipeline-breaking failure.

## Revision layer

The new `revise-plan` command is the other major addition.

It introduces a formal post-evaluation revision step.

### Inputs

`revise-plan` consumes:

- `coach_brief_training-plan.json`
- `eval_report.json`

### Outputs

It writes:

- `revised-plan.json`
- `revised-plan.txt`

### Design intent

The reviser is not meant to do arbitrary freeform rewriting.

It is prompted to:

- preserve strong parts of the original plan
- address deterministic violations and qualitative concerns
- keep the revised artifact grounded in the original signals and citations
- maintain schema compatibility
- keep weekly totals and day-level structure internally coherent

This addition changes the pipeline from one-shot generation into an iterative improvement loop.

## Why the revision loop matters

Without revision, evaluation is only advisory.

With revision, the system can do something more useful:

1. generate a first-pass plan
2. score and critique that plan
3. rewrite it using the critique as explicit input
4. check whether the revised plan improved

That closes the loop between generation and evaluation.

From an engineering perspective, this makes the project much stronger because critique is now actionable rather than decorative.

## Reliability and limitations

The system is practical, but it has important limitations.

### Data quality

Everything downstream depends on the quality and completeness of upstream activity and wellness data.

### Missing telemetry

If recent recovery metrics are sparse, confidence should be lower and interpretations should be more conservative.

### Simplified load modeling

The training-load metric is deliberately lightweight. It is useful for planning and guardrails, but it is not a complete model of fatigue, fitness, or injury risk.

### Judge output can still be messy

The soft-eval path is more robust now, but first-pass judge outputs can still be incomplete. That is partly why the repair and derivation logic exists.

### Generated and revised plans remain assistive

Outputs are decision-support artifacts, not medical advice or guaranteed coaching truth.

## Repository layout

A simplified view of the repo:

```text
.
├── demo/
├── docs/
│   ├── engineering.md
│   └── images/
├── src/trailtraining/
│   ├── llm/
│   │   ├── coach.py
│   │   ├── eval.py
│   │   ├── soft_eval.py
│   │   └── revise.py
│   └── cli.py
├── tests/
├── README.md
└── pyproject.toml
```

## Why local-first still matters

A lot of current AI fitness tooling stays shallow because it skips the systems work.

It often:

- hides context
- skips explicit evaluation
- ignores missing-data uncertainty
- produces polished text with weak operational value

`trailtraining` takes the opposite approach:

- derive context first
- generate second
- evaluate third
- revise fourth
- re-check fifth

That ordering is the architecture.

## Future directions

The current pipeline is useful, but there are clear next steps:

- richer sport-specific load modeling
- better confidence handling under sparse telemetry
- clearer comparison between original and revised plan quality
- better benchmarking of soft-eval behavior across models
- more polished demo artifacts showing the full revision loop
- additional tooling around artifact diffs and revision summaries

## Closing note

The key contribution of `trailtraining` is not just that it uses an LLM.

It is that it treats training guidance as a structured, inspectable pipeline with local artifacts, deterministic signals, explicit evaluation, and now an iterative revision loop.

That is what makes it more than a generic AI coaching demo.
