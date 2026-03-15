# Demo artifacts

This folder contains representative outputs from `trailtraining` and is meant to make the full pipeline easy to inspect without needing your own API keys, wearable accounts, or local setup.

The demo now reflects the full artifact loop:

**roll up data → generate plan → evaluate plan → revise plan → render human-readable outputs**

These files are examples, not prescriptions or ground truth. Their purpose is to show what the system produces at each stage and how later artifacts build on earlier ones.

1. `trailtraining` works from **local structured context**, not a freeform chatbot prompt.
2. generated plans are **evaluated instead of trusted blindly**.
3. evaluation can include both **deterministic checks** and an optional **second-model soft assessment**.
4. plans can be **revised from evaluator feedback** and rendered again in a human-readable format.

## Artifact flow

```text
combined_rollups.json
        │
        ▼
coach_brief_training-plan.json
        │
        ├──► coach_brief_training-plan.txt
        │
        ▼
eval_report.json
        │
        ▼
revised-plan.json
        │
        └──► revised-plan.txt
````

The first-pass plan is generated from local context.
The evaluation report critiques that plan.
The revised plan incorporates that feedback while preserving the useful parts of the original.
The `.txt` files are human-readable renderings of the structured JSON artifacts.

## Files

### `rollups/combined_rollups.json`

Example merged rollup combining recent activity data with any available recovery telemetry.

This is the main local context used by downstream forecasting, training-plan generation, and plan revision. It contains the recent windows and summary values that later stages cite and reason from.

### `plans/coach_brief_training-plan.json`

Example first-pass structured weekly training plan generated from the available local context.

This is the primary machine-readable training-plan artifact. It includes:

* metadata
* recent snapshot context
* readiness
* weekly totals
* day-by-day sessions
* recovery actions
* risks
* data notes
* citations

### `plans/coach_brief_training-plan.txt`

Human-readable rendering of the first-pass structured training plan.

### `eval/eval_report.json`

Example evaluation report for the first-pass training plan.

This artifact shows how `trailtraining` critiques generated plans. Depending on configuration, it can include:

* deterministic scoring and violations
* per-category scoring
* an optional soft assessment from a second LLM judge
* qualitative strengths, concerns, and suggested improvements

This file is the direct input to the revision step.

### `plans/revised-plan.json`

Example revised version of the original training plan.

### `plans/revised-plan.txt`

Human-readable rendering of the revised training plan.

### `status/coach_brief_recovery-status.md`

This is a lighter advisory artifact than the training plan and is meant to provide quick interpretive guidance rather than a full structured plan.

### `status/coach_brief_meal-plan.md`

Example meal-planning output generated from recent activity and training context.
