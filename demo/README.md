# Demo artifacts

This folder contains example outputs from `trailtraining` and is meant to show what the pipeline produces after local data is ingested, combined, forecasted, and passed through the coaching layer.

These artifacts are examples, not ground truth. Their purpose is to make the pipeline inspectable: you can see the merged context, the generated plan, and the advisory outputs that were produced from that context.

## Files

- `rollups/combined_rollups.json`
  Example merged rollup combining recent activity data with any available recovery telemetry. This is the main local context used by downstream forecasting and coaching steps.

- `plans/training-plan.json`
  Example structured weekly training plan generated from the available local context.

- `plans/training-plan.txt`
  Human-readable rendering of the structured weekly training plan.

- `status/recovery-status.md`
  Example lightweight recovery summary based on the available recent context.

- `status/meal-plan.md`
  Example meal-planning output based primarily on recent activity level and training context.

## Notes

`trailtraining` is designed to work with incomplete recovery data. Some demo artifacts may reflect richer recovery context than others, depending on what signals were available at generation time.

The structured training plan is the most constrained output. Advisory outputs like recovery status and meal planning are intentionally lighter-weight and should be interpreted as supportive guidance rather than strict prescriptions.
