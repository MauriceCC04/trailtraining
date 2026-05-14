from __future__ import annotations

from typing import Any

from trailtraining.contracts import (
    MachinePlanArtifact,
    PlanExplanationArtifact,
    TrainingPlanArtifact,
)

_SNAPSHOT_KEYS = [
    "distance_km",
    "moving_time_hours",
    "elevation_m",
    "activity_count",
    "sleep_hours_mean",
    "hrv_mean",
    "rhr_mean",
]


def ensure_training_plan_shape(obj: Any) -> dict[str, Any]:
    return TrainingPlanArtifact.model_validate(obj).model_dump(mode="python")


def ensure_machine_plan_shape(obj: Any) -> dict[str, Any]:
    return MachinePlanArtifact.model_validate(obj).model_dump(mode="python")


def ensure_plan_explanation_shape(obj: Any) -> dict[str, Any]:
    return PlanExplanationArtifact.model_validate(obj).model_dump(mode="python")


def _snapshot_obj_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_SNAPSHOT_KEYS),
        "properties": {
            "distance_km": {"type": "string"},
            "moving_time_hours": {"type": "string"},
            "elevation_m": {"type": "string"},
            "activity_count": {"type": "string"},
            "sleep_hours_mean": {"type": "string"},
            "hrv_mean": {"type": "string"},
            "rhr_mean": {"type": "string"},
        },
    }


_EFFECTIVE_CONSTRAINTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "allowed_week1_hours",
        "effective_max_ramp_pct",
        "effective_max_hard_per_7d",
        "effective_max_consecutive_hard",
        "min_rest_per_7d",
        "readiness_status",
        "overreach_risk_level",
        "recovery_capability_key",
        "lifestyle_notes",
        "reasons",
    ],
    "properties": {
        "allowed_week1_hours": {"type": ["number", "null"]},
        "effective_max_ramp_pct": {"type": "number"},
        "effective_max_hard_per_7d": {"type": "integer"},
        "effective_max_consecutive_hard": {"type": "integer"},
        "min_rest_per_7d": {"type": "integer"},
        "readiness_status": {"type": ["string", "null"]},
        "overreach_risk_level": {"type": ["string", "null"]},
        "recovery_capability_key": {"type": ["string", "null"]},
        "lifestyle_notes": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
}

_CITATION_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["citation_id", "signal_id", "source", "date_range", "value"],
    "properties": {
        "citation_id": {"type": "string"},
        "signal_id": {"type": "string"},
        "source": {"type": "string"},
        "date_range": {"type": "string"},
        "value": {"type": "string"},
    },
}

_CLAIM_ATTRIBUTION_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim_id",
        "field_path",
        "claim_text",
        "signal_ids",
        "citation_ids",
        "support_level",
    ],
    "properties": {
        "claim_id": {"type": "string"},
        "field_path": {"type": "string"},
        "claim_text": {"type": "string"},
        "signal_ids": {"type": "array", "items": {"type": "string"}},
        "citation_ids": {"type": "array", "items": {"type": "string"}},
        "support_level": {
            "type": "string",
            "enum": ["supported", "weak", "unsupported"],
        },
    },
}

_WEEKLY_TOTALS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "planned_distance_km",
        "planned_moving_time_hours",
        "planned_elevation_m",
    ],
    "properties": {
        "planned_distance_km": {"type": ["number", "null"], "minimum": 0},
        "planned_moving_time_hours": {"type": "number", "minimum": 0},
        "planned_elevation_m": {"type": ["number", "null"], "minimum": 0},
    },
}

_MACHINE_DAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "date",
        "session_type",
        "is_rest_day",
        "is_hard_day",
        "duration_minutes",
        "target_intensity",
        "terrain",
        "workout",
        "estimated_distance_km",
        "estimated_elevation_m",
    ],
    "properties": {
        "date": {"type": "string"},
        "session_type": {
            "type": "string",
            "enum": [
                "rest",
                "easy",
                "aerobic",
                "long",
                "tempo",
                "intervals",
                "hills",
                "strength",
                "cross",
            ],
        },
        "is_rest_day": {"type": "boolean"},
        "is_hard_day": {"type": "boolean"},
        "duration_minutes": {"type": "integer", "minimum": 0, "maximum": 420},
        "target_intensity": {"type": "string"},
        "terrain": {"type": "string"},
        "workout": {"type": "string"},
        "estimated_distance_km": {"type": ["number", "null"], "minimum": 0},
        "estimated_elevation_m": {"type": ["number", "null"], "minimum": 0},
    },
}

_TRAINING_DAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "date",
        "title",
        "session_type",
        "is_rest_day",
        "is_hard_day",
        "duration_minutes",
        "target_intensity",
        "terrain",
        "workout",
        "purpose",
        "signal_ids",
        "estimated_distance_km",
        "estimated_elevation_m",
    ],
    "properties": {
        "date": {"type": "string", "description": "YYYY-MM-DD"},
        "title": {"type": "string"},
        "session_type": {
            "type": "string",
            "enum": [
                "rest",
                "easy",
                "aerobic",
                "long",
                "tempo",
                "intervals",
                "hills",
                "strength",
                "cross",
            ],
        },
        "is_rest_day": {"type": "boolean"},
        "is_hard_day": {"type": "boolean"},
        "duration_minutes": {"type": "integer", "minimum": 0, "maximum": 420},
        "target_intensity": {"type": "string"},
        "terrain": {"type": "string"},
        "workout": {"type": "string"},
        "purpose": {"type": "string"},
        "signal_ids": {"type": "array", "items": {"type": "string"}},
        "estimated_distance_km": {"type": ["number", "null"], "minimum": 0},
        "estimated_elevation_m": {"type": ["number", "null"], "minimum": 0},
    },
}

MACHINE_PLAN_SCHEMA: dict[str, Any] = {
    "name": "trailtraining_machine_plan_v2",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["meta", "readiness", "plan"],
        "properties": {
            "meta": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "today",
                    "plan_start",
                    "plan_days",
                    "style",
                    "primary_goal",
                    "lifestyle_notes",
                ],
                "properties": {
                    "today": {"type": "string"},
                    "plan_start": {"type": "string"},
                    "plan_days": {"type": "integer", "minimum": 1, "maximum": 28},
                    "style": {"type": "string"},
                    "primary_goal": {"type": "string"},
                    "lifestyle_notes": {"type": "string"},
                },
            },
            "readiness": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status"],
                "properties": {
                    "status": {"type": "string", "enum": ["primed", "steady", "fatigued"]},
                },
            },
            "plan": {
                "type": "object",
                "additionalProperties": False,
                "required": ["days", "weekly_totals"],
                "properties": {
                    "weekly_totals": _WEEKLY_TOTALS_SCHEMA,
                    "days": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 28,
                        "items": _MACHINE_DAY_SCHEMA,
                    },
                },
            },
        },
    },
}

PLAN_EXPLANATION_SCHEMA: dict[str, Any] = {
    "name": "trailtraining_plan_explanation_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "snapshot",
            "readiness_rationale",
            "readiness_signal_ids",
            "day_explanations",
            "recovery",
            "risks",
            "data_notes",
            "citations",
            "claim_attributions",
        ],
        "properties": {
            "snapshot": {
                "type": "object",
                "additionalProperties": False,
                "required": ["last7", "baseline28", "notes"],
                "properties": {
                    "last7": _snapshot_obj_schema(),
                    "baseline28": _snapshot_obj_schema(),
                    "notes": {"type": "string"},
                },
            },
            "readiness_rationale": {"type": "string"},
            "readiness_signal_ids": {"type": "array", "items": {"type": "string"}},
            "day_explanations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["date", "title", "purpose", "signal_ids"],
                    "properties": {
                        "date": {"type": "string"},
                        "title": {"type": "string"},
                        "purpose": {"type": "string"},
                        "signal_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "recovery": {
                "type": "object",
                "additionalProperties": False,
                "required": ["actions", "signal_ids"],
                "properties": {
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "signal_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
            "risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["severity", "message", "signal_ids"],
                    "properties": {
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "message": {"type": "string"},
                        "signal_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "data_notes": {"type": "array", "items": {"type": "string"}},
            "citations": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
            "claim_attributions": {
                "type": "array",
                "items": _CLAIM_ATTRIBUTION_ITEM_SCHEMA,
            },
        },
    },
}

TRAINING_PLAN_SCHEMA: dict[str, Any] = {
    "name": "trailtraining_training_plan_v4",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "meta",
            "snapshot",
            "readiness",
            "plan",
            "recovery",
            "risks",
            "data_notes",
            "citations",
            "claim_attributions",
            "effective_constraints",
        ],
        "properties": {
            "meta": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "today",
                    "plan_start",
                    "plan_days",
                    "style",
                    "primary_goal",
                    "lifestyle_notes",
                ],
                "properties": {
                    "today": {"type": "string", "description": "YYYY-MM-DD"},
                    "plan_start": {"type": "string", "description": "YYYY-MM-DD"},
                    "plan_days": {"type": "integer", "minimum": 1, "maximum": 28},
                    "style": {"type": "string"},
                    "primary_goal": {"type": "string"},
                    "lifestyle_notes": {
                        "type": "string",
                        "description": (
                            "Athlete schedule or lifestyle constraints that affect session placement. "
                            "Copy exactly from the prompt if provided, or empty."
                        ),
                    },
                },
            },
            "snapshot": {
                "type": "object",
                "additionalProperties": False,
                "required": ["last7", "baseline28", "notes"],
                "properties": {
                    "last7": _snapshot_obj_schema(),
                    "baseline28": _snapshot_obj_schema(),
                    "notes": {"type": "string"},
                },
            },
            "readiness": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status", "rationale", "signal_ids"],
                "properties": {
                    "status": {"type": "string", "enum": ["primed", "steady", "fatigued"]},
                    "rationale": {"type": "string"},
                    "signal_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
            "plan": {
                "type": "object",
                "additionalProperties": False,
                "required": ["days", "weekly_totals"],
                "properties": {
                    "weekly_totals": _WEEKLY_TOTALS_SCHEMA,
                    "days": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 28,
                        "items": _TRAINING_DAY_SCHEMA,
                    },
                },
            },
            "recovery": {
                "type": "object",
                "additionalProperties": False,
                "required": ["actions", "signal_ids"],
                "properties": {
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "signal_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
            "risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["severity", "message", "signal_ids"],
                    "properties": {
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "message": {"type": "string"},
                        "signal_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "data_notes": {"type": "array", "items": {"type": "string"}},
            "citations": {"type": "array", "items": _CITATION_ITEM_SCHEMA},
            "claim_attributions": {
                "type": "array",
                "items": _CLAIM_ATTRIBUTION_ITEM_SCHEMA,
            },
            "effective_constraints": _EFFECTIVE_CONSTRAINTS_SCHEMA,
        },
    },
}


def training_plan_output_contract_text() -> str:
    return (
        "Output MUST be a single JSON object (no Markdown, no backticks) matching the training-plan schema.\n"
        "Rules:\n"
        "- meta.primary_goal MUST match the authoritative primary goal provided in the prompt.\n"
        '- meta.lifestyle_notes MUST copy the lifestyle constraints from the prompt exactly, or be empty string "" if none were provided.\n'
        "- meta.plan_days MUST equal the number of days in plan.days.\n"
        "- Use only signal_ids that appear in the provided Signal registry.\n"
        "- Every plan day MUST include signal_ids justifying that day.\n"
        "- readiness.signal_ids MUST justify readiness.\n"
        "- citations MUST list the signal_ids you used (dedup ok).\n"
        "- citations[].citation_id MUST be present and unique within the artifact.\n"
        '- snapshot.last7 and snapshot.baseline28 MUST include all keys; use empty string "" if unknown.\n'
        "- If data is missing, write it in data_notes; do NOT fabricate.\n"
        "- weekly_totals MUST reflect WEEK 1 values only (first 7 days).\n"
        "- planned_distance_km and planned_elevation_m MUST be null unless every non-rest day in week 1 includes matching estimated_distance_km / estimated_elevation_m values.\n"
        "- Structured fields are authoritative: title/workout text MUST agree with session_type, is_rest_day, is_hard_day, and duration_minutes.\n"
    )


def machine_plan_output_contract_text() -> str:
    return (
        "Output MUST be a single JSON object matching the machine-plan schema.\n"
        "This pass is planning only.\n"
        "- meta.primary_goal MUST match the prompt.\n"
        "- meta.lifestyle_notes MUST copy the prompt value exactly.\n"
        "- meta.plan_days MUST equal the number of days in plan.days.\n"
        "- Satisfy all hard constraints numerically.\n"
        "- weekly_totals MUST reflect week 1 only.\n"
        "- planned_distance_km and planned_elevation_m MUST be null unless every non-rest day in week 1 includes matching estimates.\n"
    )


def plan_explanation_output_contract_text() -> str:
    return (
        "Output MUST be a single JSON object matching the plan-explanation schema.\n"
        "This pass is explanation only.\n"
        "- Do not change the locked machine plan.\n"
        "- Every textual explanation claim must have claim-level attribution.\n"
        "- citations must reference only provided signal_ids.\n"
        "- If support is weak or missing, say so explicitly.\n"
        "- Every citation MUST include citation_id.\n"
    )
