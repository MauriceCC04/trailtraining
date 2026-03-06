# src/trailtraining/llm/schemas.py
from __future__ import annotations

from typing import Any, Dict


TRAINING_PLAN_SCHEMA: Dict[str, Any] = {
    "name": "trailtraining_training_plan_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["meta", "snapshot", "readiness", "plan", "recovery", "risks", "data_notes", "citations"],
        "properties": {
            "meta": {
                "type": "object",
                "additionalProperties": False,
                "required": ["today", "plan_start", "plan_days", "style"],
                "properties": {
                    "today": {"type": "string", "description": "YYYY-MM-DD"},
                    "plan_start": {"type": "string", "description": "YYYY-MM-DD"},
                    "plan_days": {"type": "integer", "minimum": 1, "maximum": 21},
                    "style": {"type": "string"},
                },
            },
            "snapshot": {
                "type": "object",
                "additionalProperties": False,
                "required": ["last7", "baseline28", "notes"],
                "properties": {
                    "last7": {"type": "object"},
                    "baseline28": {"type": "object"},
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
                    "weekly_totals": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["planned_distance_km", "planned_moving_time_hours", "planned_elevation_m"],
                        "properties": {
                            "planned_distance_km": {"type": "number", "minimum": 0},
                            "planned_moving_time_hours": {"type": "number", "minimum": 0},
                            "planned_elevation_m": {"type": "number", "minimum": 0},
                        },
                    },
                    "days": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 21,
                        "items": {
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
                            },
                        },
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
            "citations": {
                "type": "array",
                "description": "Citations to the signal registry. Include only signal_ids present in the registry.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["signal_id", "source", "date_range", "value"],
                    "properties": {
                        "signal_id": {"type": "string"},
                        "source": {"type": "string"},
                        "date_range": {"type": "string"},
                        "value": {},
                    },
                },
            },
        },
    },
}


def training_plan_output_contract_text() -> str:
    # Appended to prompt so even without schema-mode the model is pushed into strict JSON.
    return (
        "Output MUST be a single JSON object (no Markdown, no backticks) matching the training-plan schema.\n"
        "Rules:\n"
        "- Use only signal_ids that appear in the provided Signal registry.\n"
        "- Every plan day MUST include signal_ids justifying that day.\n"
        "- readiness.signal_ids MUST justify readiness.\n"
        "- citations MUST list the signal_ids you used (dedup ok).\n"
        "- If data is missing, write it in data_notes; do NOT fabricate.\n"
    )


def _require(obj: Dict[str, Any], key: str, typ: Any) -> Any:
    if key not in obj:
        raise ValueError(f"Missing required key: {key}")
    v = obj[key]
    if not isinstance(v, typ):
        raise ValueError(f"Key {key} must be {typ}, got {type(v)}")
    return v


def ensure_training_plan_shape(obj: Any) -> Dict[str, Any]:
    """
    Lightweight validation (no external jsonschema dependency).
    Raises ValueError if critical structure is missing.
    """
    if not isinstance(obj, dict):
        raise ValueError("Training plan output must be a JSON object (dict).")

    _require(obj, "meta", dict)
    _require(obj, "snapshot", dict)
    _require(obj, "readiness", dict)
    _require(obj, "plan", dict)
    _require(obj, "recovery", dict)
    _require(obj, "risks", list)
    _require(obj, "data_notes", list)
    _require(obj, "citations", list)

    plan = obj["plan"]
    _require(plan, "weekly_totals", dict)
    days = _require(plan, "days", list)
    if not days:
        raise ValueError("plan.days must be a non-empty list.")

    for i, d in enumerate(days):
        if not isinstance(d, dict):
            raise ValueError(f"plan.days[{i}] must be an object.")
        for k in ("date", "session_type", "is_hard_day", "is_rest_day", "duration_minutes", "signal_ids"):
            if k not in d:
                raise ValueError(f"plan.days[{i}] missing required key: {k}")

    return obj