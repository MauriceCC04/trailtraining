from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from trailtraining.llm.windowing import extract_last7_hours, normalize_plan_days, rolling_windows
from trailtraining.util.dates import _as_date


@dataclass(frozen=True)
class ConstraintConfig:
    max_ramp_pct: float = 10.0
    max_consecutive_hard: int = 2

    max_hard_per_7d: int = 3
    min_rest_per_7d: int = 1
    min_signal_ids_per_day: int = 1

    weekly_time_tolerance_pct: float = 30.0

    rest_day_max_minutes: int = 30
    require_rest_session_type: bool = True

    fatigued_max_hard_per_7d: int = 1
    high_risk_max_hard_per_7d: int = 1
    high_risk_max_consecutive_hard: int = 1
    high_risk_max_ramp_pct: float = 0.0

    sparse_data_max_hard_per_7d: int = 2
    sparse_data_max_ramp_pct: float = 5.0


@dataclass(frozen=True)
class EffectiveConstraintContext:
    allowed_week1_hours: Optional[float]
    effective_max_ramp_pct: float
    effective_max_hard_per_7d: int
    effective_max_consecutive_hard: int
    min_rest_per_7d: int
    readiness_status: Optional[str]
    overreach_risk_level: Optional[str]
    recovery_capability_key: Optional[str]
    lifestyle_notes: str
    reasons: list[str]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def constraint_config_from_env(
    *,
    max_ramp_pct: float | None = None,
    max_consecutive_hard: int | None = None,
) -> ConstraintConfig:
    base = ConstraintConfig()
    return ConstraintConfig(
        max_ramp_pct=(
            float(max_ramp_pct)
            if max_ramp_pct is not None
            else _env_float("TRAILTRAINING_MAX_RAMP_PCT", base.max_ramp_pct)
        ),
        max_consecutive_hard=(
            int(max_consecutive_hard)
            if max_consecutive_hard is not None
            else _env_int("TRAILTRAINING_MAX_CONSEC_HARD", base.max_consecutive_hard)
        ),
        max_hard_per_7d=_env_int("TRAILTRAINING_MAX_HARD_PER_7D", base.max_hard_per_7d),
        min_rest_per_7d=_env_int("TRAILTRAINING_MIN_REST_PER_7D", base.min_rest_per_7d),
        min_signal_ids_per_day=_env_int(
            "TRAILTRAINING_MIN_SIGNAL_IDS_PER_DAY", base.min_signal_ids_per_day
        ),
        weekly_time_tolerance_pct=_env_float(
            "TRAILTRAINING_WEEKLY_TIME_TOLERANCE_PCT", base.weekly_time_tolerance_pct
        ),
        rest_day_max_minutes=_env_int(
            "TRAILTRAINING_REST_DAY_MAX_MINUTES", base.rest_day_max_minutes
        ),
        require_rest_session_type=_env_bool(
            "TRAILTRAINING_REQUIRE_REST_SESSION_TYPE", base.require_rest_session_type
        ),
        fatigued_max_hard_per_7d=_env_int(
            "TRAILTRAINING_FATIGUED_MAX_HARD_PER_7D", base.fatigued_max_hard_per_7d
        ),
        high_risk_max_hard_per_7d=_env_int(
            "TRAILTRAINING_HIGH_RISK_MAX_HARD_PER_7D", base.high_risk_max_hard_per_7d
        ),
        high_risk_max_consecutive_hard=_env_int(
            "TRAILTRAINING_HIGH_RISK_MAX_CONSEC_HARD", base.high_risk_max_consecutive_hard
        ),
        high_risk_max_ramp_pct=_env_float(
            "TRAILTRAINING_HIGH_RISK_MAX_RAMP_PCT", base.high_risk_max_ramp_pct
        ),
        sparse_data_max_hard_per_7d=_env_int(
            "TRAILTRAINING_SPARSE_DATA_MAX_HARD_PER_7D", base.sparse_data_max_hard_per_7d
        ),
        sparse_data_max_ramp_pct=_env_float(
            "TRAILTRAINING_SPARSE_DATA_MAX_RAMP_PCT", base.sparse_data_max_ramp_pct
        ),
    )


def _pct_increase(new: float, old: Optional[float]) -> Optional[float]:
    if old is None or old <= 0:
        return None
    return (new - old) / old * 100.0


def _default_penalty(severity: str) -> int:
    return {"low": 3, "medium": 10, "high": 30}.get(severity, 10)


def _v(
    code: str,
    severity: str,
    category: str,
    message: str,
    *,
    penalty: Optional[int] = None,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "category": category,
        "penalty": int(_default_penalty(severity) if penalty is None else penalty),
        "message": message,
        "details": details or {},
    }


def _as_clean_str(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    return s or None


def _citation_value(plan_obj: dict[str, Any], signal_id: str) -> Any:
    cits = plan_obj.get("citations")
    if not isinstance(cits, list):
        return None
    for c in cits:
        if not isinstance(c, dict):
            continue
        if c.get("signal_id") != signal_id:
            continue
        if "value" in c:
            return c.get("value")
        if isinstance(c.get("quote"), str):
            return c.get("quote")
        if isinstance(c.get("text"), str):
            return c.get("text")
    return None


def _extract_forecast_context_from_citations(plan_obj: dict[str, Any]) -> dict[str, Optional[str]]:
    readiness = _as_clean_str(_citation_value(plan_obj, "forecast.readiness.status"))
    overreach = _as_clean_str(_citation_value(plan_obj, "forecast.overreach_risk.level"))
    capability_key = _as_clean_str(_citation_value(plan_obj, "forecast.recovery_capability.key"))
    capability_label = _as_clean_str(
        _citation_value(plan_obj, "forecast.recovery_capability.label")
    )

    return {
        "readiness_status": readiness.lower() if readiness else None,
        "overreach_risk_level": overreach.lower() if overreach else None,
        "recovery_capability_key": capability_key.lower() if capability_key else None,
        "recovery_capability_label": capability_label,
    }


def _extract_effective_constraints(
    plan_obj: dict[str, Any],
) -> Optional[EffectiveConstraintContext]:
    raw = plan_obj.get("effective_constraints")
    if not isinstance(raw, dict):
        return None

    def _float(name: str, default: float) -> float:
        value = raw.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _int(name: str, default: int) -> int:
        value = raw.get(name, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    allowed_raw = raw.get("allowed_week1_hours")
    try:
        allowed = float(allowed_raw) if isinstance(allowed_raw, (int, float)) else None
    except (TypeError, ValueError):
        allowed = None

    return EffectiveConstraintContext(
        allowed_week1_hours=allowed,
        effective_max_ramp_pct=_float("effective_max_ramp_pct", ConstraintConfig().max_ramp_pct),
        effective_max_hard_per_7d=_int(
            "effective_max_hard_per_7d", ConstraintConfig().max_hard_per_7d
        ),
        effective_max_consecutive_hard=_int(
            "effective_max_consecutive_hard", ConstraintConfig().max_consecutive_hard
        ),
        min_rest_per_7d=_int("min_rest_per_7d", ConstraintConfig().min_rest_per_7d),
        readiness_status=_as_clean_str(raw.get("readiness_status")),
        overreach_risk_level=_as_clean_str(raw.get("overreach_risk_level")),
        recovery_capability_key=_as_clean_str(raw.get("recovery_capability_key")),
        lifestyle_notes=str(raw.get("lifestyle_notes", "") or "").strip(),
        reasons=[
            str(item).strip()
            for item in raw.get("reasons", [])
            if isinstance(item, str) and str(item).strip()
        ],
    )


def _extract_forecast_context_from_artifact(plan_obj: dict[str, Any]) -> dict[str, Optional[str]]:
    effective = _extract_effective_constraints(plan_obj)
    if effective is None:
        return _extract_forecast_context_from_citations(plan_obj)

    return {
        "readiness_status": effective.readiness_status.lower()
        if effective.readiness_status
        else None,
        "overreach_risk_level": effective.overreach_risk_level.lower()
        if effective.overreach_risk_level
        else None,
        "recovery_capability_key": effective.recovery_capability_key.lower()
        if effective.recovery_capability_key
        else None,
        "recovery_capability_label": None,
    }


def _extract_forecast_context(plan_obj: dict[str, Any]) -> dict[str, Optional[str]]:
    effective = _extract_effective_constraints(plan_obj)
    if effective is not None:
        return _extract_forecast_context_from_artifact(plan_obj)
    return _extract_forecast_context_from_citations(plan_obj)


def _is_sparse_capability(ctx: dict[str, Optional[str]]) -> bool:
    key = ctx.get("recovery_capability_key")
    label = (ctx.get("recovery_capability_label") or "").lower()

    if key in {"load_only", "load_sleep", "load_resting_hr", "load_hrv"}:
        return True

    if "only have training data" in label:
        return True

    return label.startswith("i have load + ") and " only" in label


def _forecast_reason_text(ctx: dict[str, Optional[str]]) -> str:
    reasons: list[str] = []

    if ctx.get("readiness_status") == "fatigued":
        reasons.append("readiness is fatigued")
    if ctx.get("overreach_risk_level") == "high":
        reasons.append("overreach risk is high")
    if _is_sparse_capability(ctx):
        reasons.append("recovery telemetry is sparse")

    if not reasons:
        return "current forecast context"

    return "; ".join(reasons)


def derive_effective_constraints(
    *,
    det_forecast: Optional[dict[str, Any]],
    rollups: Optional[dict[str, Any]],
    cfg: ConstraintConfig,
    lifestyle_notes: str = "",
) -> EffectiveConstraintContext:
    readiness_status: Optional[str] = None
    overreach_risk_level: Optional[str] = None
    recovery_capability_key: Optional[str] = None
    recovery_capability_label: Optional[str] = None

    if isinstance(det_forecast, dict):
        result = det_forecast.get("result") or {}
        if isinstance(result, dict):
            readiness = result.get("readiness") or {}
            risk = result.get("overreach_risk") or {}
            inputs = result.get("inputs") or {}
            if isinstance(readiness, dict):
                readiness_status = _as_clean_str(readiness.get("status"))
                readiness_status = readiness_status.lower() if readiness_status else None
            if isinstance(risk, dict):
                overreach_risk_level = _as_clean_str(risk.get("level"))
                overreach_risk_level = (
                    overreach_risk_level.lower() if overreach_risk_level else None
                )
            if isinstance(inputs, dict):
                recovery_capability_key = _as_clean_str(inputs.get("recovery_capability_key"))
                recovery_capability_key = (
                    recovery_capability_key.lower() if recovery_capability_key else None
                )
                recovery_capability_label = _as_clean_str(inputs.get("recovery_capability_label"))

    effective_max_hard_per_7d = cfg.max_hard_per_7d
    effective_max_consecutive_hard = cfg.max_consecutive_hard
    effective_max_ramp_pct = cfg.max_ramp_pct
    reasons: list[str] = []

    forecast_ctx = {
        "readiness_status": readiness_status,
        "overreach_risk_level": overreach_risk_level,
        "recovery_capability_key": recovery_capability_key,
        "recovery_capability_label": recovery_capability_label,
    }

    if readiness_status == "fatigued":
        effective_max_hard_per_7d = min(effective_max_hard_per_7d, cfg.fatigued_max_hard_per_7d)
        reasons.append("readiness is fatigued")

    if overreach_risk_level == "high":
        effective_max_hard_per_7d = min(effective_max_hard_per_7d, cfg.high_risk_max_hard_per_7d)
        effective_max_consecutive_hard = min(
            effective_max_consecutive_hard, cfg.high_risk_max_consecutive_hard
        )
        effective_max_ramp_pct = min(effective_max_ramp_pct, cfg.high_risk_max_ramp_pct)
        reasons.append("overreach risk is high")

    if _is_sparse_capability(forecast_ctx):
        effective_max_hard_per_7d = min(effective_max_hard_per_7d, cfg.sparse_data_max_hard_per_7d)
        effective_max_ramp_pct = min(effective_max_ramp_pct, cfg.sparse_data_max_ramp_pct)
        reasons.append("recovery telemetry is sparse")

    if lifestyle_notes.strip():
        reasons.append("lifestyle constraints apply")

    last7 = extract_last7_hours(rollups)
    allowed_week1_hours = (
        float(last7) * (1.0 + effective_max_ramp_pct / 100.0)
        if isinstance(last7, (int, float)) and float(last7) > 0
        else None
    )

    return EffectiveConstraintContext(
        allowed_week1_hours=allowed_week1_hours,
        effective_max_ramp_pct=effective_max_ramp_pct,
        effective_max_hard_per_7d=effective_max_hard_per_7d,
        effective_max_consecutive_hard=effective_max_consecutive_hard,
        min_rest_per_7d=cfg.min_rest_per_7d,
        readiness_status=readiness_status,
        overreach_risk_level=overreach_risk_level,
        recovery_capability_key=recovery_capability_key,
        lifestyle_notes=lifestyle_notes.strip(),
        reasons=reasons,
    )


def _planned_week_hours(plan_obj: dict[str, Any]) -> Optional[float]:
    wt = (plan_obj.get("plan") or {}).get("weekly_totals") or {}
    v = wt.get("planned_moving_time_hours")
    return float(v) if isinstance(v, (int, float)) else None


def _sum_hours(days: list[dict[str, Any]]) -> float:
    total_min = 0.0
    for d in days:
        m = d.get("duration_minutes")
        if isinstance(m, (int, float)):
            total_min += float(m)
    return total_min / 60.0


def _pct_diff(a: float, b: float) -> Optional[float]:
    if b <= 0:
        return None
    return abs(a - b) / b * 100.0


def _citation_lookup(plan_obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cits = plan_obj.get("citations")
    if not isinstance(cits, list):
        return out
    for item in cits:
        if not isinstance(item, dict):
            continue
        signal_id = str(item.get("signal_id", "") or "").strip()
        if not signal_id:
            continue
        citation_id = str(
            item.get("citation_id", "") or f"cit_{signal_id.replace(' ', '_')}"
        ).strip()
        normalized = dict(item)
        normalized["citation_id"] = citation_id
        out[citation_id] = normalized
    return out


def _required_claim_field_paths(plan_obj: dict[str, Any]) -> list[str]:
    if "claim_attributions" not in plan_obj:
        return []

    required: list[str] = ["readiness.rationale"]

    recovery = plan_obj.get("recovery") or {}
    if isinstance(recovery, dict):
        for idx, action in enumerate(recovery.get("actions") or []):
            if isinstance(action, str) and action.strip():
                required.append(f"recovery.actions[{idx}]")

    for idx, risk in enumerate(plan_obj.get("risks") or []):
        if isinstance(risk, dict) and str(risk.get("message", "") or "").strip():
            required.append(f"risks[{idx}].message")

    days = normalize_plan_days(plan_obj)
    for idx, day in enumerate(days):
        if str(day.get("purpose", "") or "").strip():
            required.append(f"plan.days[{idx}].purpose")

    return required


def _preferred_signal_prefixes(field_path: str, claim_text: str) -> list[str]:
    low_text = claim_text.lower()

    if field_path == "readiness.rationale":
        if "missing" in low_text or "sparse" in low_text or "telemetry" in low_text:
            return ["forecast.recovery_capability."]
        if "overreach" in low_text or "risk" in low_text:
            return ["forecast.overreach_risk.", "load."]
        return ["forecast.readiness.", "recovery.", "load."]

    if field_path.startswith("recovery.actions["):
        return ["forecast.readiness.", "forecast.recovery_capability.", "recovery.", "load."]

    if field_path.startswith("risks["):
        return ["forecast.overreach_risk.", "forecast.readiness.", "load.", "recovery."]

    if field_path.startswith("plan.days[") and field_path.endswith("].purpose"):
        return ["forecast.", "load.", "recovery."]

    return ["forecast.", "load.", "recovery."]


def _has_preferred_signal(signal_ids: list[str], prefixes: list[str]) -> bool:
    return any(any(signal_id.startswith(prefix) for prefix in prefixes) for signal_id in signal_ids)


def validate_claim_support(plan_obj: dict[str, Any]) -> list[dict[str, Any]]:
    if "claim_attributions" not in plan_obj:
        return []

    violations: list[dict[str, Any]] = []
    claim_items = plan_obj.get("claim_attributions")
    if not isinstance(claim_items, list):
        return [
            _v(
                "CLAIM_ATTRIBUTIONS_BAD_TYPE",
                "medium",
                "justification",
                "claim_attributions must be a list when present.",
            )
        ]

    by_field: dict[str, list[dict[str, Any]]] = {}
    citation_lookup = _citation_lookup(plan_obj)

    for item in claim_items:
        if not isinstance(item, dict):
            violations.append(
                _v(
                    "CLAIM_ATTRIBUTION_BAD_ITEM",
                    "low",
                    "justification",
                    "claim_attributions contains a non-object item.",
                )
            )
            continue

        field_path = str(item.get("field_path", "") or "").strip()
        claim_text = str(item.get("claim_text", "") or "").strip()
        signal_ids = [
            str(s).strip()
            for s in item.get("signal_ids", [])
            if isinstance(s, str) and str(s).strip()
        ]
        citation_ids = [
            str(c).strip()
            for c in item.get("citation_ids", [])
            if isinstance(c, str) and str(c).strip()
        ]
        support_level = str(item.get("support_level", "") or "").strip().lower()

        if field_path:
            by_field.setdefault(field_path, []).append(item)

        if not field_path:
            violations.append(
                _v(
                    "CLAIM_MISSING_FIELD_PATH",
                    "medium",
                    "justification",
                    "Claim attribution is missing field_path.",
                )
            )

        if not claim_text:
            violations.append(
                _v(
                    "CLAIM_MISSING_TEXT",
                    "low",
                    "justification",
                    f"Claim attribution for {field_path or '(unknown field)'} is missing claim_text.",
                )
            )

        if not signal_ids:
            violations.append(
                _v(
                    "CLAIM_MISSING_SIGNAL_IDS",
                    "medium",
                    "justification",
                    f"Claim attribution for {field_path or '(unknown field)'} has no signal_ids.",
                )
            )

        if not citation_ids:
            violations.append(
                _v(
                    "CLAIM_MISSING_CITATIONS",
                    "medium",
                    "justification",
                    f"Claim attribution for {field_path or '(unknown field)'} has no citation_ids.",
                )
            )

        for citation_id in citation_ids:
            cited = citation_lookup.get(citation_id)
            if not cited:
                violations.append(
                    _v(
                        "CLAIM_UNKNOWN_CITATION",
                        "high",
                        "justification",
                        f"Claim attribution references unknown citation_id '{citation_id}'.",
                        details={"citation_id": citation_id, "field_path": field_path},
                    )
                )
                continue

            cited_signal = str(cited.get("signal_id", "") or "").strip()
            if cited_signal and cited_signal not in signal_ids:
                violations.append(
                    _v(
                        "CLAIM_SOURCE_MISMATCH",
                        "medium",
                        "justification",
                        f"Claim attribution cites {citation_id} ({cited_signal}) but does not list that signal_id.",
                        details={
                            "citation_id": citation_id,
                            "signal_id": cited_signal,
                            "field_path": field_path,
                        },
                    )
                )

        prefixes = _preferred_signal_prefixes(field_path, claim_text)
        if signal_ids and not _has_preferred_signal(signal_ids, prefixes):
            violations.append(
                _v(
                    "WEAK_CLAIM_SUPPORT",
                    "low",
                    "justification",
                    f"Claim attribution for {field_path or '(unknown field)'} does not use the most relevant evidence family.",
                    details={"field_path": field_path, "signal_ids": signal_ids},
                )
            )

        if support_level == "unsupported":
            violations.append(
                _v(
                    "UNSUPPORTED_RATIONALE",
                    "medium",
                    "justification",
                    f"Claim attribution for {field_path or '(unknown field)'} is marked unsupported.",
                    details={"field_path": field_path, "claim_text": claim_text},
                )
            )

    for required_path in _required_claim_field_paths(plan_obj):
        if required_path not in by_field:
            violations.append(
                _v(
                    "CLAIM_MISSING_ATTRIBUTION",
                    "medium",
                    "justification",
                    f"Missing claim attribution for {required_path}.",
                    details={"field_path": required_path},
                )
            )

    return violations


def validate_training_plan(
    plan_obj: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    cfg: ConstraintConfig,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    days = normalize_plan_days(plan_obj)

    declared_planned_hours = _planned_week_hours(plan_obj)
    actual_first7_hours = _sum_hours(days[: min(7, len(days))]) if days else None

    last7_hours = extract_last7_hours(rollups)

    ramp_basis_hours = (
        actual_first7_hours if actual_first7_hours is not None else declared_planned_hours
    )

    if isinstance(ramp_basis_hours, (int, float)) and isinstance(last7_hours, (int, float)):
        inc = _pct_increase(float(ramp_basis_hours), float(last7_hours))
        if inc is not None and inc > cfg.max_ramp_pct:
            violations.append(
                _v(
                    "MAX_RAMP_PCT",
                    "high",
                    "safety",
                    f"Planned first-7-day moving time ramps {inc:.1f}% vs last 7 days "
                    f"(max {cfg.max_ramp_pct:.1f}%).",
                    details={
                        "ramp_basis_hours": ramp_basis_hours,
                        "declared_planned_hours": declared_planned_hours,
                        "last7_hours": last7_hours,
                        "ramp_pct": inc,
                    },
                )
            )

    consec = 0
    for d in days:
        hard = bool(d.get("is_hard_day"))
        if hard:
            consec += 1
            if consec > cfg.max_consecutive_hard:
                violations.append(
                    _v(
                        "TOO_MANY_CONSEC_HARD",
                        "high",
                        "safety",
                        f"More than {cfg.max_consecutive_hard} hard days in a row (hit {consec}).",
                        details={"date": d.get("date")},
                    )
                )
        else:
            consec = 0

    return violations


def evaluate_training_plan_quality(
    plan_obj: dict[str, Any],
    rollups: Optional[dict[str, Any]],
    cfg: ConstraintConfig,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for v0 in validate_training_plan(plan_obj, rollups, cfg):
        if isinstance(v0, dict):
            v0.setdefault("category", "safety")
            v0.setdefault("penalty", _default_penalty(str(v0.get("severity", "medium"))))
            violations.append(v0)

    days = normalize_plan_days(plan_obj)

    hard_days = sum(1 for d in days if bool(d.get("is_hard_day")))
    rest_days = sum(1 for d in days if bool(d.get("is_rest_day")))
    stats: dict[str, Any] = {"days": len(days), "hard_days": hard_days, "rest_days": rest_days}

    fx = _extract_forecast_context(plan_obj)
    for k, v in fx.items():
        if v is not None:
            stats[k] = v

    effective = _extract_effective_constraints(plan_obj)
    effective_max_hard_per_7d = cfg.max_hard_per_7d
    effective_max_consecutive_hard = cfg.max_consecutive_hard
    effective_max_ramp_pct = cfg.max_ramp_pct

    if effective is not None:
        effective_max_hard_per_7d = effective.effective_max_hard_per_7d
        effective_max_consecutive_hard = effective.effective_max_consecutive_hard
        effective_max_ramp_pct = effective.effective_max_ramp_pct
    else:
        if fx.get("readiness_status") == "fatigued":
            effective_max_hard_per_7d = min(effective_max_hard_per_7d, cfg.fatigued_max_hard_per_7d)

        if fx.get("overreach_risk_level") == "high":
            effective_max_hard_per_7d = min(
                effective_max_hard_per_7d, cfg.high_risk_max_hard_per_7d
            )
            effective_max_consecutive_hard = min(
                effective_max_consecutive_hard, cfg.high_risk_max_consecutive_hard
            )
            effective_max_ramp_pct = min(effective_max_ramp_pct, cfg.high_risk_max_ramp_pct)

        if _is_sparse_capability(fx):
            effective_max_hard_per_7d = min(
                effective_max_hard_per_7d, cfg.sparse_data_max_hard_per_7d
            )
            effective_max_ramp_pct = min(effective_max_ramp_pct, cfg.sparse_data_max_ramp_pct)

    stats["effective_max_hard_per_7d"] = effective_max_hard_per_7d
    stats["effective_max_consecutive_hard"] = effective_max_consecutive_hard
    stats["effective_max_ramp_pct"] = effective_max_ramp_pct

    seen = set()
    prev: Optional[date] = None
    for d in days:
        ds = d.get("date")
        dd = _as_date(ds)
        if not dd:
            violations.append(
                _v(
                    "BAD_DATE",
                    "low",
                    "structure",
                    "Day has invalid/missing date.",
                    details={"date": ds},
                )
            )
            continue

        if dd in seen:
            violations.append(
                _v(
                    "DUPLICATE_DATE",
                    "high",
                    "structure",
                    "Duplicate date in plan.days.",
                    details={"date": ds},
                )
            )
        seen.add(dd)

        if prev and (dd - prev).days != 1:
            violations.append(
                _v(
                    "NON_CONSECUTIVE_DATES",
                    "medium",
                    "structure",
                    "Plan dates are not consecutive (gap or reorder).",
                    details={"prev": prev.isoformat(), "curr": dd.isoformat()},
                )
            )
        prev = dd

    planned_hours = _planned_week_hours(plan_obj)
    if planned_hours is not None and days:
        first7 = days[: min(7, len(days))]
        sum_hours = _sum_hours(first7)
        diff = _pct_diff(planned_hours, sum_hours)
        if diff is not None and diff > cfg.weekly_time_tolerance_pct:
            violations.append(
                _v(
                    "WEEKLY_TOTALS_MISMATCH",
                    "low",
                    "structure",
                    f"weekly_totals.planned_moving_time_hours ({planned_hours:.1f}h) doesn't match "
                    f"sum(duration) ({sum_hours:.1f}h) within {cfg.weekly_time_tolerance_pct:.0f}%.",
                    details={
                        "planned_hours": planned_hours,
                        "sum_hours_first7": sum_hours,
                        "pct_diff": diff,
                    },
                )
            )

    last7_hours = extract_last7_hours(rollups)

    actual_first7_hours = _sum_hours(days[: min(7, len(days))]) if days else None
    ramp_basis_hours = actual_first7_hours if actual_first7_hours is not None else planned_hours

    if isinstance(ramp_basis_hours, (int, float)) and isinstance(last7_hours, (int, float)):
        inc = _pct_increase(float(ramp_basis_hours), float(last7_hours))
        if (
            inc is not None
            and inc > effective_max_ramp_pct
            and effective_max_ramp_pct < cfg.max_ramp_pct
        ):
            reason_text = _forecast_reason_text(fx)
            sev = "high" if fx.get("overreach_risk_level") == "high" else "medium"
            violations.append(
                _v(
                    "FORECAST_RAMP_TOO_AGGRESSIVE",
                    sev,
                    "safety",
                    f"Planned first-7-day moving time ramps {inc:.1f}% vs last 7 days, "
                    f"but limit is {effective_max_ramp_pct:.1f}% because {reason_text}.",
                    details={
                        "ramp_basis_hours": ramp_basis_hours,
                        "last7_hours": last7_hours,
                        "ramp_pct": inc,
                        "effective_max_ramp_pct": effective_max_ramp_pct,
                        "forecast_context": fx,
                    },
                    penalty=35 if sev == "high" else 15,
                )
            )

    for i, wk in enumerate(rolling_windows(days, size=7)):
        h = sum(1 for d in wk if bool(d.get("is_hard_day")))
        if h > cfg.max_hard_per_7d:
            violations.append(
                _v(
                    "TOO_MANY_HARD_PER_7D",
                    "high",
                    "safety",
                    f"Rolling 7-day window {i} has {h} hard days (max {cfg.max_hard_per_7d}).",
                    details={
                        "window_index": i,
                        "window_start": wk[0].get("date"),
                        "window_end": wk[-1].get("date"),
                        "hard_days": h,
                    },
                )
            )

    for i, wk in enumerate(rolling_windows(days, size=7)):
        r = sum(1 for d in wk if bool(d.get("is_rest_day")))
        if r < cfg.min_rest_per_7d:
            sev = "high" if r == 0 else "medium"
            violations.append(
                _v(
                    "NOT_ENOUGH_REST",
                    sev,
                    "safety",
                    f"Rolling 7-day window {i} has {r} rest days (min {cfg.min_rest_per_7d}).",
                    details={
                        "window_index": i,
                        "window_start": wk[0].get("date"),
                        "window_end": wk[-1].get("date"),
                        "rest_days": r,
                    },
                    penalty=35 if sev == "high" else 15,
                )
            )

    if effective_max_hard_per_7d < cfg.max_hard_per_7d:
        reason_text = _forecast_reason_text(fx)
        sev = "high" if fx.get("overreach_risk_level") == "high" else "medium"

        for i, wk in enumerate(rolling_windows(days, size=7)):
            h = sum(1 for d in wk if bool(d.get("is_hard_day")))
            if h > effective_max_hard_per_7d:
                violations.append(
                    _v(
                        "FORECAST_HARD_DAY_LIMIT",
                        sev,
                        "safety",
                        f"Rolling 7-day window {i} has {h} hard days, but limit is "
                        f"{effective_max_hard_per_7d} because {reason_text}.",
                        details={
                            "window_index": i,
                            "window_start": wk[0].get("date"),
                            "window_end": wk[-1].get("date"),
                            "hard_days": h,
                            "effective_max_hard_per_7d": effective_max_hard_per_7d,
                            "forecast_context": fx,
                        },
                        penalty=35 if sev == "high" else 15,
                    )
                )

    if effective_max_consecutive_hard < cfg.max_consecutive_hard:
        reason_text = _forecast_reason_text(fx)
        consec = 0
        streak_start: Optional[str] = None

        for d in days:
            if bool(d.get("is_hard_day")):
                consec += 1
                if consec == 1:
                    streak_start = d.get("date")
                if consec > effective_max_consecutive_hard:
                    violations.append(
                        _v(
                            "FORECAST_CONSEC_HARD_LIMIT",
                            "high",
                            "safety",
                            f"More than {effective_max_consecutive_hard} hard days in a row "
                            f"because {reason_text}.",
                            details={
                                "streak_start": streak_start,
                                "date": d.get("date"),
                                "consecutive_hard_days": consec,
                                "effective_max_consecutive_hard": effective_max_consecutive_hard,
                                "forecast_context": fx,
                            },
                            penalty=35,
                        )
                    )
            else:
                consec = 0
                streak_start = None

    for d in days:
        if not bool(d.get("is_rest_day")):
            continue
        mins = d.get("duration_minutes")
        if isinstance(mins, (int, float)) and float(mins) > float(cfg.rest_day_max_minutes):
            violations.append(
                _v(
                    "REST_DAY_TOO_LONG",
                    "medium",
                    "structure",
                    f"Rest day exceeds {cfg.rest_day_max_minutes} minutes.",
                    details={"date": d.get("date"), "duration_minutes": mins},
                )
            )
        if cfg.require_rest_session_type:
            st = d.get("session_type")
            if isinstance(st, str) and st != "rest":
                violations.append(
                    _v(
                        "REST_DAY_BAD_SESSION_TYPE",
                        "low",
                        "structure",
                        "Rest day should have session_type == 'rest'.",
                        details={"date": d.get("date"), "session_type": st},
                    )
                )

    for idx, d in enumerate(days):
        sig = d.get("signal_ids")
        n = len(sig) if isinstance(sig, list) else 0
        if n < cfg.min_signal_ids_per_day:
            violations.append(
                _v(
                    "MISSING_SIGNAL_IDS",
                    "medium",
                    "justification",
                    f"plan.days[{idx}] has empty/insufficient signal_ids (min {cfg.min_signal_ids_per_day}).",
                    details={"date": d.get("date")},
                )
            )

    cited = set()
    cits = plan_obj.get("citations")
    if isinstance(cits, list):
        for c in cits:
            if isinstance(c, dict) and isinstance(c.get("signal_id"), str):
                cited.add(c["signal_id"])

    used = set()
    for d in days:
        sig = d.get("signal_ids")
        if isinstance(sig, list):
            for s in sig:
                if isinstance(s, str):
                    used.add(s)

    readiness = plan_obj.get("readiness") or {}
    if isinstance(readiness, dict):
        for s in readiness.get("signal_ids", []):
            if isinstance(s, str):
                used.add(s)

    recovery = plan_obj.get("recovery") or {}
    if isinstance(recovery, dict):
        for s in recovery.get("signal_ids", []):
            if isinstance(s, str):
                used.add(s)

    for risk in plan_obj.get("risks", []) or []:
        if isinstance(risk, dict):
            for s in risk.get("signal_ids", []):
                if isinstance(s, str):
                    used.add(s)

    if used and not cited:
        violations.append(
            _v(
                "MISSING_CITATIONS",
                "medium",
                "justification",
                "Plan uses signal_ids but citations[] is empty/missing.",
                details={"used_signal_ids_count": len(used)},
            )
        )
    elif used and cited:
        missing = sorted(list(used - cited))
        if missing:
            violations.append(
                _v(
                    "UNCITED_SIGNAL_IDS",
                    "medium",
                    "justification",
                    "Some signal_ids used in plan.days/readiness/recovery/risks are not present in citations[].signal_id.",
                    details={"missing_signal_ids": missing[:50], "missing_count": len(missing)},
                )
            )

    violations.extend(validate_claim_support(plan_obj))

    return score_from_violations(violations, stats=stats)


def score_from_violations(
    violations: list[dict[str, Any]],
    *,
    stats: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    total_pen = 0
    by_cat: dict[str, int] = {}
    for v in violations:
        if not isinstance(v, dict):
            continue
        try:
            pen = int(v.get("penalty", _default_penalty(str(v.get("severity", "medium")))))
        except Exception:
            pen = 10
        total_pen += pen
        cat = str(v.get("category") or "other")
        by_cat[cat] = by_cat.get(cat, 0) + pen

    score = max(0, 100 - total_pen)
    subscores = {cat: max(0, 100 - pen) for cat, pen in by_cat.items()}

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": score,
        "grade": grade,
        "subscores": subscores,
        "stats": stats or {},
        "violations": violations,
    }
