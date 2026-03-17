from __future__ import annotations

import datetime as dt
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- training-plan artifact ----------


class SnapshotStats(StrictModel):
    """Snapshot statistics for a time window.

    Numeric-looking fields are kept as ``str`` because the LLM may emit
    empty strings for unavailable values.  Validators coerce obvious
    numerics so downstream code can parse them, but empty string
    (``""``) is preserved as-is to mean "not available".
    """

    distance_km: str
    moving_time_hours: str
    elevation_m: str
    activity_count: str
    sleep_hours_mean: str
    hrv_mean: str
    rhr_mean: str

    @field_validator(
        "distance_km",
        "moving_time_hours",
        "elevation_m",
        "activity_count",
        "sleep_hours_mean",
        "hrv_mean",
        "rhr_mean",
        mode="before",
    )
    @classmethod
    def _coerce_to_str(cls, v: Any) -> str:
        """Accept numeric inputs and coerce to string representation."""
        if v is None:
            return ""
        if isinstance(v, (int, float)):
            return str(v)
        return str(v)


class TrainingMeta(StrictModel):
    today: dt.date
    plan_start: dt.date
    plan_days: int = Field(ge=1, le=28)
    style: str
    primary_goal: str = "to become a faster endurance athlete"

    @field_validator("today", "plan_start", mode="before")
    @classmethod
    def _parse_date(cls, v: Any) -> dt.date:
        if isinstance(v, dt.date):
            return v
        if isinstance(v, str):
            return dt.date.fromisoformat(v[:10])
        raise ValueError(f"Cannot parse date from {v!r}")


class Readiness(StrictModel):
    status: Literal["primed", "steady", "fatigued"]
    rationale: str
    signal_ids: list[str] = Field(default_factory=list)


class WeeklyTotals(StrictModel):
    planned_distance_km: float = Field(ge=0)
    planned_moving_time_hours: float = Field(ge=0)
    planned_elevation_m: float = Field(ge=0)


class PlanDay(StrictModel):
    date: dt.date
    title: str
    session_type: Literal[
        "rest",
        "easy",
        "aerobic",
        "long",
        "tempo",
        "intervals",
        "hills",
        "strength",
        "cross",
    ]
    is_rest_day: bool
    is_hard_day: bool
    duration_minutes: int = Field(ge=0, le=420)
    target_intensity: str
    terrain: str
    workout: str
    purpose: str
    signal_ids: list[str] = Field(default_factory=list)

    @field_validator("date", mode="before")
    @classmethod
    def _parse_date(cls, v: Any) -> dt.date:
        if isinstance(v, dt.date):
            return v
        if isinstance(v, str):
            return dt.date.fromisoformat(v[:10])
        raise ValueError(f"Cannot parse date from {v!r}")


class Plan(StrictModel):
    weekly_totals: WeeklyTotals
    days: list[PlanDay] = Field(min_length=1, max_length=28)


class Recovery(StrictModel):
    actions: list[str] = Field(default_factory=list)
    signal_ids: list[str] = Field(default_factory=list)


class RiskItem(StrictModel):
    severity: Literal["low", "medium", "high"]
    message: str
    signal_ids: list[str] = Field(default_factory=list)


class Citation(StrictModel):
    signal_id: str
    source: str
    date_range: str
    value: str


class Snapshot(StrictModel):
    last7: SnapshotStats
    baseline28: SnapshotStats
    notes: str


class TrainingPlanArtifact(StrictModel):
    meta: TrainingMeta
    snapshot: Snapshot
    readiness: Readiness
    plan: Plan
    recovery: Recovery
    risks: list[RiskItem] = Field(default_factory=list)
    data_notes: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


# ---------- eval-coach artifact ----------


class Violation(StrictModel):
    code: str
    severity: Literal["low", "medium", "high"]
    category: str
    penalty: int
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RubricScoreArtifact(StrictModel):
    score: Union[int, float]
    reasoning: str


class MarkerAssessmentArtifact(StrictModel):
    rubric: str
    marker_id: str
    marker: str
    verdict: Literal["pass", "partial", "fail"]
    score: Union[int, float]
    evidence: str
    improvement_hint: str


class SoftAssessmentArtifact(StrictModel):
    model: str
    style: Optional[str] = None
    primary_goal: str
    summary: str
    overall_score: Union[int, float]
    grade: str
    confidence: Literal["low", "medium", "high"]
    rubric_scores: dict[str, RubricScoreArtifact] = Field(default_factory=dict)
    marker_results: list[MarkerAssessmentArtifact] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    suggested_improvements: list[str] = Field(default_factory=list)
    repaired: bool = False
    derived_fields: list[str] = Field(default_factory=list)


class EvaluationReportArtifact(StrictModel):
    score: Union[int, float]
    grade: str
    subscores: dict[str, Union[int, float]] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    violations: list[Violation] = Field(default_factory=list)
    soft_assessment: Optional[SoftAssessmentArtifact] = None


# ---------- forecast artifact ----------


class ForecastInputs(StrictModel):
    as_of_date: dt.date
    rhr_7d_mean_bpm: Optional[float] = None
    rhr_28d_mean_bpm: Optional[float] = None
    rhr_28d_std_bpm: Optional[float] = None
    rhr_delta_bpm: Optional[float] = None
    rhr_z: Optional[float] = None
    training_load_7d_hours: Optional[float] = None
    training_load_rolling7_mean_hours: Optional[float] = None
    training_load_rolling7_std_hours: Optional[float] = None
    training_load_delta_hours: Optional[float] = None
    training_load_z: Optional[float] = None

    sleep_7d_mean_hours: Optional[float] = None
    hrv_7d_mean_ms: Optional[float] = None

    recovery_capability_key: str = "load_only"
    recovery_capability_label: str = "I only have training data"

    sleep_days_7d: int = 0
    resting_hr_days_7d: int = 0
    hrv_days_7d: int = 0

    notes: list[str] = Field(default_factory=list)

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _parse_date(cls, v: Any) -> dt.date:
        if isinstance(v, dt.date):
            return v
        if isinstance(v, str):
            return dt.date.fromisoformat(v[:10])
        raise ValueError(f"Cannot parse date from {v!r}")


class ForecastDrivers(StrictModel):
    readiness: list[str] = Field(default_factory=list)
    overreach_risk: list[str] = Field(default_factory=list)


class ForecastReadiness(StrictModel):
    score: float
    status: Literal["primed", "steady", "fatigued"]


class ForecastRisk(StrictModel):
    score: float
    level: Literal["low", "moderate", "high"]


class ForecastResultArtifact(StrictModel):
    date: dt.date
    readiness: ForecastReadiness
    overreach_risk: ForecastRisk
    inputs: ForecastInputs
    drivers: ForecastDrivers

    @field_validator("date", mode="before")
    @classmethod
    def _parse_date(cls, v: Any) -> dt.date:
        if isinstance(v, dt.date):
            return v
        if isinstance(v, str):
            return dt.date.fromisoformat(v[:10])
        raise ValueError(f"Cannot parse date from {v!r}")


class ForecastArtifact(StrictModel):
    generated_at: str
    result: ForecastResultArtifact
