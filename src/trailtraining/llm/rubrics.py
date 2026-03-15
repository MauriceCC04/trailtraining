from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_PRIMARY_GOAL_BY_STYLE = {
    "trailrunning": "to become a faster trail runner",
    "triathlon": "to become a faster triathlete",
}

DEFAULT_PRIMARY_GOAL = DEFAULT_PRIMARY_GOAL_BY_STYLE["trailrunning"]


def default_primary_goal_for_style(style: str | None) -> str:
    s = (style or "").strip().lower()
    return DEFAULT_PRIMARY_GOAL_BY_STYLE.get(s, "to become a faster endurance athlete")


@dataclass(frozen=True)
class MarkerDefinition:
    marker_id: str
    label: str
    description: str


@dataclass(frozen=True)
class RubricDefinition:
    rubric_id: str
    label: str
    weight: int
    description: str
    markers: tuple[MarkerDefinition, ...]


_BASE_RUBRIC_META: tuple[dict[str, Any], ...] = (
    {
        "rubric_id": "goal_alignment",
        "label": "Goal alignment",
        "weight": 30,
        "description": (
            "The plan should clearly support the athlete's stated goal rather than drifting into "
            "generic wellness advice or irrelevant training."
        ),
    },
    {
        "rubric_id": "plan_coherence",
        "label": "Plan coherence",
        "weight": 25,
        "description": (
            "The week should hang together logically: hard/easy spacing, sensible duration mix, "
            "and day-level purposes that match the claimed weekly totals."
        ),
    },
    {
        "rubric_id": "explanation_quality",
        "label": "Explanation quality",
        "weight": 20,
        "description": (
            "Reasoning and explanations should be useful, specific, and clearly tied to the plan "
            "and context, not vague template language."
        ),
    },
    {
        "rubric_id": "caution_proportionality",
        "label": "Caution proportionality",
        "weight": 15,
        "description": (
            "Cautions and reasoning should feel proportionate to readiness, overreach risk, and "
            "data quality. Neither reckless nor excessively defensive."
        ),
    },
    {
        "rubric_id": "actionability",
        "label": "Actionability",
        "weight": 10,
        "description": (
            "A real athlete should be able to follow the week without guessing what sessions mean "
            "or how to execute them."
        ),
    },
)

_STYLE_MARKERS: dict[str, dict[str, tuple[MarkerDefinition, ...]]] = {
    "trailrunning": {
        "goal_alignment": (
            MarkerDefinition(
                marker_id="goal_specificity",
                label="Goal specificity",
                description="The plan clearly targets becoming a faster trail runner.",
            ),
            MarkerDefinition(
                marker_id="trail_specificity",
                label="Trail specificity",
                description=(
                    "The sessions reflect trail-running demands such as aerobic durability, climbing, "
                    "terrain handling, or durability work when appropriate."
                ),
            ),
            MarkerDefinition(
                marker_id="progression_fit",
                label="Progression fit",
                description="The week moves the athlete forward without obvious mismatch to the goal.",
            ),
            MarkerDefinition(
                marker_id="non_competing_focus",
                label="Non-competing focus",
                description="Extra sessions support the main goal rather than distracting from it.",
            ),
        ),
        "plan_coherence": (
            MarkerDefinition(
                marker_id="hard_easy_spacing",
                label="Hard/easy spacing",
                description="Hard sessions are spaced sensibly with recovery around them.",
            ),
            MarkerDefinition(
                marker_id="weekly_internal_consistency",
                label="Internal consistency",
                description="Day details, session types, and weekly totals feel mutually consistent.",
            ),
            MarkerDefinition(
                marker_id="workout_purpose_fit",
                label="Workout-purpose fit",
                description="Each workout description matches the stated purpose.",
            ),
            MarkerDefinition(
                marker_id="load_progression_logic",
                label="Load progression logic",
                description="The load and emphasis across the week feel plausible and coherent.",
            ),
        ),
        "explanation_quality": (
            MarkerDefinition(
                marker_id="specificity",
                label="Specificity",
                description="The explanation uses specific plan details and context, not boilerplate.",
            ),
            MarkerDefinition(
                marker_id="actionable_reasoning",
                label="Actionable reasoning",
                description="The athlete can understand why the plan is structured this way.",
            ),
            MarkerDefinition(
                marker_id="non_generic_language",
                label="Non-generic language",
                description="The language avoids bland motivational filler and generic advice.",
            ),
            MarkerDefinition(
                marker_id="useful_day_purposes",
                label="Useful day purposes",
                description="Day-level purposes explain what each session is doing for the athlete.",
            ),
        ),
        "caution_proportionality": (
            MarkerDefinition(
                marker_id="risk_context_match",
                label="Risk-context match",
                description="Warnings match the actual context instead of overreacting or underreacting.",
            ),
            MarkerDefinition(
                marker_id="missing_data_handling",
                label="Missing-data handling",
                description="The plan reacts sensibly to sparse telemetry or uncertainty.",
            ),
            MarkerDefinition(
                marker_id="caution_tone",
                label="Caution tone",
                description="The tone is calm and proportionate rather than alarmist or dismissive.",
            ),
        ),
        "actionability": (
            MarkerDefinition(
                marker_id="session_clarity",
                label="Session clarity",
                description="Sessions are concrete enough to execute.",
            ),
            MarkerDefinition(
                marker_id="followability",
                label="Followability",
                description="The week is practical to follow from day to day.",
            ),
            MarkerDefinition(
                marker_id="recovery_integration",
                label="Recovery integration",
                description="Recovery actions are useful and connected to the week.",
            ),
        ),
    },
    "triathlon": {
        "goal_alignment": (
            MarkerDefinition(
                marker_id="goal_specificity",
                label="Goal specificity",
                description="The plan clearly targets becoming a faster triathlete.",
            ),
            MarkerDefinition(
                marker_id="discipline_balance",
                label="Discipline balance",
                description=(
                    "The plan balances swim, bike, and run appropriately for a broad triathlon goal "
                    "instead of collapsing into run-only fitness."
                ),
            ),
            MarkerDefinition(
                marker_id="triathlon_specificity",
                label="Triathlon specificity",
                description=(
                    "The sessions reflect triathlon demands such as cross-discipline endurance, "
                    "race-specific distribution, and sensible use of bricks when appropriate."
                ),
            ),
            MarkerDefinition(
                marker_id="non_competing_focus",
                label="Non-competing focus",
                description="Extra sessions support the main goal rather than distracting from it.",
            ),
        ),
        "plan_coherence": (
            MarkerDefinition(
                marker_id="hard_easy_spacing",
                label="Hard/easy spacing",
                description="Hard sessions are spaced sensibly with recovery around them across disciplines.",
            ),
            MarkerDefinition(
                marker_id="discipline_distribution",
                label="Discipline distribution",
                description="Swim, bike, and run stress are distributed in a coherent way across the week.",
            ),
            MarkerDefinition(
                marker_id="workout_purpose_fit",
                label="Workout-purpose fit",
                description="Each workout description matches the stated purpose.",
            ),
            MarkerDefinition(
                marker_id="load_progression_logic",
                label="Load progression logic",
                description="The load and emphasis across the week feel plausible and coherent.",
            ),
        ),
        "explanation_quality": (
            MarkerDefinition(
                marker_id="specificity",
                label="Specificity",
                description="The explanation uses specific plan details and context, not boilerplate.",
            ),
            MarkerDefinition(
                marker_id="discipline_reasoning",
                label="Discipline reasoning",
                description="The plan explains why swim, bike, and run work are included and how they fit together.",
            ),
            MarkerDefinition(
                marker_id="non_generic_language",
                label="Non-generic language",
                description="The language avoids bland motivational filler and generic advice.",
            ),
            MarkerDefinition(
                marker_id="useful_day_purposes",
                label="Useful day purposes",
                description="Day-level purposes explain what each session is doing for the athlete.",
            ),
        ),
        "caution_proportionality": (
            MarkerDefinition(
                marker_id="risk_context_match",
                label="Risk-context match",
                description="Warnings match the actual context instead of overreacting or underreacting.",
            ),
            MarkerDefinition(
                marker_id="cross_discipline_load_awareness",
                label="Cross-discipline load awareness",
                description="Cautions reflect total stress across swim, bike, and run rather than only one modality.",
            ),
            MarkerDefinition(
                marker_id="missing_data_handling",
                label="Missing-data handling",
                description="The plan reacts sensibly to sparse telemetry or uncertainty.",
            ),
        ),
        "actionability": (
            MarkerDefinition(
                marker_id="session_clarity",
                label="Session clarity",
                description="Sessions are concrete enough to execute.",
            ),
            MarkerDefinition(
                marker_id="followability",
                label="Followability",
                description="The week is practical to follow from day to day.",
            ),
            MarkerDefinition(
                marker_id="recovery_integration",
                label="Recovery integration",
                description="Recovery actions are useful and connected to the week.",
            ),
        ),
    },
}


def _normalize_style(style: str | None) -> str:
    s = (style or "").strip().lower()
    if s in _STYLE_MARKERS:
        return s
    return "trailrunning"


def _build_rubrics_for_style(style: str | None) -> tuple[RubricDefinition, ...]:
    s = _normalize_style(style)
    style_markers = _STYLE_MARKERS[s]
    rubrics: list[RubricDefinition] = []
    for meta in _BASE_RUBRIC_META:
        rubrics.append(
            RubricDefinition(
                rubric_id=str(meta["rubric_id"]),
                label=str(meta["label"]),
                weight=int(meta["weight"]),
                description=str(meta["description"]),
                markers=style_markers[str(meta["rubric_id"])],
            )
        )
    return tuple(rubrics)


DEFAULT_RUBRICS: tuple[RubricDefinition, ...] = _build_rubrics_for_style("trailrunning")


def get_default_rubrics(style: str | None = None) -> tuple[RubricDefinition, ...]:
    return _build_rubrics_for_style(style)


def rubric_map(
    rubrics: Iterable[RubricDefinition] | None = None,
    *,
    style: str | None = None,
) -> dict[str, RubricDefinition]:
    rr = tuple(rubrics or get_default_rubrics(style))
    return {rubric.rubric_id: rubric for rubric in rr}


def marker_map(
    rubrics: Iterable[RubricDefinition] | None = None,
    *,
    style: str | None = None,
) -> dict[str, MarkerDefinition]:
    out: dict[str, MarkerDefinition] = {}
    for rubric in tuple(rubrics or get_default_rubrics(style)):
        for marker in rubric.markers:
            out[marker.marker_id] = marker
    return out


def render_rubrics_for_prompt(
    rubrics: Iterable[RubricDefinition] | None = None,
    *,
    style: str | None = None,
    primary_goal: str | None = None,
) -> str:
    resolved_style = _normalize_style(style)
    resolved_goal = str(primary_goal or "").strip() or default_primary_goal_for_style(
        resolved_style
    )
    rr = tuple(rubrics or get_default_rubrics(resolved_style))
    lines = [
        f"Style: {resolved_style}",
        f"Primary goal: {resolved_goal}",
        "Score each rubric from 0 to 100.",
        "Score each marker from 0 to 5.",
        "Use pass / partial / fail for each marker.",
        "Reason from the provided plan and context only.",
        "",
    ]
    for rubric in rr:
        lines.append(f"- {rubric.rubric_id} ({rubric.weight}%): {rubric.description}")
        for marker in rubric.markers:
            lines.append(f"  - {marker.marker_id}: {marker.description}")
        lines.append("")
    return "\n".join(lines).strip()


def weighted_score_from_rubric_scores(
    rubric_scores: dict[str, Any],
    rubrics: Iterable[RubricDefinition] | None = None,
    *,
    style: str | None = None,
) -> float:
    rr = tuple(rubrics or get_default_rubrics(style))
    total_weight = sum(r.weight for r in rr) or 1
    weighted_total = 0.0
    for rubric in rr:
        raw = rubric_scores.get(rubric.rubric_id, 0)
        if isinstance(raw, dict):
            raw = raw.get("score", 0)
        try:
            score = float(raw)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(100.0, score))
        weighted_total += score * rubric.weight
    return round(weighted_total / total_weight, 1)


def grade_from_score(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"
