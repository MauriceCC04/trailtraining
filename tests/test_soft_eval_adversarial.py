"""
tests/test_soft_eval_adversarial.py

Adversarial soft-eval tests: deliberately bad training plans that must score
below defined thresholds.  All LLM calls are monkeypatched so the tests run
offline without API keys.

What the bad plan contains
--------------------------
- 3 consecutive hard days (intervals → tempo → hills), no rest day, and a 4th
  hard day on day 5.  The hard/easy constraint is violated wall-to-wall.
- No rest day in the entire 7-day block.
- Every ``purpose`` field is the generic placeholder "run to get fit", which
  deliberately fails the actionability and goal-specificity markers.
- weekly_totals claim 10 h of moving time but the sum of session durations is
  only ~4 h, so weekly_totals_arithmetic fails.
- Session types and workouts don't match (e.g. an "easy" session with a 5x1 km
  interval workout).

Expected behaviour
------------------
The soft evaluator must return:
- ``plan_coherence``   rubric score < 60
- ``actionability``    rubric score < 70

The thresholds are intentionally conservative; a plan this bad should score
well below them even across different model temperatures.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY = dt.date(2025, 6, 1)


def _day(
    *,
    date: dt.date,
    title: str,
    session_type: str,
    is_rest_day: bool,
    is_hard_day: bool,
    duration_minutes: int,
    target_intensity: str,
    terrain: str,
    workout: str,
    purpose: str = "run to get fit",
) -> dict[str, Any]:
    return {
        "date": date.isoformat(),
        "title": title,
        "session_type": session_type,
        "is_rest_day": is_rest_day,
        "is_hard_day": is_hard_day,
        "duration_minutes": duration_minutes,
        "target_intensity": target_intensity,
        "terrain": terrain,
        "workout": workout,
        "purpose": purpose,
        "signal_ids": [],
    }


@pytest.fixture()
def bad_plan() -> dict[str, Any]:
    """
    A training plan that violates every important quality marker:

    Day 1 (Mon): Hard - intervals (purpose: generic)
    Day 2 (Tue): Hard - tempo (purpose: generic)
    Day 3 (Wed): Hard - hills (purpose: generic)
    Day 4 (Thu): Labelled "easy" but workout is 5x1 km intervals (type mismatch)
    Day 5 (Fri): Hard - long run hammered at race pace (purpose: generic)
    Day 6 (Sat): Hard - strength (no trail rationale)
    Day 7 (Sun): Hard - aerobic (still no rest day)

    weekly_totals claims 600 min (10 h) but sum of durations = 240 min (4 h).
    """
    base = TODAY
    days = [
        _day(
            date=base,
            title="Interval blast",
            session_type="intervals",
            is_rest_day=False,
            is_hard_day=True,
            duration_minutes=50,
            target_intensity="95% max HR",
            terrain="road",
            workout="12x400 m at 5 km pace with 60 s rest",
            purpose="run to get fit",
        ),
        _day(
            date=base + dt.timedelta(days=1),
            title="Tempo Tuesday",
            session_type="tempo",
            is_rest_day=False,
            is_hard_day=True,
            duration_minutes=45,
            target_intensity="lactate threshold",
            terrain="road",
            workout="3x10 min at threshold",
            purpose="run to get fit",
        ),
        _day(
            date=base + dt.timedelta(days=2),
            title="Hill repeats",
            session_type="hills",
            is_rest_day=False,
            is_hard_day=True,
            duration_minutes=40,
            target_intensity="hard effort",
            terrain="trail",
            workout="10x90 s hill sprints",
            purpose="run to get fit",
        ),
        _day(
            # session_type says easy but workout is intervals - type/purpose mismatch
            date=base + dt.timedelta(days=3),
            title="Easy recovery",
            session_type="easy",
            is_rest_day=False,
            is_hard_day=False,
            duration_minutes=35,
            target_intensity="zone 2",
            terrain="road",
            workout="5x1 km at 5 km race pace",
            purpose="run to get fit",
        ),
        _day(
            date=base + dt.timedelta(days=4),
            title="Fast long run",
            session_type="long",
            is_rest_day=False,
            is_hard_day=True,
            duration_minutes=30,
            target_intensity="race pace",
            terrain="trail",
            workout="25 km at half-marathon effort",
            purpose="run to get fit",
        ),
        _day(
            date=base + dt.timedelta(days=5),
            title="Strength session",
            session_type="strength",
            is_rest_day=False,
            is_hard_day=True,
            duration_minutes=20,
            target_intensity="maximal",
            terrain="gym",
            workout="Leg press, squats, deadlifts - 5x5 at 90% 1RM",
            purpose="run to get fit",
        ),
        _day(
            date=base + dt.timedelta(days=6),
            title="Sunday aerobic",
            session_type="aerobic",
            is_rest_day=False,
            is_hard_day=True,
            duration_minutes=20,
            target_intensity="zone 3",
            terrain="road",
            workout="Steady jog",
            purpose="run to get fit",
        ),
    ]

    return {
        "meta": {
            "today": TODAY.isoformat(),
            "plan_start": TODAY.isoformat(),
            "plan_days": 7,
            "style": "trailrunning",
            "primary_goal": "run to get fit",
        },
        "snapshot": {
            "last7": {
                "distance_km": "30",
                "moving_time_hours": "4",
                "elevation_m": "500",
                "activity_count": "5",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "",
            },
            "baseline28": {
                "distance_km": "120",
                "moving_time_hours": "16",
                "elevation_m": "2000",
                "activity_count": "20",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "",
            },
            "notes": "All recovery signals missing.",
        },
        "readiness": {
            "status": "steady",
            "rationale": "No data available.",
            "signal_ids": [],
        },
        "plan": {
            # Claims 600 min = 10 h but actual sum is 240 min = 4 h
            "weekly_totals": {
                "planned_distance_km": 80.0,
                "planned_moving_time_hours": 10.0,
                "planned_elevation_m": 2500.0,
            },
            "days": days,
        },
        "recovery": {"actions": [], "signal_ids": []},
        "risks": [],
        "data_notes": ["No sleep or HRV data available."],
        "citations": [],
    }


@pytest.fixture()
def empty_deterministic_report() -> dict[str, Any]:
    """Minimal deterministic report used as context input to the soft evaluator."""
    return {
        "score": 30,
        "grade": "F",
        "subscores": {},
        "violations": [
            {
                "code": "CONSECUTIVE_HARD_DAYS",
                "severity": "high",
                "category": "load",
                "penalty": 20,
                "message": "3 consecutive hard days detected.",
                "details": {},
            },
        ],
        "soft_assessment": None,
    }


# ---------------------------------------------------------------------------
# Helpers - canned LLM responses
# ---------------------------------------------------------------------------


def _make_canned_marker_response(
    rubric_ids: list[str],
    *,
    plan_coherence_score: float = 1.5,
    actionability_score: float = 2.0,
    default_score: float = 2.5,
) -> str:
    """
    Build a canned JSON response for a batch rubric call.

    We deliberately return low scores for plan_coherence and actionability
    so that the evaluator's derived rubric scores will fall below the
    assertion thresholds.  All other rubrics get middling scores.
    """
    # Rough marker → rubric mapping for canned output generation
    from trailtraining.llm.rubrics import get_default_rubrics

    marker_results = []
    for rubric in get_default_rubrics("trailrunning"):
        if rubric.rubric_id not in rubric_ids:
            continue
        for marker in rubric.markers:
            if rubric.rubric_id == "plan_coherence":
                score = plan_coherence_score
            elif rubric.rubric_id == "actionability":
                score = actionability_score
            else:
                score = default_score

            verdict = "fail" if score < 2.0 else ("partial" if score < 4.0 else "pass")
            marker_results.append(
                {
                    "rubric": rubric.rubric_id,
                    "marker_id": marker.marker_id,
                    "marker": marker.label,
                    "observation": (
                        f"The plan shows problems with {marker.marker_id}: "
                        "3 consecutive hard days, no rest day, generic purposes."
                    ),
                    "verdict": verdict,
                    "score": score,
                    "evidence": "All purpose fields read 'run to get fit'. No rest day present.",
                    "improvement_hint": f"Improve {marker.marker_id} specifically.",
                }
            )
    return json.dumps({"marker_results": marker_results})


def _make_canned_synthesis_response() -> str:
    return json.dumps(
        {
            "summary": (
                "This plan has critical structural flaws: 3 consecutive hard days, "
                "no rest day, and entirely generic purposes provide no useful guidance."
            ),
            "confidence": "high",
            "strengths": ["Some sessions have specific workout prescriptions."],
            "concerns": [
                "No rest day across the entire 7-day block.",
                "All purpose fields are identical generic placeholders.",
                "weekly_totals claim 10 h but sessions total only 4 h.",
            ],
            "suggested_improvements": [
                "Add at least one rest day.",
                "Replace generic purposes with session-specific rationale.",
                "Fix weekly_totals to match actual session durations.",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for an OpenRouter API response object."""

    def __init__(self, text: str) -> None:
        self.output_text = text


def _make_call_with_schema_mock(
    *,
    plan_coherence_score: float = 1.5,
    actionability_score: float = 2.0,
) -> Any:
    """
    Return a mock for ``call_with_schema`` that inspects the schema name and
    returns an appropriate canned response.
    """

    def _side_effect(client: Any, kwargs: dict[str, Any], schema: dict[str, Any]) -> Any:
        schema_name: str = schema.get("name", "")

        if schema_name.startswith("trailtraining_batch_"):
            # Infer which rubric_ids are in scope from the batch name
            batch_name = schema_name.replace("trailtraining_batch_", "").replace("_v1", "")
            rubric_map = {
                "goal_coherence": ["goal_alignment", "plan_coherence"],
                "explanation": ["explanation_quality"],
                "caution": ["caution_proportionality"],
                "actionability": ["actionability"],
            }
            rubric_ids = rubric_map.get(batch_name, [])
            return _FakeResponse(
                _make_canned_marker_response(
                    rubric_ids,
                    plan_coherence_score=plan_coherence_score,
                    actionability_score=actionability_score,
                )
            )

        if schema_name == "trailtraining_soft_eval_synthesis_v1":
            return _FakeResponse(_make_canned_synthesis_response())

        # Fallback - return empty marker results
        return _FakeResponse(json.dumps({"marker_results": []}))

    return MagicMock(side_effect=_side_effect)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdversarialBadPlan:
    """
    The soft evaluator must detect a structurally broken plan and score it
    below the defined thresholds.
    """

    @pytest.mark.parametrize(
        "plan_coherence_score,actionability_score",
        [
            # Extreme case - nearly everything fails
            (1.0, 1.5),
            # Moderate case - still clearly below thresholds
            (2.0, 2.5),
        ],
    )
    def test_bad_plan_scores_below_thresholds(
        self,
        bad_plan: dict[str, Any],
        empty_deterministic_report: dict[str, Any],
        plan_coherence_score: float,
        actionability_score: float,
    ) -> None:
        """
        The soft evaluator must score plan_coherence < 60 and actionability < 70
        for a plan with 3 consecutive hard days, no rest, and generic purposes.

        Rubric scores are derived from marker averages:
          rubric_score = (mean_marker_score / 5.0) * 100

        So marker score 1.5 → rubric score 30, marker score 2.0 → rubric score 40,
        marker score 2.5 → rubric score 50 - all comfortably below the thresholds.
        """
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        cfg = SoftEvalConfig(enabled=True, model="anthropic/claude-sonnet-4")
        mock_call = _make_call_with_schema_mock(
            plan_coherence_score=plan_coherence_score,
            actionability_score=actionability_score,
        )

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch("trailtraining.llm.soft_eval.call_with_schema", mock_call),
        ):
            result = evaluate_training_plan_soft(
                bad_plan,
                empty_deterministic_report,
                rollups=None,
                cfg=cfg,
            )

        rubric_scores = result["rubric_scores"]
        plan_coherence = float(rubric_scores["plan_coherence"]["score"])
        actionability = float(rubric_scores["actionability"]["score"])

        assert plan_coherence < 60, (
            f"plan_coherence score {plan_coherence:.1f} should be < 60 for a plan with "
            "3 consecutive hard days and no rest day."
        )
        assert actionability < 70, (
            f"actionability score {actionability:.1f} should be < 70 for a plan with "
            "entirely generic 'run to get fit' purposes."
        )

    def test_bad_plan_marker_results_populated(
        self,
        bad_plan: dict[str, Any],
        empty_deterministic_report: dict[str, Any],
    ) -> None:
        """All expected markers must be present in marker_results."""
        from trailtraining.llm.rubrics import get_default_rubrics
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        cfg = SoftEvalConfig(enabled=True, model="anthropic/claude-sonnet-4")
        mock_call = _make_call_with_schema_mock()

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch("trailtraining.llm.soft_eval.call_with_schema", mock_call),
        ):
            result = evaluate_training_plan_soft(
                bad_plan,
                empty_deterministic_report,
                rollups=None,
                cfg=cfg,
            )

        returned_ids = {m["marker_id"] for m in result["marker_results"]}
        expected_ids = {
            marker.marker_id
            for rubric in get_default_rubrics("trailrunning")
            for marker in rubric.markers
        }
        missing = expected_ids - returned_ids
        assert not missing, f"marker_results missing entries for: {missing}"

    def test_bad_plan_observations_present(
        self,
        bad_plan: dict[str, Any],
        empty_deterministic_report: dict[str, Any],
    ) -> None:
        """
        observation fields must be populated - the batch schema requires them
        before the score to make scoring auditable.
        """
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        cfg = SoftEvalConfig(enabled=True, model="anthropic/claude-sonnet-4")
        mock_call = _make_call_with_schema_mock()

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch("trailtraining.llm.soft_eval.call_with_schema", mock_call),
        ):
            result = evaluate_training_plan_soft(
                bad_plan,
                empty_deterministic_report,
                rollups=None,
                cfg=cfg,
            )

        markers_with_obs = [m for m in result["marker_results"] if m.get("observation")]
        total = len(result["marker_results"])
        assert len(markers_with_obs) > 0, (
            "No marker_results have an observation field populated; "
            "the batch schema requires observation before score."
        )
        # Expect most markers to have observations (>50%)
        assert len(markers_with_obs) >= total // 2, (
            f"Only {len(markers_with_obs)}/{total} markers have observations; "
            "expected at least half."
        )

    def test_week_coherence_marker_present(
        self,
        bad_plan: dict[str, Any],
        empty_deterministic_report: dict[str, Any],
    ) -> None:
        """week_coherence must appear in marker_results for plan_coherence rubric."""
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        cfg = SoftEvalConfig(enabled=True, model="anthropic/claude-sonnet-4")
        mock_call = _make_call_with_schema_mock()

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch("trailtraining.llm.soft_eval.call_with_schema", mock_call),
        ):
            result = evaluate_training_plan_soft(
                bad_plan,
                empty_deterministic_report,
                rollups=None,
                cfg=cfg,
            )

        coherence_markers = [m for m in result["marker_results"] if m["rubric"] == "plan_coherence"]
        marker_ids = [m["marker_id"] for m in coherence_markers]
        assert "week_coherence" in marker_ids, (
            "week_coherence marker not found in plan_coherence marker_results. "
            f"Got: {marker_ids}"
        )

    def test_split_markers_present(
        self,
        bad_plan: dict[str, Any],
        empty_deterministic_report: dict[str, Any],
    ) -> None:
        """
        The split markers must all be present:
        - weekly_totals_arithmetic (was: weekly_internal_consistency)
        - session_type_purpose_alignment (was: weekly_internal_consistency)
        - missing_data_acknowledgment (was: missing_data_handling)
        - missing_data_behavioral_response (was: missing_data_handling)
        """
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        cfg = SoftEvalConfig(enabled=True, model="anthropic/claude-sonnet-4")
        mock_call = _make_call_with_schema_mock()

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch("trailtraining.llm.soft_eval.call_with_schema", mock_call),
        ):
            result = evaluate_training_plan_soft(
                bad_plan,
                empty_deterministic_report,
                rollups=None,
                cfg=cfg,
            )

        returned_ids = {m["marker_id"] for m in result["marker_results"]}
        required = {
            "weekly_totals_arithmetic",
            "session_type_purpose_alignment",
            "missing_data_acknowledgment",
            "missing_data_behavioral_response",
        }
        missing = required - returned_ids
        assert not missing, (
            f"Split markers not found in marker_results: {missing}. "
            "Ensure rubrics.py uses the new split names."
        )

    def test_failure_conditions_on_rubric_definitions(self) -> None:
        """
        failure_condition fields must be set on the expected markers.
        """
        from trailtraining.llm.rubrics import get_default_rubrics

        mmap = {m.marker_id: m for r in get_default_rubrics("trailrunning") for m in r.markers}

        lp = mmap.get("load_progression_logic")
        assert lp is not None, "load_progression_logic marker not found"
        assert (
            "15%" in lp.failure_condition
        ), f"load_progression_logic.failure_condition should mention 15%: {lp.failure_condition!r}"

        nc = mmap.get("non_competing_focus")
        assert nc is not None, "non_competing_focus marker not found"
        assert (
            "non-running" in nc.failure_condition.lower()
            or "non_running" in nc.failure_condition.lower()
            or "trail" in nc.failure_condition.lower()
        ), f"non_competing_focus.failure_condition looks wrong: {nc.failure_condition!r}"

    def test_per_rubric_batch_called_once_per_batch(
        self,
        bad_plan: dict[str, Any],
        empty_deterministic_report: dict[str, Any],
    ) -> None:
        """
        The evaluator must issue exactly 4 batch calls + 1 synthesis call
        (one per _RUBRIC_BATCHES entry plus the synthesis call).
        """
        from trailtraining.llm.soft_eval import (
            _RUBRIC_BATCHES,
            SoftEvalConfig,
            evaluate_training_plan_soft,
        )

        cfg = SoftEvalConfig(enabled=True, model="anthropic/claude-sonnet-4")
        mock_call = _make_call_with_schema_mock()

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch("trailtraining.llm.soft_eval.call_with_schema", mock_call),
        ):
            evaluate_training_plan_soft(
                bad_plan,
                empty_deterministic_report,
                rollups=None,
                cfg=cfg,
            )

        expected_calls = len(_RUBRIC_BATCHES) + 1  # batches + synthesis
        actual_calls = mock_call.call_count
        assert actual_calls == expected_calls, (
            f"Expected {expected_calls} LLM calls (4 batches + 1 synthesis), "
            f"got {actual_calls}."
        )


class TestComparePlans:
    """compare_plans returns a valid preference dict."""

    def test_compare_plans_returns_preference(
        self,
        bad_plan: dict[str, Any],
    ) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig, compare_plans

        canned = json.dumps(
            {
                "preferred": "plan_b",
                "reasoning": "Plan B has rest days and specific session purposes.",
                "plan_a_advantages": ["More total volume."],
                "plan_b_advantages": ["Has a rest day.", "Specific purposes."],
            }
        )

        cfg = SoftEvalConfig(enabled=True, model="anthropic/claude-sonnet-4")

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch(
                "trailtraining.llm.soft_eval.call_with_schema",
                return_value=_FakeResponse(canned),
            ),
        ):
            result = compare_plans(bad_plan, bad_plan, rollups=None, cfg=cfg)

        assert result["preferred"] in {"plan_a", "plan_b", "tie"}
        assert isinstance(result["reasoning"], str) and result["reasoning"]
        assert isinstance(result["plan_a_advantages"], list)
        assert isinstance(result["plan_b_advantages"], list)
