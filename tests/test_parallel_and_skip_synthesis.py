"""
Tests for the parallel batch execution and skip_synthesis features
in soft_eval.py.
"""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from trailtraining.llm.rubrics import get_default_rubrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_marker_results_for_rubrics(rubric_ids: list[str], score: float = 3.0) -> str:
    """Build a canned JSON batch response for the given rubric_ids."""
    results = []
    for rubric in get_default_rubrics("trailrunning"):
        if rubric.rubric_id not in rubric_ids:
            continue
        for marker in rubric.markers:
            verdict = "pass" if score >= 4.0 else ("partial" if score >= 2.0 else "fail")
            results.append(
                {
                    "rubric": rubric.rubric_id,
                    "marker_id": marker.marker_id,
                    "marker": marker.label,
                    "observation": f"Observation for {marker.marker_id}.",
                    "verdict": verdict,
                    "score": score,
                    "evidence": f"Evidence for {marker.marker_id}.",
                    "improvement_hint": f"Improve {marker.marker_id}.",
                }
            )
    return json.dumps({"marker_results": results})


def _make_synthesis_response() -> str:
    return json.dumps(
        {
            "summary": "Test synthesis summary.",
            "confidence": "medium",
            "strengths": ["Strength 1", "Strength 2"],
            "concerns": ["Concern 1"],
            "suggested_improvements": ["Improvement 1", "Improvement 2"],
        }
    )


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.output_text = text


_BATCH_MAP = {
    "goal_coherence": ["goal_alignment", "plan_coherence"],
    "explanation": ["explanation_quality"],
    "caution": ["caution_proportionality"],
    "actionability": ["actionability"],
}


def _mock_call_with_schema(
    client: Any, kwargs: dict[str, Any], schema: dict[str, Any]
) -> _FakeResponse:
    schema_name: str = schema.get("name", "")
    if schema_name.startswith("trailtraining_batch_"):
        batch = schema_name.replace("trailtraining_batch_", "").replace("_v1", "")
        rubric_ids = _BATCH_MAP.get(batch, [])
        return _FakeResponse(_make_marker_results_for_rubrics(rubric_ids))
    if schema_name == "trailtraining_soft_eval_synthesis_v1":
        return _FakeResponse(_make_synthesis_response())
    return _FakeResponse(json.dumps({"marker_results": []}))


def _minimal_plan() -> dict[str, Any]:
    return {
        "meta": {
            "today": "2026-03-13",
            "plan_start": "2026-03-14",
            "plan_days": 7,
            "style": "trailrunning",
            "primary_goal": "to become a faster trail runner",
        },
        "snapshot": {
            "last7": {
                "distance_km": "22",
                "moving_time_hours": "2.2",
                "elevation_m": "300",
                "activity_count": "2",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "46",
            },
            "baseline28": {
                "distance_km": "18",
                "moving_time_hours": "1.9",
                "elevation_m": "220",
                "activity_count": "2",
                "sleep_hours_mean": "",
                "hrv_mean": "",
                "rhr_mean": "45",
            },
            "notes": "",
        },
        "readiness": {"status": "steady", "rationale": "Test.", "signal_ids": []},
        "plan": {
            "weekly_totals": {
                "planned_distance_km": 30.0,
                "planned_moving_time_hours": 4.5,
                "planned_elevation_m": 600.0,
            },
            "days": [
                {
                    "date": "2026-03-14",
                    "title": "Easy run",
                    "session_type": "easy",
                    "is_rest_day": False,
                    "is_hard_day": False,
                    "duration_minutes": 45,
                    "target_intensity": "easy",
                    "terrain": "road",
                    "workout": "45 min easy",
                    "purpose": "aerobic maintenance",
                    "signal_ids": [],
                }
            ],
        },
        "recovery": {"actions": [], "signal_ids": []},
        "risks": [],
        "data_notes": [],
        "citations": [],
    }


def _minimal_deterministic_report() -> dict[str, Any]:
    return {"score": 100, "grade": "A", "subscores": {}, "violations": []}


# ---------------------------------------------------------------------------
# Tests: parallel vs sequential produce same marker set
# ---------------------------------------------------------------------------


class TestParallelBatches:
    def test_parallel_produces_all_expected_markers(self) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        cfg = SoftEvalConfig(enabled=True, parallel_batches=True)

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch(
                "trailtraining.llm.soft_eval.call_with_schema",
                side_effect=_mock_call_with_schema,
            ),
        ):
            result = evaluate_training_plan_soft(
                _minimal_plan(), _minimal_deterministic_report(), None, cfg
            )

        expected_ids = {m.marker_id for r in get_default_rubrics("trailrunning") for m in r.markers}
        returned_ids = {m["marker_id"] for m in result["marker_results"]}
        assert expected_ids == returned_ids

    def test_sequential_produces_all_expected_markers(self) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        cfg = SoftEvalConfig(enabled=True, parallel_batches=False)

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch(
                "trailtraining.llm.soft_eval.call_with_schema",
                side_effect=_mock_call_with_schema,
            ),
        ):
            result = evaluate_training_plan_soft(
                _minimal_plan(), _minimal_deterministic_report(), None, cfg
            )

        expected_ids = {m.marker_id for r in get_default_rubrics("trailrunning") for m in r.markers}
        returned_ids = {m["marker_id"] for m in result["marker_results"]}
        assert expected_ids == returned_ids

    def test_parallel_and_sequential_produce_same_scores(self) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        plan = _minimal_plan()
        report = _minimal_deterministic_report()

        results = {}
        for mode, parallel in [("parallel", True), ("sequential", False)]:
            cfg = SoftEvalConfig(enabled=True, parallel_batches=parallel)
            with (
                patch(
                    "trailtraining.llm.soft_eval.make_openrouter_client",
                    return_value=MagicMock(),
                ),
                patch(
                    "trailtraining.llm.soft_eval.call_with_schema",
                    side_effect=_mock_call_with_schema,
                ),
            ):
                results[mode] = evaluate_training_plan_soft(plan, report, None, cfg)

        assert results["parallel"]["overall_score"] == results["sequential"]["overall_score"]
        assert results["parallel"]["rubric_scores"] == results["sequential"]["rubric_scores"]

    def test_parallel_actually_uses_threads(self) -> None:
        """Verify that batch calls happen on different threads."""
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        thread_ids: list[int] = []
        lock = threading.Lock()
        barrier = threading.Barrier(4, timeout=5)

        def _tracking_mock(client: Any, kwargs: dict[str, Any], schema: dict[str, Any]) -> Any:
            with lock:
                thread_ids.append(threading.current_thread().ident or 0)
            barrier.wait()  # force all 4 batch threads to overlap
            return _mock_call_with_schema(client, kwargs, schema)

        cfg = SoftEvalConfig(enabled=True, parallel_batches=True)

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch(
                "trailtraining.llm.soft_eval.call_with_schema",
                side_effect=_tracking_mock,
            ),
        ):
            evaluate_training_plan_soft(_minimal_plan(), _minimal_deterministic_report(), None, cfg)

        # 4 batch calls + 1 synthesis = 5 calls total
        assert len(thread_ids) == 5
        # At least 2 distinct thread IDs among the batch calls (first 4)
        batch_threads = set(thread_ids[:4])
        assert (
            len(batch_threads) >= 2
        ), f"Expected multiple threads for parallel batches, got thread IDs: {thread_ids[:4]}"

    def test_parallel_handles_single_batch_failure(self) -> None:
        """One batch failing should not crash the whole evaluation."""
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        call_count = 0

        def _failing_mock(client: Any, kwargs: dict[str, Any], schema: dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            schema_name = schema.get("name", "")
            if "caution" in schema_name:
                raise RuntimeError("Simulated API failure")
            return _mock_call_with_schema(client, kwargs, schema)

        cfg = SoftEvalConfig(enabled=True, parallel_batches=True)

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch(
                "trailtraining.llm.soft_eval.call_with_schema",
                side_effect=_failing_mock,
            ),
        ):
            result = evaluate_training_plan_soft(
                _minimal_plan(), _minimal_deterministic_report(), None, cfg
            )

        # Should still have results from the other 3 batches
        returned_ids = {m["marker_id"] for m in result["marker_results"]}
        assert "goal_specificity" in returned_ids  # from goal_coherence batch
        assert "session_clarity" in returned_ids  # from actionability batch
        # Caution markers should have score 0 (defaults from normalization)
        caution_markers = [
            m for m in result["marker_results"] if m["rubric"] == "caution_proportionality"
        ]
        assert all(m["score"] == 0.0 for m in caution_markers)


# ---------------------------------------------------------------------------
# Tests: skip_synthesis
# ---------------------------------------------------------------------------


class TestSkipSynthesis:
    def test_skip_synthesis_skips_llm_call(self) -> None:
        """With skip_synthesis=True, the synthesis LLM call should not be made."""
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        call_schemas: list[str] = []

        def _tracking_mock(client: Any, kwargs: dict[str, Any], schema: dict[str, Any]) -> Any:
            call_schemas.append(schema.get("name", ""))
            return _mock_call_with_schema(client, kwargs, schema)

        cfg = SoftEvalConfig(enabled=True, skip_synthesis=True, parallel_batches=False)

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch(
                "trailtraining.llm.soft_eval.call_with_schema",
                side_effect=_tracking_mock,
            ),
        ):
            evaluate_training_plan_soft(_minimal_plan(), _minimal_deterministic_report(), None, cfg)

        # Should have 4 batch calls but NO synthesis call
        assert len(call_schemas) == 4
        assert "trailtraining_soft_eval_synthesis_v1" not in call_schemas

    def test_skip_synthesis_still_has_feedback_fields(self) -> None:
        """Even without synthesis, strengths/concerns/improvements should be populated."""
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        cfg = SoftEvalConfig(enabled=True, skip_synthesis=True)

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch(
                "trailtraining.llm.soft_eval.call_with_schema",
                side_effect=_mock_call_with_schema,
            ),
        ):
            result = evaluate_training_plan_soft(
                _minimal_plan(), _minimal_deterministic_report(), None, cfg
            )

        # Feedback fields derived locally
        assert len(result["strengths"]) >= 2
        assert len(result["concerns"]) >= 1
        assert len(result["suggested_improvements"]) >= 2
        # derived_fields should indicate synthesis was skipped
        assert "summary" in result["derived_fields"]
        assert "strengths" in result["derived_fields"]
        assert "concerns" in result["derived_fields"]
        assert "suggested_improvements" in result["derived_fields"]

    def test_skip_synthesis_saves_one_call_vs_full(self) -> None:
        """Verify the call count difference between skip and no-skip."""
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        for skip, expected_calls in [(False, 5), (True, 4)]:
            mock_fn = MagicMock(side_effect=_mock_call_with_schema)
            cfg = SoftEvalConfig(enabled=True, skip_synthesis=skip, parallel_batches=False)

            with (
                patch(
                    "trailtraining.llm.soft_eval.make_openrouter_client",
                    return_value=MagicMock(),
                ),
                patch("trailtraining.llm.soft_eval.call_with_schema", mock_fn),
            ):
                evaluate_training_plan_soft(
                    _minimal_plan(), _minimal_deterministic_report(), None, cfg
                )

            assert mock_fn.call_count == expected_calls, (
                f"skip_synthesis={skip}: expected {expected_calls} calls, "
                f"got {mock_fn.call_count}"
            )

    def test_no_skip_synthesis_uses_llm_narrative(self) -> None:
        """Without skip_synthesis, summary should come from the LLM."""
        from trailtraining.llm.soft_eval import SoftEvalConfig, evaluate_training_plan_soft

        cfg = SoftEvalConfig(enabled=True, skip_synthesis=False)

        with (
            patch(
                "trailtraining.llm.soft_eval.make_openrouter_client",
                return_value=MagicMock(),
            ),
            patch(
                "trailtraining.llm.soft_eval.call_with_schema",
                side_effect=_mock_call_with_schema,
            ),
        ):
            result = evaluate_training_plan_soft(
                _minimal_plan(), _minimal_deterministic_report(), None, cfg
            )

        assert result["summary"] == "Test synthesis summary."


# ---------------------------------------------------------------------------
# Tests: SoftEvalConfig env var handling
# ---------------------------------------------------------------------------


class TestSoftEvalConfigEnv:
    def test_skip_synthesis_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig

        monkeypatch.setenv("TRAILTRAINING_SOFT_EVAL_SKIP_SYNTHESIS", "true")
        cfg = SoftEvalConfig.from_env()
        assert cfg.skip_synthesis is True

    def test_skip_synthesis_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig

        monkeypatch.delenv("TRAILTRAINING_SOFT_EVAL_SKIP_SYNTHESIS", raising=False)
        cfg = SoftEvalConfig.from_env()
        assert cfg.skip_synthesis is False

    def test_no_parallel_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig

        monkeypatch.setenv("TRAILTRAINING_SOFT_EVAL_NO_PARALLEL", "1")
        cfg = SoftEvalConfig.from_env()
        assert cfg.parallel_batches is False

    def test_parallel_default_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig

        monkeypatch.delenv("TRAILTRAINING_SOFT_EVAL_NO_PARALLEL", raising=False)
        cfg = SoftEvalConfig.from_env()
        assert cfg.parallel_batches is True


# ---------------------------------------------------------------------------
# Tests: CLI flag parsing
# ---------------------------------------------------------------------------


class TestCLIFlags:
    def test_skip_synthesis_flag_parsed(self) -> None:
        from trailtraining.commands.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["eval-coach", "--soft-eval", "--skip-synthesis"])
        assert args.skip_synthesis is True

    def test_no_parallel_batches_flag_parsed(self) -> None:
        from trailtraining.commands.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["eval-coach", "--soft-eval", "--no-parallel-batches"])
        assert args.no_parallel_batches is True

    def test_defaults_without_flags(self) -> None:
        from trailtraining.commands.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["eval-coach", "--soft-eval"])
        assert args.skip_synthesis is False
        assert args.no_parallel_batches is False
