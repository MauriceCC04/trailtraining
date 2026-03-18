"""
Tests for the lifestyle_notes feature.

Verifies that schedule/lifestyle constraints:
1. Are stored in meta.lifestyle_notes in the plan artifact
2. Are injected into the soft evaluator prompts
3. Are read from env var, CLI flag, or plan meta (priority order)
4. Appear in the human-readable text rendering
5. Are preserved through revise-plan
"""

from __future__ import annotations

import pytest

SAMPLE_LIFESTYLE = (
    "Weekdays: road runs or cycling only. One long mountain day on Saturday or Sunday."
)


# ---------------------------------------------------------------------------
# contracts.py — TrainingMeta accepts lifestyle_notes
# ---------------------------------------------------------------------------


class TestTrainingMetaLifestyleNotes:
    def test_default_empty_string(self) -> None:
        from trailtraining.contracts import TrainingMeta

        meta = TrainingMeta(
            today="2026-03-13",
            plan_start="2026-03-14",
            plan_days=7,
            style="trailrunning",
        )
        assert meta.lifestyle_notes == ""

    def test_stores_lifestyle_notes(self) -> None:
        from trailtraining.contracts import TrainingMeta

        meta = TrainingMeta(
            today="2026-03-13",
            plan_start="2026-03-14",
            plan_days=7,
            style="trailrunning",
            lifestyle_notes=SAMPLE_LIFESTYLE,
        )
        assert meta.lifestyle_notes == SAMPLE_LIFESTYLE

    def test_roundtrips_through_json(self) -> None:
        from trailtraining.contracts import TrainingPlanArtifact

        payload = {
            "meta": {
                "today": "2026-03-13",
                "plan_start": "2026-03-14",
                "plan_days": 7,
                "style": "trailrunning",
                "primary_goal": "test",
                "lifestyle_notes": SAMPLE_LIFESTYLE,
            },
            "snapshot": {
                "last7": {
                    "distance_km": "",
                    "moving_time_hours": "",
                    "elevation_m": "",
                    "activity_count": "",
                    "sleep_hours_mean": "",
                    "hrv_mean": "",
                    "rhr_mean": "",
                },
                "baseline28": {
                    "distance_km": "",
                    "moving_time_hours": "",
                    "elevation_m": "",
                    "activity_count": "",
                    "sleep_hours_mean": "",
                    "hrv_mean": "",
                    "rhr_mean": "",
                },
                "notes": "",
            },
            "readiness": {"status": "steady", "rationale": "Test.", "signal_ids": []},
            "plan": {
                "weekly_totals": {
                    "planned_distance_km": 0,
                    "planned_moving_time_hours": 0,
                    "planned_elevation_m": 0,
                },
                "days": [
                    {
                        "date": "2026-03-14",
                        "title": "Rest",
                        "session_type": "rest",
                        "is_rest_day": True,
                        "is_hard_day": False,
                        "duration_minutes": 0,
                        "target_intensity": "Off",
                        "terrain": "N/A",
                        "workout": "Rest",
                        "purpose": "Recovery",
                        "signal_ids": [],
                    }
                ],
            },
            "recovery": {"actions": [], "signal_ids": []},
            "risks": [],
            "data_notes": [],
            "citations": [],
        }
        artifact = TrainingPlanArtifact.model_validate(payload)
        dumped = artifact.model_dump(mode="json")
        assert dumped["meta"]["lifestyle_notes"] == SAMPLE_LIFESTYLE

        # Roundtrip
        artifact2 = TrainingPlanArtifact.model_validate(dumped)
        assert artifact2.meta.lifestyle_notes == SAMPLE_LIFESTYLE

    def test_missing_lifestyle_notes_defaults_empty(self) -> None:
        """Plans without lifestyle_notes (backward compat) should default to ''."""
        from trailtraining.contracts import TrainingPlanArtifact

        payload = {
            "meta": {
                "today": "2026-03-13",
                "plan_start": "2026-03-14",
                "plan_days": 7,
                "style": "trailrunning",
                "primary_goal": "test",
                # no lifestyle_notes key
            },
            "snapshot": {
                "last7": {
                    "distance_km": "",
                    "moving_time_hours": "",
                    "elevation_m": "",
                    "activity_count": "",
                    "sleep_hours_mean": "",
                    "hrv_mean": "",
                    "rhr_mean": "",
                },
                "baseline28": {
                    "distance_km": "",
                    "moving_time_hours": "",
                    "elevation_m": "",
                    "activity_count": "",
                    "sleep_hours_mean": "",
                    "hrv_mean": "",
                    "rhr_mean": "",
                },
                "notes": "",
            },
            "readiness": {"status": "steady", "rationale": "Test.", "signal_ids": []},
            "plan": {
                "weekly_totals": {
                    "planned_distance_km": 0,
                    "planned_moving_time_hours": 0,
                    "planned_elevation_m": 0,
                },
                "days": [
                    {
                        "date": "2026-03-14",
                        "title": "Rest",
                        "session_type": "rest",
                        "is_rest_day": True,
                        "is_hard_day": False,
                        "duration_minutes": 0,
                        "target_intensity": "Off",
                        "terrain": "N/A",
                        "workout": "Rest",
                        "purpose": "Recovery",
                        "signal_ids": [],
                    }
                ],
            },
            "recovery": {"actions": [], "signal_ids": []},
            "risks": [],
            "data_notes": [],
            "citations": [],
        }
        artifact = TrainingPlanArtifact.model_validate(payload)
        assert artifact.meta.lifestyle_notes == ""


# ---------------------------------------------------------------------------
# coach.py — lifestyle_notes flows into CoachConfig and prompt
# ---------------------------------------------------------------------------


class TestCoachLifestyleNotes:
    def test_coach_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trailtraining.llm.coach import CoachConfig

        monkeypatch.setenv("TRAILTRAINING_LIFESTYLE_NOTES", SAMPLE_LIFESTYLE)
        cfg = CoachConfig.from_env()
        assert cfg.lifestyle_notes == SAMPLE_LIFESTYLE

    def test_coach_config_default_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trailtraining.llm.coach import CoachConfig

        monkeypatch.delenv("TRAILTRAINING_LIFESTYLE_NOTES", raising=False)
        cfg = CoachConfig.from_env()
        assert cfg.lifestyle_notes == ""

    def test_lifestyle_notes_in_prompt_text(self) -> None:
        from trailtraining.llm.coach import _lifestyle_notes_section

        section = _lifestyle_notes_section(SAMPLE_LIFESTYLE)
        joined = "\n".join(section)
        assert "Lifestyle constraints" in joined
        assert "road runs or cycling only" in joined
        assert "meta.lifestyle_notes" in joined

    def test_lifestyle_notes_absent_produces_empty_section(self) -> None:
        from trailtraining.llm.coach import _lifestyle_notes_section

        assert _lifestyle_notes_section("") == []
        assert _lifestyle_notes_section("   ") == []

    def test_apply_lifestyle_notes_sets_meta(self) -> None:
        from trailtraining.llm.coach import _apply_lifestyle_notes

        plan = {"meta": {"style": "trailrunning"}}
        _apply_lifestyle_notes(plan, SAMPLE_LIFESTYLE)
        assert plan["meta"]["lifestyle_notes"] == SAMPLE_LIFESTYLE

    def test_apply_lifestyle_notes_empty(self) -> None:
        from trailtraining.llm.coach import _apply_lifestyle_notes

        plan = {"meta": {"style": "trailrunning"}}
        _apply_lifestyle_notes(plan, "")
        assert plan["meta"]["lifestyle_notes"] == ""


# ---------------------------------------------------------------------------
# soft_eval.py — lifestyle_notes extracted and injected into prompts
# ---------------------------------------------------------------------------


class TestSoftEvalLifestyleNotes:
    def test_resolve_from_config(self) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig, _resolve_style_goal_and_lifestyle

        plan = {
            "meta": {
                "style": "trailrunning",
                "primary_goal": "test",
                "lifestyle_notes": "from plan",
            }
        }
        cfg = SoftEvalConfig(enabled=True, lifestyle_notes="from config")
        _, _, notes = _resolve_style_goal_and_lifestyle(plan, cfg)
        assert notes == "from config"  # config takes priority

    def test_resolve_from_plan_meta(self) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig, _resolve_style_goal_and_lifestyle

        plan = {
            "meta": {
                "style": "trailrunning",
                "primary_goal": "test",
                "lifestyle_notes": SAMPLE_LIFESTYLE,
            }
        }
        cfg = SoftEvalConfig(enabled=True, lifestyle_notes="")
        _, _, notes = _resolve_style_goal_and_lifestyle(plan, cfg)
        assert notes == SAMPLE_LIFESTYLE

    def test_resolve_empty_when_neither(self) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig, _resolve_style_goal_and_lifestyle

        plan = {"meta": {"style": "trailrunning", "primary_goal": "test"}}
        cfg = SoftEvalConfig(enabled=True)
        _, _, notes = _resolve_style_goal_and_lifestyle(plan, cfg)
        assert notes == ""

    def test_lifestyle_context_for_eval_content(self) -> None:
        from trailtraining.llm.soft_eval import _lifestyle_context_for_eval

        section = _lifestyle_context_for_eval(SAMPLE_LIFESTYLE)
        joined = "\n".join(section)
        assert "Lifestyle constraints" in joined
        assert "trail_specificity" in joined
        assert "non_competing_focus" in joined
        assert "score the plan against the BEST plan possible GIVEN these constraints" in joined

    def test_lifestyle_context_empty_when_no_notes(self) -> None:
        from trailtraining.llm.soft_eval import _lifestyle_context_for_eval

        assert _lifestyle_context_for_eval("") == []
        assert _lifestyle_context_for_eval("  ") == []

    def test_lifestyle_notes_in_batch_prompt(self) -> None:
        from trailtraining.llm.soft_eval import _build_batch_prompt

        prompt = _build_batch_prompt(
            ["goal_alignment"],
            {"meta": {"style": "trailrunning"}},
            {"score": 100, "violations": []},
            None,
            style="trailrunning",
            primary_goal="test",
            lifestyle_notes=SAMPLE_LIFESTYLE,
        )
        assert "road runs or cycling only" in prompt
        assert "trail_specificity" in prompt

    def test_no_lifestyle_notes_in_batch_prompt(self) -> None:
        from trailtraining.llm.soft_eval import _build_batch_prompt

        prompt = _build_batch_prompt(
            ["goal_alignment"],
            {"meta": {"style": "trailrunning"}},
            {"score": 100, "violations": []},
            None,
            style="trailrunning",
            primary_goal="test",
            lifestyle_notes="",
        )
        assert "Lifestyle constraints" not in prompt

    def test_lifestyle_notes_in_synthesis_prompt(self) -> None:
        from trailtraining.llm.soft_eval import _build_synthesis_prompt

        prompt = _build_synthesis_prompt(
            {"meta": {"style": "trailrunning"}},
            [],
            None,
            style="trailrunning",
            primary_goal="test",
            lifestyle_notes=SAMPLE_LIFESTYLE,
        )
        assert "road runs or cycling only" in prompt

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trailtraining.llm.soft_eval import SoftEvalConfig

        monkeypatch.setenv("TRAILTRAINING_LIFESTYLE_NOTES", SAMPLE_LIFESTYLE)
        cfg = SoftEvalConfig.from_env()
        assert cfg.lifestyle_notes == SAMPLE_LIFESTYLE


# ---------------------------------------------------------------------------
# revise.py — lifestyle_notes flows through revision
# ---------------------------------------------------------------------------


class TestReviseLifestyleNotes:
    def test_revise_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trailtraining.llm.revise import RevisePlanConfig

        monkeypatch.setenv("TRAILTRAINING_LIFESTYLE_NOTES", SAMPLE_LIFESTYLE)
        cfg = RevisePlanConfig.from_env()
        assert cfg.lifestyle_notes == SAMPLE_LIFESTYLE

    def test_lifestyle_notes_in_revise_prompt(self) -> None:
        from trailtraining.llm.revise import _lifestyle_notes_for_revise

        section = _lifestyle_notes_for_revise(SAMPLE_LIFESTYLE)
        joined = "\n".join(section)
        assert "Lifestyle constraints" in joined
        assert "road runs or cycling only" in joined

    def test_no_lifestyle_notes_in_revise_prompt(self) -> None:
        from trailtraining.llm.revise import _lifestyle_notes_for_revise

        assert _lifestyle_notes_for_revise("") == []


# ---------------------------------------------------------------------------
# shared.py — text rendering includes lifestyle notes
# ---------------------------------------------------------------------------


class TestTextRendering:
    def test_text_output_includes_lifestyle_notes(self) -> None:
        from trailtraining.llm.shared import training_plan_to_text

        plan = {
            "meta": {
                "today": "2026-03-13",
                "plan_start": "2026-03-14",
                "plan_days": 7,
                "style": "trailrunning",
                "primary_goal": "test",
                "lifestyle_notes": SAMPLE_LIFESTYLE,
            },
            "readiness": {"status": "steady", "rationale": "Test."},
            "plan": {
                "weekly_totals": {"planned_moving_time_hours": 4.5},
                "days": [],
            },
            "recovery": {"actions": []},
            "risks": [],
        }
        text = training_plan_to_text(plan)
        assert "Lifestyle constraints:" in text
        assert "road runs or cycling only" in text

    def test_text_output_omits_when_empty(self) -> None:
        from trailtraining.llm.shared import training_plan_to_text

        plan = {
            "meta": {
                "today": "2026-03-13",
                "plan_start": "2026-03-14",
                "plan_days": 7,
                "style": "trailrunning",
                "primary_goal": "test",
                "lifestyle_notes": "",
            },
            "readiness": {"status": "steady", "rationale": "Test."},
            "plan": {"weekly_totals": {"planned_moving_time_hours": 4.5}, "days": []},
            "recovery": {"actions": []},
            "risks": [],
        }
        text = training_plan_to_text(plan)
        assert "Lifestyle constraints" not in text


# ---------------------------------------------------------------------------
# CLI — parser accepts --lifestyle-notes on all three commands
# ---------------------------------------------------------------------------


class TestCLILifestyleNotesFlag:
    def test_coach_accepts_lifestyle_notes(self) -> None:
        from trailtraining.commands.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "coach",
                "--prompt",
                "training-plan",
                "--lifestyle-notes",
                SAMPLE_LIFESTYLE,
            ]
        )
        assert args.lifestyle_notes == SAMPLE_LIFESTYLE

    def test_eval_coach_accepts_lifestyle_notes(self) -> None:
        from trailtraining.commands.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "eval-coach",
                "--soft-eval",
                "--lifestyle-notes",
                SAMPLE_LIFESTYLE,
            ]
        )
        assert args.lifestyle_notes == SAMPLE_LIFESTYLE

    def test_revise_plan_accepts_lifestyle_notes(self) -> None:
        from trailtraining.commands.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "revise-plan",
                "--lifestyle-notes",
                SAMPLE_LIFESTYLE,
            ]
        )
        assert args.lifestyle_notes == SAMPLE_LIFESTYLE

    def test_default_is_none(self) -> None:
        from trailtraining.commands.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["coach", "--prompt", "training-plan"])
        assert args.lifestyle_notes is None


# ---------------------------------------------------------------------------
# schemas.py — lifestyle_notes in the JSON schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_includes_lifestyle_notes(self) -> None:
        from trailtraining.llm.schemas import TRAINING_PLAN_SCHEMA

        meta_props = TRAINING_PLAN_SCHEMA["schema"]["properties"]["meta"]["properties"]
        assert "lifestyle_notes" in meta_props
        assert meta_props["lifestyle_notes"]["type"] == "string"

    def test_output_contract_mentions_lifestyle_notes(self) -> None:
        from trailtraining.llm.schemas import training_plan_output_contract_text

        text = training_plan_output_contract_text()
        assert "lifestyle_notes" in text
