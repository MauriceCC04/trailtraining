# src/trailtraining/llm/coach.py

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from trailtraining import config
from trailtraining.util.state import load_json, save_json


def _as_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _coerce_path(p: Optional[str]) -> Optional[Path]:
    return Path(p).expanduser().resolve() if p else None


def _resolve_input_paths(
    input_path: Optional[str],
    personal_path: Optional[str],
    summary_path: Optional[str],
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Mirrors your CLI behavior:
      - if --personal/--summary provided, use them
      - else use --input (dir) or prompting directory
    """
    base: Path
    if input_path:
        base = Path(input_path).expanduser().resolve()
    else:
        base = Path(config.PROMPTING_DIRECTORY).expanduser().resolve()

    if base.is_file() and base.suffix.lower() == ".zip":
        raise RuntimeError("Zip input not supported in this optimized version. Use a directory path.")

    personal = _coerce_path(personal_path) or (base / "formatted_personal_data.json")
    summary = _coerce_path(summary_path) or (base / "combined_summary.json")
    rollups = base / "combined_rollups.json"
    return personal, summary, (rollups if rollups.exists() else None)


def _dedup_activities_in_place(combined: List[Dict[str, Any]]) -> None:
    seen = set()
    for day in combined:
        acts = day.get("activities")
        if not isinstance(acts, list):
            day["activities"] = []
            continue
        new_acts = []
        for a in acts:
            if not isinstance(a, dict):
                continue
            aid = a.get("id")
            if aid is None:
                new_acts.append(a)
                continue
            key = str(aid)
            if key in seen:
                continue
            seen.add(key)
            new_acts.append(a)
        day["activities"] = new_acts


def _filter_last_days(combined: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
    if days <= 0:
        return combined
    if not combined:
        return combined

    # assume combined sorted by date asc (as produced by combine.py)
    last = _as_date(combined[-1].get("date", ""))
    if not last:
        return combined

    cutoff = last - timedelta(days=days - 1)
    out = []
    for d in combined:
        ds = d.get("date")
        if not isinstance(ds, str):
            continue
        dd = _as_date(ds)
        if dd and dd >= cutoff:
            out.append(d)
    return out


def _summarize_activity(a: Dict[str, Any]) -> str:
    sport = a.get("sport_type") or a.get("type") or "unknown"
    dist_m = a.get("distance")
    elev_m = a.get("total_elevation_gain")
    mv_s = a.get("moving_time")
    hr = a.get("average_heartrate")

    parts = [str(sport)]
    if isinstance(dist_m, (int, float)):
        parts.append(f"{dist_m/1000.0:.2f} km")
    if isinstance(elev_m, (int, float)):
        parts.append(f"{elev_m:.0f} m+")
    if isinstance(mv_s, (int, float)):
        parts.append(f"{mv_s/60.0:.0f} min")
    if isinstance(hr, (int, float)):
        parts.append(f"avgHR {hr:.0f}")

    name = a.get("name")
    if isinstance(name, str) and name.strip():
        parts.append(f"({name.strip()})")

    return " • " + " | ".join(parts)


def _summarize_day(day: Dict[str, Any]) -> str:
    d = day.get("date", "unknown-date")
    lines = [f"## {d}"]

    sleep = day.get("sleep")
    if isinstance(sleep, dict):
        # keep it lightweight: only include a few small fields if present
        keys = ["sleep_score", "score", "duration", "total_sleep", "resting_hr", "rhr", "readiness", "stress"]
        picked = {k: sleep.get(k) for k in keys if k in sleep}
        if picked:
            lines.append(f"Sleep: {picked}")
        else:
            # fallback: avoid dumping huge dicts
            lines.append("Sleep: (data present)")
    elif sleep is None:
        lines.append("Sleep: (none)")

    acts = day.get("activities") or []
    if isinstance(acts, list) and acts:
        lines.append(f"Activities ({len(acts)}):")
        for a in acts[:50]:  # hard cap per day
            if isinstance(a, dict):
                lines.append(_summarize_activity(a))
    else:
        lines.append("Activities: (none)")

    return "\n".join(lines) + "\n"


def _build_prompt_text(
    prompt_name: str,
    personal: Any,
    rollups: Optional[Any],
    combined: List[Dict[str, Any]],
    *,
    max_chars: int,
    detail_days: int,
) -> str:
    """
    F: Budgeted assembly.
    We build from newest → oldest until we hit max_chars, then stop.
    """
    header = [
        f"# TrailTraining Coach Brief: {prompt_name}",
        "",
        "## Personal profile (raw JSON)",
        str(personal)[:50_000],  # safety cap so a giant profile doesn't consume everything
        "",
    ]

    if rollups is not None:
        header += [
            "## Recent rollups (7d/28d)",
            str(rollups),
            "",
        ]

    # Limit daily detail further even if combined is large
    if detail_days > 0 and len(combined) > detail_days:
        combined_detail = combined[-detail_days:]
        combined_older = combined[:-detail_days]
    else:
        combined_detail = combined
        combined_older = []

    # We include only minimal reference to older days (optional)
    if combined_older:
        header += [
            f"## Older days included in window: {len(combined_older)} (details omitted; rely on rollups + recent detail)",
            "",
        ]

    base = "\n".join(header)
    budget = max_chars if max_chars > 0 else 200_000

    # Start with base; then add day blocks newest→oldest until budget is exhausted
    text_parts: List[str] = [base]
    used = len(base)

    # Add detailed days newest→oldest
    for day in reversed(combined_detail):
        block = _summarize_day(day)
        if used + len(block) > budget:
            break
        text_parts.append(block)
        used += len(block)

    # Add tail instruction
    tail = "\n## Task\n" + _prompt_instruction(prompt_name) + "\n"
    if used + len(tail) <= budget:
        text_parts.append(tail)

    return "\n".join(text_parts)


def _prompt_instruction(prompt_name: str) -> str:
    # Try to use your centralized prompts if present, else fall back.
    try:
        from trailtraining.llm.prompts import PROMPTS  # type: ignore
        if isinstance(PROMPTS, dict) and prompt_name in PROMPTS:
            return str(PROMPTS[prompt_name])
    except Exception:
        pass

    if prompt_name == "training-plan":
        return "Generate a trail-running training plan for the next 7–14 days based on fatigue, recent volume, and sleep."
    if prompt_name == "recovery-status":
        return "Assess recovery status for the last 7 days and give actionable guidance for today and tomorrow."
    if prompt_name == "meal-plan":
        return "Suggest a practical meal plan for the next 3 days aligned with training load and recovery."
    return "Provide coaching guidance based on the provided data."


@dataclass(frozen=True)
class CoachConfig:
    model: str = os.getenv("TRAILTRAINING_LLM_MODEL", "gpt-5.2")
    reasoning_effort: str = os.getenv("TRAILTRAINING_REASONING_EFFORT", "medium")  # none|low|medium|high|xhigh
    verbosity: str = os.getenv("TRAILTRAINING_VERBOSITY", "medium")  # low|medium|high
    days: int = int(os.getenv("TRAILTRAINING_COACH_DAYS", "60"))
    max_chars: int = int(os.getenv("TRAILTRAINING_COACH_MAX_CHARS", "200000"))
    temperature: Optional[float] = None


def run_coach_brief(
    *,
    prompt: str,
    cfg: CoachConfig,
    input_path: Optional[str] = None,
    personal_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    config.ensure_directories()

    personal_p, summary_p, rollups_p = _resolve_input_paths(input_path, personal_path, summary_path)

    personal = load_json(personal_p, default={})
    combined = load_json(summary_p, default=[])
    rollups = load_json(rollups_p, default=None) if rollups_p else None

    if not isinstance(combined, list):
        raise RuntimeError("combined_summary.json must be a list of day objects")

    # F: early pruning
    _dedup_activities_in_place(combined)
    combined = _filter_last_days(combined, cfg.days)

    detail_days = int(os.getenv("TRAILTRAINING_COACH_DETAIL_DAYS", "14"))
    detail_days = max(1, min(detail_days, len(combined))) if combined else 0

    prompt_text = _build_prompt_text(
        prompt_name=prompt,
        personal=personal,
        rollups=rollups,
        combined=combined,
        max_chars=cfg.max_chars,
        detail_days=detail_days,
    )

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("TRAILTRAINING_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI API key. Set OPENAI_API_KEY (recommended) or TRAILTRAINING_OPENAI_API_KEY.\n"
            "Example:\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "Then rerun: trailtraining coach --prompt training-plan"
        )

    client = OpenAI(api_key=api_key)

    # Official pattern: reasoning + text verbosity via Responses API
    kwargs: Dict[str, Any] = {
        "model": cfg.model,
        "input": prompt_text,
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {"verbosity": cfg.verbosity},
    }
    # API restriction: temperature typically only allowed when reasoning.effort == "none"
    if cfg.reasoning_effort == "none" and cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature

    resp = client.responses.create(**kwargs)
    out_text = getattr(resp, "output_text", None) or str(resp)

    # Save output
    if output_path:
        out_p = Path(output_path).expanduser().resolve()
    else:
        out_p = Path(config.PROMPTING_DIRECTORY) / f"coach_brief_{prompt}.md"

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(out_text, encoding="utf-8")

    return out_text, str(out_p)