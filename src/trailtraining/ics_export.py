# src/trailtraining/ics_export.py
"""Generate .ics calendar files from TrainingPlanArtifact."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trailtraining.contracts import TrainingPlanArtifact
from trailtraining.util.state import atomic_write_text


def _ics_escape(text: str) -> str:
    """Escape special characters per RFC 5545."""
    text = text.replace("\\", "\\\\")
    text = text.replace(";", "\\;")
    text = text.replace(",", "\\,")
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "")
    return text


def _fold_line(line: str) -> str:
    """Fold long lines at 75 octets per RFC 5545."""
    encoded = line.encode("utf-8")
    if len(encoded) <= 75:
        return line
    parts: list[str] = []
    # Fold on character boundaries; simple ASCII-safe approach
    while len(line.encode("utf-8")) > 75:
        # Find safe split point
        chunk = line[:74]
        while len(chunk.encode("utf-8")) > 74:
            chunk = chunk[:-1]
        parts.append(chunk)
        line = line[len(chunk) :]
    parts.append(line)
    return "\r\n ".join(parts)


def _prop(name: str, value: str) -> str:
    return _fold_line(f"{name}:{value}") + "\r\n"


def _session_type_to_category(session_type: str) -> str:
    mapping = {
        "rest": "REST",
        "easy": "EASY RUN",
        "aerobic": "AEROBIC",
        "long": "LONG RUN",
        "tempo": "TEMPO",
        "intervals": "INTERVALS",
        "hills": "HILLS",
        "strength": "STRENGTH",
        "cross": "CROSS TRAINING",
    }
    return mapping.get(session_type, session_type.upper())


def plan_to_ics(
    artifact: TrainingPlanArtifact,
    *,
    start_hour: int = 7,
    calendar_name: str = "Training Plan",
    timezone_id: str | None = None,
) -> str:
    """Return ICS text for a TrainingPlanArtifact.

    Non-rest sessions become timed events starting at ``start_hour``.
    Rest days become all-day events.

    Timezone behavior:
    - If ``timezone_id`` is provided (e.g. ``"Europe/Rome"``), a VTIMEZONE
      reference is emitted and DTSTART/DTEND use TZID.
    - If ``timezone_id`` is None (default), events use floating local times
      (no TZID, no ``Z`` suffix).  RFC 5545 §3.3.5 says these are
      interpreted in the viewer's local timezone, which is the correct
      default for a training plan generated without knowing the user's tz.
    """
    dtstamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines: list[str] = []

    lines.append("BEGIN:VCALENDAR\r\n")
    lines.append("VERSION:2.0\r\n")
    lines.append("PRODID:-//TrailTraining//Training Plan//EN\r\n")
    lines.append("CALSCALE:GREGORIAN\r\n")
    lines.append("METHOD:PUBLISH\r\n")
    lines.append(_prop("X-WR-CALNAME", _ics_escape(calendar_name)))

    if timezone_id:
        lines.append(_prop("X-WR-TIMEZONE", timezone_id))

    for day in artifact.plan.days:
        # Parse date — day.date is now a proper date object from contracts
        day_date = day.date
        year, month, day_num = day_date.year, day_date.month, day_date.day

        event_uid = f"trailtraining-{day_date.isoformat()}-{uuid.uuid4().hex[:8]}@trailtraining"

        # Build description block
        desc_parts = []
        if day.workout:
            desc_parts.append(day.workout)
        if day.purpose:
            desc_parts.append(f"Purpose: {day.purpose}")
        if day.target_intensity and day.target_intensity not in ("N/A", "Off", ""):
            desc_parts.append(f"Intensity: {day.target_intensity}")
        if day.terrain and day.terrain not in ("N/A", ""):
            desc_parts.append(f"Terrain: {day.terrain}")
        description = _ics_escape("\\n\\n".join(desc_parts))

        summary = _ics_escape(day.title)
        category = _session_type_to_category(day.session_type)

        lines.append("BEGIN:VEVENT\r\n")
        lines.append(_prop("UID", event_uid))
        lines.append(_prop("DTSTAMP", dtstamp))
        lines.append(_prop("SUMMARY", summary))
        lines.append(_prop("DESCRIPTION", description))
        lines.append(_prop("CATEGORIES", category))

        if day.is_rest_day or day.duration_minutes == 0:
            # All-day event (always floating — DATE values have no tz)
            dtstart = f"{year:04d}{month:02d}{day_num:02d}"
            next_day = datetime(year, month, day_num) + timedelta(days=1)
            dtend = next_day.strftime("%Y%m%d")
            lines.append(_prop("DTSTART;VALUE=DATE", dtstart))
            lines.append(_prop("DTEND;VALUE=DATE", dtend))
        else:
            # Timed event at start_hour
            start_dt = datetime(year, month, day_num, start_hour, 0, 0)
            end_dt = start_dt + timedelta(minutes=day.duration_minutes)
            fmt = "%Y%m%dT%H%M%S"

            if timezone_id:
                lines.append(_prop(f"DTSTART;TZID={timezone_id}", start_dt.strftime(fmt)))
                lines.append(_prop(f"DTEND;TZID={timezone_id}", end_dt.strftime(fmt)))
            else:
                # Floating local time (no TZID, no Z suffix)
                lines.append(_prop("DTSTART", start_dt.strftime(fmt)))
                lines.append(_prop("DTEND", end_dt.strftime(fmt)))

            lines.append(_prop("DURATION", f"PT{day.duration_minutes}M"))

        lines.append("END:VEVENT\r\n")

    lines.append("END:VCALENDAR\r\n")
    return "".join(lines)


def find_latest_plan(prompting_dir: str) -> Path:
    """Return the most recently modified training plan JSON in *prompting_dir*.

    Checks ``revised-plan.json`` and ``coach_brief_training-plan.json``; picks
    whichever has the newer mtime.  Raises ``FileNotFoundError`` if neither
    exists.
    """
    candidates = [
        Path(prompting_dir) / "revised-plan.json",
        Path(prompting_dir) / "coach_brief_training-plan.json",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        raise FileNotFoundError(
            f"No training plan found in {prompting_dir}. "
            "Run `trailtraining coach` or `trailtraining revise-plan` first."
        )
    return max(found, key=lambda p: p.stat().st_mtime)


def export_plan_to_ics(
    prompting_dir: str,
    *,
    output_path: str | None = None,
    start_hour: int = 7,
    calendar_name: str = "Training Plan",
    timezone_id: str | None = None,
) -> tuple[Path, Path]:
    """Load the most recent plan, write an ICS file, return (plan_path, ics_path)."""
    import json

    plan_path = find_latest_plan(prompting_dir)
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    artifact = TrainingPlanArtifact.model_validate(raw)

    ics_text = plan_to_ics(
        artifact,
        start_hour=start_hour,
        calendar_name=calendar_name,
        timezone_id=timezone_id,
    )

    if output_path:
        ics_path = Path(output_path).expanduser().resolve()
    else:
        ics_path = Path(prompting_dir) / "training-plan.ics"

    atomic_write_text(ics_path, ics_text)
    return plan_path, ics_path


# Allow running as a standalone script for quick testing
if __name__ == "__main__":  # pragma: no cover
    import sys

    from trailtraining import config

    _, ics_path = export_plan_to_ics(config.prompting_directory())
    print(f"[Saved] {ics_path}")
    if "--open" in sys.argv:
        import subprocess

        subprocess.run(["open", str(ics_path)], check=False)
