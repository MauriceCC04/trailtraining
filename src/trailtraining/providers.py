from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trailtraining import config

ProviderName = Literal["garmin", "intervals"]
RequestedProvider = Literal["auto", "garmin", "intervals"]
ResolutionSource = Literal["explicit", "env", "auto"]


@dataclass(frozen=True)
class ProviderResolution:
    provider: ProviderName
    requested: RequestedProvider
    source: ResolutionSource
    intervals_ready: bool
    garmin_ready: bool


def _normalize_provider(value: str | None) -> RequestedProvider:
    v = (value or "").strip().lower()
    if v == "garmin":
        return "garmin"
    if v == "intervals":
        return "intervals"
    return "auto"


def intervals_ready() -> bool:
    return bool(config.intervals_api_key())


def garmin_ready() -> bool:
    return bool(config.garmin_email() and config.garmin_password())


def resolve_wellness_provider(explicit: str | None = None) -> ProviderResolution:
    if explicit and explicit.strip():
        requested = _normalize_provider(explicit)
        source: ResolutionSource = "explicit"
    else:
        requested = _normalize_provider(config.wellness_provider_setting())
        source = "env" if requested != "auto" else "auto"

    i_ready = intervals_ready()
    g_ready = garmin_ready()

    if requested == "intervals":
        provider: ProviderName = "intervals"
    elif requested == "garmin":
        provider = "garmin"
    elif i_ready:
        provider = "intervals"
    elif g_ready:
        provider = "garmin"
    else:
        provider = "intervals"

    return ProviderResolution(
        provider=provider,
        requested=requested,
        source=source,
        intervals_ready=i_ready,
        garmin_ready=g_ready,
    )
