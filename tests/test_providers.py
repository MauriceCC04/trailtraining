# tests/test_providers.py

import pytest
from trailtraining.providers import resolve_wellness_provider


def clear_env(monkeypatch):
    for name in [
        "TRAILTRAINING_WELLNESS_PROVIDER",
        "WELLNESS_PROVIDER",
        "INTERVALS_API_KEY",
        "GARMIN_EMAIL",
        "GARMIN_PASSWORD",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_explicit_provider_wins(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("INTERVALS_API_KEY", "abc")
    res = resolve_wellness_provider("garmin")
    assert res.provider == "garmin"
    assert res.requested == "garmin"
    assert res.source == "explicit"


def test_env_provider_wins(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("TRAILTRAINING_WELLNESS_PROVIDER", "garmin")
    res = resolve_wellness_provider()
    assert res.provider == "garmin"
    assert res.requested == "garmin"
    assert res.source == "env"


def test_auto_prefers_intervals_when_configured(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("INTERVALS_API_KEY", "abc")
    res = resolve_wellness_provider()
    assert res.provider == "intervals"
    assert res.requested == "auto"


def test_auto_uses_garmin_when_intervals_missing(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("GARMIN_EMAIL", "user@example.com")
    monkeypatch.setenv("GARMIN_PASSWORD", "secret")
    res = resolve_wellness_provider()
    assert res.provider == "garmin"
    assert res.requested == "auto"


def test_auto_raises_when_nothing_configured(monkeypatch):
    from trailtraining.util.errors import ConfigError

    clear_env(monkeypatch)
    with pytest.raises(ConfigError, match="No wellness provider is configured"):
        resolve_wellness_provider()
