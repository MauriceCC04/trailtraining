import sys
from types import ModuleType, SimpleNamespace

import trailtraining
import trailtraining.data as data_pkg
import trailtraining.forecast as forecast_pkg
import trailtraining.pipelines as pipelines_pkg
from trailtraining.commands import pipeline_commands as pc


def _install_module(monkeypatch, package, full_name: str, attr_name: str, **attrs):
    module = ModuleType(full_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, full_name, module)
    monkeypatch.setattr(package, attr_name, module, raising=False)
    return module


def test_pipeline_commands_delegate_to_underlying_modules(monkeypatch):
    calls = []

    monkeypatch.setattr(pc, "_run", lambda func: func())

    _install_module(
        monkeypatch,
        pipelines_pkg,
        "trailtraining.pipelines.strava",
        "strava",
        auth_main=lambda force=False: calls.append(("auth_strava", force)),
        main=lambda: calls.append(("fetch_strava", None)),
    )
    _install_module(
        monkeypatch,
        pipelines_pkg,
        "trailtraining.pipelines.garmin",
        "garmin",
        main=lambda: calls.append(("fetch_garmin", None)),
    )
    _install_module(
        monkeypatch,
        data_pkg,
        "trailtraining.data.combine",
        "combine",
        main=lambda: calls.append(("combine", None)),
    )
    _install_module(
        monkeypatch,
        pipelines_pkg,
        "trailtraining.pipelines.run_all",
        "run_all",
        main=lambda **kwargs: calls.append(("run_all", kwargs)),
    )
    _install_module(
        monkeypatch,
        pipelines_pkg,
        "trailtraining.pipelines.intervals",
        "intervals",
        main=lambda **kwargs: calls.append(("fetch_intervals", kwargs)),
    )
    _install_module(
        monkeypatch,
        pipelines_pkg,
        "trailtraining.pipelines.run_all_intervals",
        "run_all_intervals",
        main=lambda **kwargs: calls.append(("run_all_intervals", kwargs)),
    )
    _install_module(
        monkeypatch,
        trailtraining,
        "trailtraining.doctor",
        "doctor",
        main=lambda: calls.append(("doctor", None)),
    )

    pc.cmd_auth_strava(SimpleNamespace(force=True))
    pc.cmd_fetch_strava(SimpleNamespace())
    pc.cmd_fetch_garmin(SimpleNamespace())
    pc.cmd_combine(SimpleNamespace())
    pc.cmd_run_all(
        SimpleNamespace(
            clean=True,
            clean_processing=False,
            clean_prompting=True,
            wellness_provider="intervals",
        )
    )
    pc.cmd_fetch_intervals(SimpleNamespace(oldest="2026-01-01", newest="2026-01-31"))
    pc.cmd_run_all_intervals(
        SimpleNamespace(clean=False, clean_processing=True, clean_prompting=False)
    )
    pc.cmd_doctor(SimpleNamespace())

    assert calls == [
        ("auth_strava", True),
        ("fetch_strava", None),
        ("fetch_garmin", None),
        ("combine", None),
        (
            "run_all",
            {
                "clean": True,
                "clean_processing": False,
                "clean_prompting": True,
                "wellness_provider": "intervals",
            },
        ),
        ("fetch_intervals", {"oldest": "2026-01-01", "newest": "2026-01-31"}),
        (
            "run_all_intervals",
            {
                "clean": False,
                "clean_processing": True,
                "clean_prompting": False,
            },
        ),
        ("doctor", None),
    ]


def test_forecast_command_prints_saved_path_and_result(monkeypatch, capsys):
    monkeypatch.setattr(pc, "_run", lambda func: func())
    _install_module(
        monkeypatch,
        forecast_pkg,
        "trailtraining.forecast.forecast",
        "forecast",
        run_forecasts=lambda input_dir, output_path: {
            "saved": output_path,
            "result": f"forecast from {input_dir}",
        },
    )

    pc.cmd_forecast(SimpleNamespace(input="in-dir", output="out.json"))

    out = capsys.readouterr().out
    assert "[Saved] out.json" in out
    assert "forecast from in-dir" in out
