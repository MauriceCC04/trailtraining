import logging
from pathlib import Path

import pytest
from trailtraining.commands import common


def test_run_passes_through_success_and_system_exit():
    common._run(lambda: None)

    with pytest.raises(SystemExit) as exc:
        common._run(lambda: (_ for _ in ()).throw(SystemExit(3)))
    assert exc.value.code == 3


def test_run_logs_and_exits_on_unhandled_exception(caplog):
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        common._run(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert exc.value.code == 1
    assert "Unhandled error" in caplog.text


def test_run_formats_trailtraining_errors(capfd):
    from trailtraining.util.errors import ConfigError

    with pytest.raises(SystemExit) as exc:
        common._run(
            lambda: (_ for _ in ()).throw(
                ConfigError(message="bad config", hint="set ENV", exit_code=2)
            )
        )

    assert exc.value.code == 2
    err = capfd.readouterr().err
    assert "Error: bad config" in err
    assert "Hint: set ENV" in err


def test_env_truthy_and_load_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("TT_FLAG", raising=False)
    assert common._env_truthy("TT_FLAG", default=True) is True

    monkeypatch.setenv("TT_FLAG", "off")
    assert common._env_truthy("TT_FLAG", default=True) is False

    env_path = tmp_path / "profile.env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "FOO=bar",
                'QUOTED="value"',
                "SINGLE='value2'",
                "EMPTY_KEY=",
                "EXISTING=from-file",
                "not-an-env-line",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING", "keep-me")

    common._load_env_file(env_path)

    assert common.os.environ["FOO"] == "bar"
    assert common.os.environ["QUOTED"] == "value"
    assert common.os.environ["SINGLE"] == "value2"
    assert common.os.environ["EMPTY_KEY"] == ""
    assert common.os.environ["EXISTING"] == "keep-me"


def test_apply_profile_sets_profile_and_default_base_dir(monkeypatch, tmp_path):
    loaded = {}

    def fake_load_env_file(path: Path) -> None:
        loaded["path"] = path

    monkeypatch.setattr(common, "_load_env_file", fake_load_env_file)
    monkeypatch.setattr(common.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("TRAILTRAINING_BASE_DIR", raising=False)

    profile = common.apply_profile("  athlete  ")

    assert profile == "athlete"
    assert common.os.environ["TRAILTRAINING_PROFILE"] == "athlete"
    assert loaded["path"] == tmp_path / ".trailtraining" / "profiles" / "athlete.env"
    assert common.os.environ["TRAILTRAINING_BASE_DIR"] == str(
        tmp_path / "trailtraining-data" / "athlete"
    )
