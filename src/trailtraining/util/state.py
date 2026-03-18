# src/trailtraining/util/state.py

from __future__ import annotations

import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

from trailtraining.util.errors import ArtifactError

PathLike = Union[str, Path]


def load_json(path: PathLike, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ArtifactError(
            message=f"Invalid JSON in {p}",
            hint=f"Parse error at line {err.lineno}, column {err.colno}: {err.msg}",
        ) from err
    except OSError as err:
        raise ArtifactError(
            message=f"Could not read {p}",
            hint=str(err),
        ) from err


def atomic_write_text(path: PathLike, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _json_default(o: object) -> str:
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def save_json(
    path: PathLike,
    obj: Any,
    *,
    compact: bool = True,
    ensure_ascii: bool = False,
) -> None:
    if compact:
        text = json.dumps(
            obj, ensure_ascii=ensure_ascii, separators=(",", ":"), default=_json_default
        )
    else:
        text = json.dumps(
            obj, ensure_ascii=ensure_ascii, indent=2, sort_keys=True, default=_json_default
        )
    atomic_write_text(path, text)
