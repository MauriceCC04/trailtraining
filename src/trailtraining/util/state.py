# src/trailtraining/util/state.py

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_write_text(path: str | Path, text: str) -> None:
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


def save_json(
    path: str | Path,
    obj: Any,
    *,
    compact: bool = True,
    ensure_ascii: bool = False,
) -> None:
    if compact:
        text = json.dumps(obj, ensure_ascii=ensure_ascii, separators=(",", ":"))
    else:
        text = json.dumps(obj, ensure_ascii=ensure_ascii, indent=2, sort_keys=True)
    atomic_write_text(path, text)