from __future__ import annotations

from flask import Flask, request
import threading
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_app = Flask(__name__)
_auth_code: dict[str, str] = {}


@_app.get("/authorization")
def authorization():
    code = request.args.get("code")
    if code:
        _auth_code["code"] = code
        return "Authorization successful! You can close this window."
    return "No authorization code found.", 400


def start_auth_server(host: str = "127.0.0.1", port: int = 5000) -> threading.Thread:
    """
    Starts Flask server in a daemon thread.
    """
    _auth_code.clear()
    t = threading.Thread(
        target=_app.run,
        kwargs={"host": host, "port": port, "use_reloader": False},
        daemon=True,
    )
    t.start()
    logger.info("Auth server started on http://%s:%s/authorization", host, port)
    return t


def wait_for_code(timeout: Optional[float] = 300.0, poll: float = 0.5) -> str:
    """
    Wait for an auth code to be set by the /authorization route.
    """
    deadline = time.time() + timeout if timeout else None
    while "code" not in _auth_code:
        if deadline and time.time() > deadline:
            raise TimeoutError("Timed out waiting for Strava authorization code.")
        time.sleep(poll)
    return _auth_code["code"]