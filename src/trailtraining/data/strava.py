# src/trailtraining/data/strava.py

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests

# Strava OAuth endpoints
STRAVA_AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"

# Reuse a single session (E: connection pooling)
_SESSION = requests.Session()


def _get_env(name: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        raise RuntimeError(
            f"Missing environment variable: {name}\n"
            f"Set it in your shell or a .env file (but do NOT commit secrets)."
        )
    return value


@dataclass(frozen=True)
class StravaOAuthConfig:
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: str = "read,activity:read_all"

    @staticmethod
    def from_env() -> StravaOAuthConfig:
        return StravaOAuthConfig(
            client_id=_get_env("STRAVA_CLIENT_ID"),
            client_secret=_get_env("STRAVA_CLIENT_SECRET"),
            redirect_uri=_get_env("STRAVA_REDIRECT_URI"),
            scope=os.getenv("STRAVA_SCOPE", "read,activity:read_all"),
        )


def default_token_path() -> Path:
    base_dir = Path(
        os.getenv("TRAILTRAINING_BASE_DIR", str(Path.home() / ".trailtraining"))
    ).expanduser()
    token_dir = base_dir / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    return token_dir / "strava_token.json"


def build_authorize_url(config: StravaOAuthConfig, state: Optional[str] = None) -> tuple[str, str]:
    if state is None:
        state = secrets.token_urlsafe(24)

    params = {
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": config.scope,
        "state": state,
    }
    return f"{STRAVA_AUTHORIZE_URL}?{urlencode(params)}", state


def exchange_code_for_token(
    config: StravaOAuthConfig, code: str, session: requests.Session = _SESSION
) -> dict[str, Any]:
    resp = session.post(
        STRAVA_TOKEN_URL,
        data={
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "code": code,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(
    config: StravaOAuthConfig, refresh_token: str, session: requests.Session = _SESSION
) -> dict[str, Any]:
    resp = session.post(
        STRAVA_TOKEN_URL,
        data={
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def load_token(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    path = path or default_token_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_token(token: dict[str, Any], path: Optional[Path] = None) -> Path:
    path = path or default_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # compact is fine (small file)
    path.write_text(json.dumps(token, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return path


def token_is_valid(token: dict[str, Any], leeway_seconds: int = 60) -> bool:
    expires_at = token.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        return False
    return time.time() < (float(expires_at) - leeway_seconds)


def get_valid_token(
    config: StravaOAuthConfig, token_path: Optional[Path] = None
) -> Optional[dict[str, Any]]:
    token_path = token_path or default_token_path()
    token = load_token(token_path)
    if token is None:
        return None

    if token_is_valid(token):
        return token

    refresh_token_value = token.get("refresh_token")
    if not refresh_token_value:
        return None

    new_token = refresh_access_token(config, refresh_token_value)
    save_token(new_token, token_path)
    return new_token
