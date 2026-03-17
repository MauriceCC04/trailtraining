from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrailTrainingError(Exception):
    message: str
    hint: str | None = None
    exit_code: int = 1

    def __str__(self) -> str:
        return self.message


class ConfigError(TrailTrainingError):
    pass


class DataValidationError(TrailTrainingError):
    pass


class ExternalServiceError(TrailTrainingError):
    pass


class ArtifactError(TrailTrainingError):
    pass


class LLMUnsupportedParameterError(TrailTrainingError):
    """Raised when an LLM provider rejects an unsupported parameter.

    Only this error type should trigger parameter-stripping fallback.
    Auth, network, rate-limit, and server errors must NOT be retried
    via parameter stripping.
    """

    pass


class MissingArtifactError(TrailTrainingError):
    """Raised when a required pipeline artifact is missing or empty."""

    pass
