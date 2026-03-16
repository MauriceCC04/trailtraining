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
