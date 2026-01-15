from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppError(Exception):
    """A deterministic application error surfaced to clients as JSON."""

    detail: str
    code: str = "error"
    status_code: int = 400

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.code}: {self.detail}"
