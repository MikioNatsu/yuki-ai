from __future__ import annotations

from functools import lru_cache
from typing import Literal, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Notes:
      - We intentionally support comma-separated CORS_ORIGINS ("a,b,c") and "*".
      - Extra env vars are ignored to keep upgrades painless.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Ollama ----
    OLLAMA_BASE_URL: str = Field(default="http://127.0.0.1:11434")
    OLLAMA_MODEL: str = Field(default="yuki:latest")
    # default = generate (per requirement). Set to "auto" or "chat" if you want /api/chat.
    OLLAMA_API_MODE: Literal["generate", "chat", "auto"] = Field(default="generate")

    # ---- Runtime ----
    TIMEOUT: float = Field(default=120.0, description="HTTP read timeout (seconds)")
    MAX_HISTORY: int = Field(default=20, description="Max stored messages per session")
    MAX_INPUT_CHARS: int = Field(default=2000, description="Max characters in user input")
    LOG_LEVEL: str = Field(default="INFO")
    ENABLE_STREAMING: bool = Field(default=True)

    # ---- Storage ----
    STORAGE: Literal["memory", "sqlite"] = Field(default="memory")
    SQLITE_PATH: str = Field(default="./yuki.sqlite3")

    # ---- Security / UX ----
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Per-IP requests per minute")

    # ---- Retry policy ----
    RETRY_MAX_ATTEMPTS: int = Field(default=3, description="Total attempts (1 = no retry)")
    RETRY_BACKOFF_BASE: float = Field(default=0.5, description="Base backoff seconds")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        if v is None or v == "":
            return ["*"]
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            if s == "*":
                return ["*"]
            return [o.strip() for o in s.split(",") if o.strip()]
        return ["*"]

    @field_validator("TIMEOUT")
    @classmethod
    def _timeout_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("TIMEOUT must be > 0")
        return float(v)

    @field_validator("MAX_HISTORY", "MAX_INPUT_CHARS", "RATE_LIMIT_PER_MINUTE", "RETRY_MAX_ATTEMPTS")
    @classmethod
    def _ints_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be > 0")
        return int(v)

    @field_validator("RETRY_BACKOFF_BASE")
    @classmethod
    def _backoff_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("RETRY_BACKOFF_BASE must be >= 0")
        return float(v)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
