from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator


EmotionLabel = Literal[
    "playful",
    "comforting",
    "excited",
    "curious",
    "sad",
    "flirty",
    "calm",
    "surprised",
    "empathetic",
]


class EmotionOut(BaseModel):
    label: EmotionLabel
    valence: float = Field(ge=0.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)


class MetaOut(BaseModel):
    model: str
    latency_ms: int = Field(ge=0)
    tokens_estimate: int = Field(ge=0)


class ChatResponse(BaseModel):
    session_id: str
    display_text: str
    speech_text: str
    emotion: EmotionOut
    meta: MetaOut


class HistoryMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("history message content is empty")
        return v2


class ChatRequest(BaseModel):
    """Incoming request to /chat and /chat/stream.

    Backward-compatible: accepts user_text OR message OR prompt OR text.
    """

    session_id: Optional[str] = Field(default=None)
    user_text: str = Field(
        ...,
        validation_alias=AliasChoices("user_text", "message", "prompt", "text"),
    )
    history: Optional[List[HistoryMessage]] = None

    @field_validator("session_id")
    @classmethod
    def _session_strip(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v2 = v.strip()
        return v2 or None

    @field_validator("user_text")
    @classmethod
    def _user_text_strip(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("user_text is empty")
        return v2


class ErrorResponse(BaseModel):
    detail: str
    code: str
