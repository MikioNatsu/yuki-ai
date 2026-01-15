from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/voice", tags=["voice"])


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_style: str | None = None


@router.post("/asr")
async def asr(audio: UploadFile = File(...)):
    # TODO: integrate faster-whisper / whisper.cpp
    return {"text": "[TODO] transcribed text"}


@router.post("/tts")
async def tts(req: TTSRequest):
    # TODO: integrate Piper / Coqui XTTS and return audio bytes
    return {"status": "ok", "note": "TODO: generate audio", "voice_style": req.voice_style}
