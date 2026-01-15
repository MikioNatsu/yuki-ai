from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.core.errors import AppError
from app.core.logging import get_logger, set_log_context
from app.memory.store import ChatMessage, Store
from app.schemas import ChatRequest, ChatResponse, EmotionOut, MetaOut
from app.yuki.prompting import build_chat_messages, build_generate_prompt, build_system_prompt
from app.yuki.speech_director import (
    direct_speech,
    infer_emotion,
    sanitize_display_text,
    update_session_state,
)

router = APIRouter(tags=["chat"])
log = get_logger("yuki.routes.chat")


def _ensure_session_id(session_id: Optional[str]) -> str:
    if session_id and session_id.strip():
        sid = session_id.strip()
        if len(sid) > 128:
            raise AppError(detail="session_id too long", code="invalid_session_id", status_code=400)
        return sid
    return str(uuid.uuid4())


def _require_text_limits(user_text: str, max_chars: int) -> None:
    if len(user_text) > max_chars:
        raise AppError(
            detail=f"Input too long. max={max_chars} chars",
            code="input_too_long",
            status_code=413,
        )


def _sse(event: str, data: Any) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    settings = request.app.state.settings
    store: Store = request.app.state.store
    llm = request.app.state.ollama

    session_id = _ensure_session_id(req.session_id)
    set_log_context(session_id=session_id)

    _require_text_limits(req.user_text, settings.MAX_INPUT_CHARS)

    t0 = time.perf_counter()

    emotion: EmotionOut = infer_emotion(req.user_text)

    state = store.get_state(session_id)
    if req.history is not None:
        history = [ChatMessage(role=m.role, content=m.content, ts=time.time()) for m in req.history]
        history = history[-settings.MAX_HISTORY:]
    else:
        history = store.get_history(session_id, limit=settings.MAX_HISTORY)

    system_prompt = build_system_prompt(state, target_emotion=emotion.label)
    messages = build_chat_messages(system_prompt=system_prompt, history=history, user_text=req.user_text)
    prompt = build_generate_prompt(history=history, user_text=req.user_text)

    result = await llm.complete(system=system_prompt, prompt=prompt, messages=messages, stream=False)

    display_text = sanitize_display_text(result.text)
    speech_text = direct_speech(display_text, emotion)

    new_state = update_session_state(state, emotion)
    store.set_state(session_id, new_state)
    store.append_message(session_id, "user", req.user_text, max_history=settings.MAX_HISTORY)
    store.append_message(session_id, "assistant", display_text, max_history=settings.MAX_HISTORY)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    meta = MetaOut(model=result.model, latency_ms=latency_ms, tokens_estimate=result.tokens_estimate)

    log.info(
        "chat ok endpoint=%s ollama_latency_ms=%s total_latency_ms=%s tokens=%s",
        result.endpoint,
        result.latency_ms,
        latency_ms,
        result.tokens_estimate,
    )

    return ChatResponse(
        session_id=session_id,
        display_text=display_text,
        speech_text=speech_text,
        emotion=emotion,
        meta=meta,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    settings = request.app.state.settings
    if not settings.ENABLE_STREAMING:
        raise AppError(detail="Streaming disabled", code="streaming_disabled", status_code=404)

    store: Store = request.app.state.store
    llm = request.app.state.ollama

    session_id = _ensure_session_id(req.session_id)
    set_log_context(session_id=session_id)

    _require_text_limits(req.user_text, settings.MAX_INPUT_CHARS)

    emotion: EmotionOut = infer_emotion(req.user_text)

    state = store.get_state(session_id)
    if req.history is not None:
        history = [ChatMessage(role=m.role, content=m.content, ts=time.time()) for m in req.history]
        history = history[-settings.MAX_HISTORY:]
    else:
        history = store.get_history(session_id, limit=settings.MAX_HISTORY)

    system_prompt = build_system_prompt(state, target_emotion=emotion.label)
    messages = build_chat_messages(system_prompt=system_prompt, history=history, user_text=req.user_text)
    prompt = build_generate_prompt(history=history, user_text=req.user_text)

    async def gen() -> AsyncGenerator[bytes, None]:
        t0 = time.perf_counter()
        full = ""
        try:
            yield _sse("start", {"type": "start", "session_id": session_id, "emotion": emotion.model_dump()})

            async for delta in llm.stream(system=system_prompt, prompt=prompt, messages=messages):
                if await request.is_disconnected():
                    break
                full += delta
                yield _sse("delta", {"type": "delta", "delta": delta})

            display_text = sanitize_display_text(full)
            speech_text = direct_speech(display_text, emotion)

            new_state = update_session_state(state, emotion)
            store.set_state(session_id, new_state)
            store.append_message(session_id, "user", req.user_text, max_history=settings.MAX_HISTORY)
            store.append_message(session_id, "assistant", display_text, max_history=settings.MAX_HISTORY)

            latency_ms = int((time.perf_counter() - t0) * 1000)
            meta = MetaOut(
                model=getattr(llm, "model", "unknown"),
                latency_ms=latency_ms,
                tokens_estimate=max(1, int((len(system_prompt) + len(prompt) + len(full)) / 4)),
            )

            final = ChatResponse(
                session_id=session_id,
                display_text=display_text,
                speech_text=speech_text,
                emotion=emotion,
                meta=meta,
            )
            yield _sse("final", {"type": "final", "final": final.model_dump()})
        except AppError as e:
            yield _sse("error", {"type": "error", "error": {"detail": e.detail, "code": e.code}})
        except Exception as e:
            yield _sse("error", {"type": "error", "error": {"detail": str(e), "code": "internal_error"}})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """Optional WS endpoint (kept for backward-compat / debugging)."""
    await ws.accept()
    app = ws.app
    settings = app.state.settings
    store: Store = app.state.store
    llm = app.state.ollama

    try:
        while True:
            data = await ws.receive_json()
            req = ChatRequest.model_validate(data)
            session_id = _ensure_session_id(req.session_id)
            set_log_context(session_id=session_id)

            _require_text_limits(req.user_text, settings.MAX_INPUT_CHARS)
            emotion = infer_emotion(req.user_text)

            state = store.get_state(session_id)
            if req.history is not None:
                history = [ChatMessage(role=m.role, content=m.content, ts=time.time()) for m in req.history]
                history = history[-settings.MAX_HISTORY:]
            else:
                history = store.get_history(session_id, limit=settings.MAX_HISTORY)

            system_prompt = build_system_prompt(state, target_emotion=emotion.label)
            messages = build_chat_messages(system_prompt=system_prompt, history=history, user_text=req.user_text)
            prompt = build_generate_prompt(history=history, user_text=req.user_text)

            full = ""
            await ws.send_json({"type": "start", "session_id": session_id, "emotion": emotion.model_dump()})

            async for delta in llm.stream(system=system_prompt, prompt=prompt, messages=messages):
                full += delta
                await ws.send_json({"type": "delta", "delta": delta})

            display_text = sanitize_display_text(full)
            speech_text = direct_speech(display_text, emotion)

            new_state = update_session_state(state, emotion)
            store.set_state(session_id, new_state)
            store.append_message(session_id, "user", req.user_text, max_history=settings.MAX_HISTORY)
            store.append_message(session_id, "assistant", display_text, max_history=settings.MAX_HISTORY)

            final = ChatResponse(
                session_id=session_id,
                display_text=display_text,
                speech_text=speech_text,
                emotion=emotion,
                meta=MetaOut(
                    model=getattr(llm, "model", "unknown"),
                    latency_ms=0,
                    tokens_estimate=max(1, int((len(system_prompt) + len(prompt) + len(full)) / 4)),
                ),
            )
            await ws.send_json({"type": "final", "final": final.model_dump()})

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        finally:
            await ws.close()
