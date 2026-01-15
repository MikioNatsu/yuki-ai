from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes_chat import router as chat_router
from app.api.routes_voice import router as voice_router
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger, set_log_context, setup_logging
from app.core.rate_limit import InMemoryRateLimiter
from app.llm.ollama_client import OllamaClient
from app.memory.store import make_store

log = get_logger("yuki.main")


def _error_json(detail: str, code: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail, "code": code})


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    app.state.settings = settings
    app.state.rate_limiter = InMemoryRateLimiter(per_minute=settings.RATE_LIMIT_PER_MINUTE)
    app.state.store = make_store(settings.STORAGE, settings.SQLITE_PATH)
    app.state.ollama = OllamaClient(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        timeout_s=settings.TIMEOUT,
        api_mode=settings.OLLAMA_API_MODE,
        retry_max_attempts=settings.RETRY_MAX_ATTEMPTS,
        retry_backoff_base=settings.RETRY_BACKOFF_BASE,
    )

    log.info(
        "startup ok | model=%s api_mode=%s storage=%s streaming=%s",
        settings.OLLAMA_MODEL,
        settings.OLLAMA_API_MODE,
        settings.STORAGE,
        settings.ENABLE_STREAMING,
    )
    try:
        yield
    finally:
        try:
            await app.state.ollama.aclose()
        except Exception:
            pass
        log.info("shutdown ok")


app = FastAPI(title="Yuki AI", version="2.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(voice_router)


@app.middleware("http")
async def request_context_and_rate_limit(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    set_log_context(request_id=rid, session_id="-")

    if request.url.path.startswith("/chat") and request.method.upper() in ("POST",):
        limiter = request.app.state.rate_limiter
        ip = request.client.host if request.client else "unknown"
        res = limiter.check(ip)
        if not res.allowed:
            resp = _error_json("Rate limit exceeded", "rate_limited", 429)
            resp.headers["Retry-After"] = str(int(res.retry_after_s) + 1)
            resp.headers["X-Request-ID"] = rid
            return resp

    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        log.debug("request %s %s done in %sms", request.method, request.url.path, dt_ms)

    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return _error_json(exc.detail, exc.code, exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return _error_json("Invalid request", "validation_error", 422)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    return _error_json(detail, "http_error", exc.status_code)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    log.exception("unhandled error: %s", exc)
    return _error_json("Internal server error", "internal_error", 500)


@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    llm = request.app.state.ollama
    ollama = await llm.health()
    return {
        "status": "ok",
        "service": "yuki-ai",
        "ollama": {
            "base_url": request.app.state.settings.OLLAMA_BASE_URL,
            "model": request.app.state.settings.OLLAMA_MODEL,
            "api_mode": request.app.state.settings.OLLAMA_API_MODE,
            **ollama,
        },
    }
