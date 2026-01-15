from .routes_chat import router as chat_router
from .routes_voice import router as voice_router


def register_routes(app):
    """Backward-compatible helper."""
    app.include_router(chat_router)
    app.include_router(voice_router)
