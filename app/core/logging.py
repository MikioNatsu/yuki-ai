from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Optional


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure these always exist so our formatter never crashes.
        record.request_id = request_id_var.get("-")
        record.session_id = session_id_var.get("-")
        return True


def set_log_context(*, request_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    if request_id is not None:
        request_id_var.set(request_id)
    if session_id is not None:
        session_id_var.set(session_id)


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(lvl)

    # Replace handlers to avoid duplicated logs under reload.
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())

    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        " | request_id=%(request_id)s session_id=%(session_id)s"
    )
    handler.setFormatter(logging.Formatter(fmt))

    root.handlers = [handler]

    # Reduce noisy loggers if desired; keep uvicorn errors visible.
    logging.getLogger("httpx").setLevel(max(lvl, logging.WARNING))
    logging.getLogger("uvicorn.access").setLevel(max(lvl, logging.INFO))


def get_logger(name: str = "yuki") -> logging.Logger:
    return logging.getLogger(name)
