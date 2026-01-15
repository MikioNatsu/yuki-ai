from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from threading import Lock
from typing import List, Literal, Protocol


Role = Literal["system", "user", "assistant"]


@dataclass
class SessionState:
    mood: str = "calm"  # coarse vibe
    trust: int = 50      # 0-100
    energy: int = 60     # 0-100
    last_emotion: str = "calm"


@dataclass
class ChatMessage:
    role: Role
    content: str
    ts: float


class Store(Protocol):
    def get_state(self, session_id: str) -> SessionState: ...
    def set_state(self, session_id: str, state: SessionState) -> None: ...
    def get_history(self, session_id: str, limit: int) -> List[ChatMessage]: ...
    def append_message(self, session_id: str, role: Role, content: str, max_history: int) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self._state = {}   # session_id -> SessionState
        self._history = {} # session_id -> List[ChatMessage]
        self._lock = Lock()

    def get_state(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._state:
                self._state[session_id] = SessionState()
            return self._state[session_id]

    def set_state(self, session_id: str, state: SessionState) -> None:
        with self._lock:
            self._state[session_id] = state

    def get_history(self, session_id: str, limit: int) -> List[ChatMessage]:
        with self._lock:
            hist = self._history.get(session_id, [])
            return hist[-limit:] if limit > 0 else []

    def append_message(self, session_id: str, role: Role, content: str, max_history: int) -> None:
        now = time.time()
        msg = ChatMessage(role=role, content=content, ts=now)
        with self._lock:
            self._history.setdefault(session_id, []).append(msg)
            if max_history > 0:
                self._history[session_id] = self._history[session_id][-max_history:]


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);")
            conn.commit()

    def _ensure_session(self, conn: sqlite3.Connection, session_id: str) -> None:
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row:
            return
        now = time.time()
        state = SessionState()
        conn.execute(
            "INSERT INTO sessions(session_id, state_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, json.dumps(asdict(state), ensure_ascii=False), now, now),
        )
        conn.commit()

    def get_state(self, session_id: str) -> SessionState:
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            row = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row:
                return SessionState()
            try:
                data = json.loads(row["state_json"])
                return SessionState(**data)
            except Exception:
                return SessionState()

    def set_state(self, session_id: str, state: SessionState) -> None:
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            now = time.time()
            conn.execute(
                "UPDATE sessions SET state_json = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(asdict(state), ensure_ascii=False), now, session_id),
            )
            conn.commit()

    def get_history(self, session_id: str, limit: int) -> List[ChatMessage]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            rows = conn.execute(
                "SELECT role, content, ts FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            out = [ChatMessage(role=row["role"], content=row["content"], ts=row["ts"]) for row in rows]
            return list(reversed(out))

    def append_message(self, session_id: str, role: Role, content: str, max_history: int) -> None:
        now = time.time()
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            if max_history > 0:
                conn.execute(
                    """
                    DELETE FROM messages
                    WHERE session_id = ?
                      AND id NOT IN (
                        SELECT id FROM messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                      )
                    """,
                    (session_id, session_id, max_history),
                )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()


def make_store(storage: str, sqlite_path: str) -> Store:
    if storage == "sqlite":
        return SQLiteStore(sqlite_path)
    return MemoryStore()
