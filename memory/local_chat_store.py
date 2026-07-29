"""Local SQLite persistence for the built-in engine chat UI."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "local_chat.sqlite3"
_LOCK = threading.Lock()


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS local_chat_users (
            id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            profile_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS local_chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES local_chat_users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_local_chat_messages_conversation
        ON local_chat_messages(user_id, conversation_id, id);
        """
    )
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_user(user_id: str, display_name: str | None = None) -> None:
    now = _now()
    name = (display_name or user_id).strip()[:120] or user_id
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO local_chat_users(id, display_name, created_at, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name=excluded.display_name,
                updated_at=excluded.updated_at
            """,
            (user_id, name, now, now),
        )


def list_messages(user_id: str, conversation_id: str, limit: int = 40) -> list[dict[str, Any]]:
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM local_chat_messages
            WHERE user_id=? AND conversation_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, conversation_id, max(1, min(limit, 200))),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def append_message(user_id: str, conversation_id: str, role: str, content: str) -> None:
    ensure_user(user_id)
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO local_chat_messages(user_id, conversation_id, role, content, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (user_id, conversation_id, role, content, _now()),
        )


def get_profile(user_id: str) -> dict[str, Any]:
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT display_name, profile_json FROM local_chat_users WHERE id=?",
            (user_id,),
        ).fetchone()
    if not row:
        return {"display_name": user_id, "profile": {}}
    try:
        profile = json.loads(row["profile_json"] or "{}")
    except json.JSONDecodeError:
        profile = {}
    return {"display_name": row["display_name"], "profile": profile}


def update_profile(user_id: str, display_name: str | None, profile: dict[str, Any] | None) -> dict[str, Any]:
    ensure_user(user_id, display_name)
    now = _now()
    with _LOCK, _connect() as conn:
        current = conn.execute(
            "SELECT display_name, profile_json FROM local_chat_users WHERE id=?",
            (user_id,),
        ).fetchone()
        current_profile: dict[str, Any] = {}
        if current:
            try:
                current_profile = json.loads(current["profile_json"] or "{}")
            except json.JSONDecodeError:
                current_profile = {}
        if profile:
            current_profile.update(profile)
        final_name = (display_name or (current["display_name"] if current else user_id)).strip()[:120]
        conn.execute(
            """
            UPDATE local_chat_users
            SET display_name=?, profile_json=?, updated_at=?
            WHERE id=?
            """,
            (final_name, json.dumps(current_profile), now, user_id),
        )
    return {"display_name": final_name, "profile": current_profile}
