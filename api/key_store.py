"""Persistent API key storage for the CriderGPT Engine.

Generated keys are shown only once. Only a SHA-256 digest is stored on disk.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "api_keys.json"
_LOCK = threading.RLock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load() -> list[dict[str, Any]]:
    with _LOCK:
        if not STORE_PATH.exists():
            return []
        try:
            value = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []


def _save(rows: list[dict[str, Any]]) -> None:
    with _LOCK:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = STORE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        temporary.replace(STORE_PATH)


def _digest(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def create_key(
    *,
    name: str,
    model: str = "*",
    max_tokens: int = 4096,
    requests_per_minute: int = 60,
    expires_in_days: int | None = None,
) -> dict[str, Any]:
    raw_key = "cgpt_" + secrets.token_urlsafe(32)
    created_at = _now()
    expires_at = (
        (created_at + timedelta(days=expires_in_days)).isoformat()
        if expires_in_days and expires_in_days > 0
        else None
    )
    row = {
        "id": str(uuid.uuid4()),
        "name": name.strip() or "Unnamed key",
        "prefix": raw_key[:12],
        "key_hash": _digest(raw_key),
        "model": model.strip() or "*",
        "max_tokens": max(1, min(int(max_tokens), 131072)),
        "requests_per_minute": max(1, min(int(requests_per_minute), 10000)),
        "created_at": created_at.isoformat(),
        "expires_at": expires_at,
        "last_used_at": None,
        "revoked": False,
    }
    rows = _load()
    rows.append(row)
    _save(rows)
    return {**public_key(row), "api_key": raw_key}


def public_key(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "key_hash"}


def list_keys() -> list[dict[str, Any]]:
    return [public_key(row) for row in reversed(_load())]


def revoke_key(key_id: str) -> bool:
    rows = _load()
    changed = False
    for row in rows:
        if row.get("id") == key_id and not row.get("revoked"):
            row["revoked"] = True
            changed = True
            break
    if changed:
        _save(rows)
    return changed


def validate_generated_key(raw_key: str) -> dict[str, Any] | None:
    candidate = _digest(raw_key)
    rows = _load()
    now = _now()
    for row in rows:
        if not secrets.compare_digest(candidate, str(row.get("key_hash", ""))):
            continue
        if row.get("revoked"):
            return None
        expires_at = row.get("expires_at")
        if expires_at:
            try:
                if datetime.fromisoformat(expires_at) <= now:
                    return None
            except ValueError:
                return None
        row["last_used_at"] = now.isoformat()
        _save(rows)
        return public_key(row)
    return None


def active_key_count() -> int:
    now = _now()
    total = 0
    for row in _load():
        if row.get("revoked"):
            continue
        expires_at = row.get("expires_at")
        if expires_at:
            try:
                if datetime.fromisoformat(expires_at) <= now:
                    continue
            except ValueError:
                continue
        total += 1
    return total
