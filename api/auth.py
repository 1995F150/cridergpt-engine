"""Header authentication shared by protected engine routes."""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import Header, HTTPException, status

from api.key_store import validate_generated_key
from config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthResult:
    """Validated engine authentication details."""

    api_key: str
    method: str
    metadata: dict[str, Any] | None = None


def authenticate_engine_key(
    authorization: str | None = None,
    x_api_key: str | None = None,
) -> AuthResult:
    """Validate either a standard Bearer token or the legacy X-API-Key header.

    Keys may come from the server environment or from the dashboard-managed,
    hashed key store under data/api_keys.json.
    """
    candidate: str | None = None
    method: str | None = None

    if authorization:
        scheme, separator, token = authorization.strip().partition(" ")
        if separator and scheme.lower() == "bearer" and token.strip():
            candidate = token.strip()
            method = "Bearer"

    if candidate is None and x_api_key:
        candidate = x_api_key.strip()
        method = "X-API-Key"

    if candidate:
        if any(hmac.compare_digest(candidate, key) for key in settings.api_keys):
            return AuthResult(api_key=candidate, method=method or "unknown", metadata={"source": "environment"})

        generated = validate_generated_key(candidate)
        if generated is not None:
            return AuthResult(api_key=candidate, method=method or "unknown", metadata={"source": "dashboard", **generated})

    if not settings.api_keys:
        logger.warning("No environment API key matched; dashboard-generated keys may not exist yet")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def validate_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """FastAPI dependency used by protected engine routes."""
    return authenticate_engine_key(authorization, x_api_key).api_key
