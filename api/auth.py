"""Header authentication shared by protected engine routes."""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthResult:
    """Validated engine authentication details."""

    api_key: str
    method: str


def authenticate_engine_key(
    authorization: str | None = None,
    x_api_key: str | None = None,
) -> AuthResult:
    """Validate either a standard Bearer token or the legacy X-API-Key header."""
    if not settings.api_keys:
        logger.error("No engine API key is configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine authentication is not configured",
        )

    candidate: str | None = None
    method: str | None = None

    if authorization:
        scheme, separator, token = authorization.strip().partition(" ")
        if separator and scheme.lower() == "bearer" and token.strip():
            candidate = token.strip()
            method = "Bearer"

    # Preserve compatibility with existing Edge Functions and other clients.
    if candidate is None and x_api_key:
        candidate = x_api_key.strip()
        method = "X-API-Key"

    if not candidate or not any(
        hmac.compare_digest(candidate, key) for key in settings.api_keys
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthResult(api_key=candidate, method=method or "unknown")


async def validate_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """FastAPI dependency used by protected engine routes."""
    return authenticate_engine_key(authorization, x_api_key).api_key
