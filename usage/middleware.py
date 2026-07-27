"""Automatic usage estimation for every AI-facing HTTP request."""

from __future__ import annotations

import json
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from usage.meter import record_usage
from usage.tokenizer import Modality, estimate_usage


class UsageMeterMiddleware(BaseHTTPMiddleware):
    """Attach token estimates and persist request usage without changing APIs."""

    ROUTES: tuple[tuple[str, Modality], ...] = (
        ("/chat", "text"),
        ("/image/generate", "image"),
        ("/image/analyze", "vision"),
        ("/video/generate", "video"),
        ("/audio", "audio"),
    )

    @classmethod
    def _modality(cls, path: str) -> Modality | None:
        normalized = path.removeprefix("/api")
        for prefix, modality in cls.ROUTES:
            if normalized.startswith(prefix):
                return modality
        return None

    async def dispatch(self, request: Request, call_next):
        modality = self._modality(request.url.path)
        if modality is None or request.method not in {"POST", "PUT", "PATCH"}:
            return await call_next(request)

        payload: dict[str, Any] = {}
        try:
            raw = await request.body()
            parsed = json.loads(raw or b"{}")
            if isinstance(parsed, dict):
                payload = parsed
        except (ValueError, json.JSONDecodeError):
            pass

        prompt_parts = [
            payload.get("prompt"), payload.get("message"), payload.get("text"),
            payload.get("negative_prompt"), payload.get("narration"),
            payload.get("sound_prompt"), payload.get("music_prompt"),
        ]
        estimate = estimate_usage(
            modality,
            input_text=" ".join(str(value) for value in prompt_parts if value),
            width=int(payload.get("width") or 0),
            height=int(payload.get("height") or 0),
            duration_seconds=float(payload.get("duration_seconds") or 0),
            fps=int(payload.get("fps") or 24),
            include_audio=bool(payload.get("include_audio", False)),
        )
        response = await call_next(request)
        response.headers["X-CriderGPT-Estimated-Tokens"] = str(estimate.total_tokens)
        response.headers["X-CriderGPT-Media-Tokens"] = str(estimate.media_tokens)
        record_usage(
            str(payload.get("user_id") or "system"),
            request.url.path.removeprefix("/api").strip("/").replace("/", "."),
            estimate,
            model=str(payload.get("model")) if payload.get("model") else None,
        )
        return response
