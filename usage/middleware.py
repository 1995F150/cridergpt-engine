"""Body-safe ASGI usage metering for every AI-facing route."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from usage.meter import record_usage
from usage.tokenizer import Modality, estimate_usage

ASGIApp = Callable[[dict[str, Any], Callable[..., Awaitable[dict[str, Any]]], Callable[..., Awaitable[None]]], Awaitable[None]]


class UsageMeterMiddleware:
    """Meter request and JSON response content without consuming either stream."""

    ROUTES: tuple[tuple[str, Modality], ...] = (
        ("/chat", "text"),
        ("/embeddings", "embedding"),
        ("/image/generate", "image"),
        ("/image/analyze", "vision"),
        ("/video/generate", "video"),
        ("/audio", "audio"),
        ("/tools", "tool"),
    )
    MAX_CAPTURE_BYTES = 2 * 1024 * 1024

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    @classmethod
    def _modality(cls, path: str) -> Modality | None:
        normalized = path.removeprefix("/api")
        for prefix, modality in cls.ROUTES:
            if normalized.startswith(prefix):
                return modality
        return None

    @staticmethod
    def _text_values(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                result.extend(UsageMeterMiddleware._text_values(item))
            return result
        if isinstance(value, dict):
            result = []
            for key, item in value.items():
                if key.lower() in {
                    "prompt", "message", "messages", "text", "content", "response",
                    "description", "negative_prompt", "narration", "sound_prompt",
                    "music_prompt", "query", "system_prompt",
                }:
                    result.extend(UsageMeterMiddleware._text_values(item))
            return result
        return []

    @staticmethod
    def _number(payload: dict[str, Any], name: str, default: float = 0) -> float:
        try:
            return float(payload.get(name, default) or default)
        except (TypeError, ValueError):
            return default

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        modality = self._modality(str(scope.get("path") or ""))
        method = str(scope.get("method") or "GET").upper()
        if modality is None or method not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        request_messages: list[dict[str, Any]] = []
        request_body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            request_messages.append(message)
            if message.get("type") == "http.request":
                chunk = message.get("body", b"")
                if len(request_body) < self.MAX_CAPTURE_BYTES:
                    request_body.extend(chunk[: self.MAX_CAPTURE_BYTES - len(request_body)])
                more_body = bool(message.get("more_body", False))
            else:
                more_body = False

        message_index = 0

        async def replay_receive() -> dict[str, Any]:
            nonlocal message_index
            if message_index < len(request_messages):
                message = request_messages[message_index]
                message_index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        payload: dict[str, Any] = {}
        try:
            parsed = json.loads(bytes(request_body) or b"{}")
            if isinstance(parsed, dict):
                payload = parsed
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            pass

        response_body = bytearray()
        response_start: dict[str, Any] | None = None

        async def capture_send(message: dict[str, Any]) -> None:
            nonlocal response_start
            if message.get("type") == "http.response.start":
                response_start = message
                return
            if message.get("type") == "http.response.body":
                chunk = message.get("body", b"")
                if len(response_body) < self.MAX_CAPTURE_BYTES:
                    response_body.extend(chunk[: self.MAX_CAPTURE_BYTES - len(response_body)])
                if not message.get("more_body", False):
                    output_text = ""
                    try:
                        decoded = json.loads(bytes(response_body) or b"{}")
                        output_text = " ".join(self._text_values(decoded))
                    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                        pass

                    input_text = " ".join(self._text_values(payload))
                    estimate = estimate_usage(
                        modality,
                        input_text=input_text,
                        output_text=output_text,
                        width=int(self._number(payload, "width")),
                        height=int(self._number(payload, "height")),
                        image_count=max(1, int(self._number(payload, "image_count", 1))),
                        duration_seconds=self._number(payload, "duration_seconds"),
                        fps=max(1, int(self._number(payload, "fps", 24))),
                        include_audio=bool(payload.get("include_audio", False)),
                    )
                    if response_start is not None:
                        headers = list(response_start.get("headers", []))
                        headers.extend(
                            [
                                (b"x-cridergpt-input-tokens", str(estimate.input_tokens).encode()),
                                (b"x-cridergpt-output-tokens", str(estimate.output_tokens).encode()),
                                (b"x-cridergpt-media-tokens", str(estimate.media_tokens).encode()),
                                (b"x-cridergpt-total-tokens", str(estimate.total_tokens).encode()),
                            ]
                        )
                        response_start["headers"] = headers
                        await send(response_start)
                    record_usage(
                        str(payload.get("user_id") or "system"),
                        str(scope.get("path") or "").removeprefix("/api").strip("/").replace("/", "."),
                        estimate,
                        model=str(payload.get("model")) if payload.get("model") else None,
                        metadata={"status_code": response_start.get("status") if response_start else None},
                    )
                await send(message)
                return
            await send(message)

        await self.app(scope, replay_receive, capture_send)
