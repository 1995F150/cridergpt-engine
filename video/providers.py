"""Pluggable HTTP video generation providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from config import settings


class VideoProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderJob:
    provider_job_id: str
    status: str
    output_url: str | None = None
    preview_url: str | None = None
    error: str | None = None
    raw: dict[str, Any] | None = None


class HttpVideoProvider:
    """Generic bearer-auth JSON provider with configurable endpoint paths."""

    def __init__(self) -> None:
        if not settings.video_api_url:
            raise VideoProviderError("VIDEO_API_URL is not configured")
        self.base_url = settings.video_api_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if settings.video_api_key:
            self.headers["Authorization"] = f"Bearer {settings.video_api_key}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = requests.request(
                method,
                f"{self.base_url}/{path.lstrip('/')}",
                headers=self.headers,
                timeout=(10, settings.video_timeout_seconds),
                **kwargs,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            raise VideoProviderError(f"Video provider request failed: {exc}") from exc
        except ValueError as exc:
            raise VideoProviderError("Video provider returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise VideoProviderError("Video provider returned an invalid payload")
        return body

    @staticmethod
    def _normalize(body: dict[str, Any]) -> ProviderJob:
        job_id = body.get("id") or body.get("job_id") or body.get("task_id")
        if not job_id:
            raise VideoProviderError("Video provider did not return a job ID")
        status = str(body.get("status") or "queued").lower()
        output_url = body.get("output_url") or body.get("video_url") or body.get("url")
        preview_url = body.get("preview_url") or body.get("thumbnail_url")
        error = body.get("error") or body.get("message") if status in {"failed", "error"} else None
        return ProviderJob(
            provider_job_id=str(job_id),
            status=status,
            output_url=str(output_url) if output_url else None,
            preview_url=str(preview_url) if preview_url else None,
            error=str(error) if error else None,
            raw=body,
        )

    def create(self, payload: dict[str, Any]) -> ProviderJob:
        body = self._request("POST", settings.video_create_path, json=payload)
        return self._normalize(body)

    def get(self, provider_job_id: str) -> ProviderJob:
        path = settings.video_status_path.format(job_id=provider_job_id)
        return self._normalize(self._request("GET", path))

    def cancel(self, provider_job_id: str) -> ProviderJob:
        path = settings.video_cancel_path.format(job_id=provider_job_id)
        return self._normalize(self._request("POST", path))


def get_provider() -> HttpVideoProvider:
    if settings.video_backend != "http":
        raise VideoProviderError(f"Unsupported VIDEO_BACKEND: {settings.video_backend}")
    return HttpVideoProvider()
