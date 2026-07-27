"""Video job persistence and provider orchestration."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from memory.memory_store import get_supabase
from video.providers import ProviderJob, VideoProviderError, get_provider

logger = logging.getLogger(__name__)
TERMINAL_STATUSES = {"completed", "succeeded", "failed", "error", "cancelled", "canceled"}


def _client():
    client = get_supabase()
    if client is None:
        raise VideoProviderError("Supabase is not configured")
    return client


def _save(row: dict[str, Any]) -> dict[str, Any]:
    data = _client().table("video_generation_jobs").upsert(row).execute().data or []
    return data[0] if data else row


def _provider_fields(job: ProviderJob) -> dict[str, Any]:
    return {
        "provider_job_id": job.provider_job_id,
        "status": job.status,
        "output_url": job.output_url,
        "preview_url": job.preview_url,
        "error_message": job.error,
        "provider_payload": job.raw or {},
    }


def create_job(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    internal_id = str(uuid4())
    submitted = get_provider().create(payload)
    return _save(
        {
            "id": internal_id,
            "user_id": user_id,
            "prompt": payload["prompt"],
            "negative_prompt": payload.get("negative_prompt"),
            "model": payload.get("model"),
            "duration_seconds": payload.get("duration_seconds"),
            "aspect_ratio": payload.get("aspect_ratio"),
            "reference_image_url": payload.get("reference_image_url"),
            "provider": "http",
            **_provider_fields(submitted),
        }
    )


def get_job(user_id: str, job_id: str, refresh: bool = True) -> dict[str, Any]:
    rows = (
        _client()
        .table("video_generation_jobs")
        .select("*")
        .eq("id", job_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        raise KeyError(job_id)
    row = rows[0]
    if refresh and row.get("status") not in TERMINAL_STATUSES:
        latest = get_provider().get(str(row["provider_job_id"]))
        row = _save({"id": job_id, "user_id": user_id, **_provider_fields(latest)})
    return row


def list_jobs(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    return (
        _client()
        .table("video_generation_jobs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(max(1, min(limit, 100)))
        .execute()
        .data
        or []
    )


def cancel_job(user_id: str, job_id: str) -> dict[str, Any]:
    row = get_job(user_id, job_id, refresh=False)
    cancelled = get_provider().cancel(str(row["provider_job_id"]))
    return _save({"id": job_id, "user_id": user_id, **_provider_fields(cancelled)})


def apply_webhook(provider_job_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    rows = (
        _client()
        .table("video_generation_jobs")
        .select("id,user_id")
        .eq("provider_job_id", provider_job_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        logger.warning("Ignoring webhook for unknown provider job %s", provider_job_id)
        return None
    normalized = get_provider()._normalize(payload)
    return _save({"id": rows[0]["id"], "user_id": rows[0]["user_id"], **_provider_fields(normalized)})
