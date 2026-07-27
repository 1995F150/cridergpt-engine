"""Background worker for independent local video generation."""

from __future__ import annotations

import logging
import signal
import time
from typing import Any

from config import settings
from memory.memory_store import get_supabase
from video.models import LocalVideoRequest
from video.pipeline import LocalVideoPipeline

logger = logging.getLogger(__name__)
_STOP = False


def _stop(*_args: Any) -> None:
    global _STOP
    _STOP = True


def _next_job() -> dict[str, Any] | None:
    client = get_supabase()
    if client is None:
        return None
    rows = (
        client.table("video_generation_jobs")
        .select("*")
        .eq("provider", "local")
        .eq("status", "queued")
        .order("created_at")
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        return None
    row = rows[0]
    claimed = (
        client.table("video_generation_jobs")
        .update({"status": "processing", "progress": 1})
        .eq("id", row["id"])
        .eq("status", "queued")
        .execute()
        .data
        or []
    )
    return claimed[0] if claimed else None


def _update(job_id: str, values: dict[str, Any]) -> None:
    client = get_supabase()
    if client is not None:
        client.table("video_generation_jobs").update(values).eq("id", job_id).execute()


def process(row: dict[str, Any]) -> None:
    payload = row.get("provider_payload") or {}
    request = LocalVideoRequest(job_id=str(row["id"]), **payload)
    try:
        result = LocalVideoPipeline().run(request, str(row["user_id"]))
        _update(
            str(row["id"]),
            {
                "status": "completed",
                "progress": 100,
                "output_url": f"/video/files/{row['id']}/video.mp4",
                "preview_url": f"/video/files/{row['id']}/preview.jpg",
                "provider_payload": {**payload, **result.metadata},
                "error_message": None,
            },
        )
    except Exception as exc:
        logger.exception("Local video job %s failed", row.get("id"))
        _update(str(row["id"]), {"status": "failed", "error_message": str(exc)[:2000]})


def main() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    logger.info("CriderGPT local video worker started")
    while not _STOP:
        job = _next_job()
        if job:
            process(job)
        else:
            time.sleep(settings.video_worker_poll_seconds)


if __name__ == "__main__":
    main()
