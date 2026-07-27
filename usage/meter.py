"""Best-effort persistence for unified AI usage events."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from memory.memory_store import get_supabase
from usage.tokenizer import UsageEstimate

logger = logging.getLogger(__name__)


def record_usage(
    user_id: str,
    tool: str,
    estimate: UsageEstimate,
    *,
    model: str | None = None,
    request_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": str(uuid4()),
        "user_id": user_id,
        "request_id": request_id,
        "tool": tool,
        "modality": estimate.modality,
        "model": model,
        "input_tokens": estimate.input_tokens,
        "output_tokens": estimate.output_tokens,
        "media_tokens": estimate.media_tokens,
        "total_tokens": estimate.total_tokens,
        "details": {**estimate.details, **(metadata or {})},
    }
    client = get_supabase()
    if client is None:
        return row
    try:
        data = client.table("ai_usage_events").insert(row).execute().data or []
        return data[0] if data else row
    except Exception as exc:
        logger.warning("Usage event persistence failed: %s", exc)
        return row
