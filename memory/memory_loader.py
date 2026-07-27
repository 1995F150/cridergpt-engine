"""Bounded, user-scoped context loading from the live CriderGPT schema."""

from __future__ import annotations

import logging
import re
from typing import Any

from config import settings
from memory.core_profile import get_core_profile
from memory.memory_store import get_supabase
from memory.project_knowledge import get_project_knowledge

logger = logging.getLogger(__name__)


def _safe_rows(table: str, select: str, configure=None) -> list[dict[str, Any]]:
    client = get_supabase()
    if client is None:
        return []
    try:
        query = client.table(table).select(select)
        if configure:
            query = configure(query)
        return query.execute().data or []
    except Exception as exc:
        logger.warning("Context query failed for %s: %s", table, exc)
        return []


def get_writing_samples(limit: int | None = None) -> list[dict[str, Any]]:
    count = limit or settings.writing_sample_limit
    return _safe_rows(
        "writing_samples",
        "title,content,category,created_at",
        lambda q: q.order("created_at", desc=False).limit(count),
    )


def get_ai_memory(
    user_id: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    if not user_id:
        return []
    count = limit or settings.memory_limit
    return _safe_rows(
        "ai_memory",
        "category,topic,details,content,source,created_at",
        lambda q: q.eq("user_id", user_id).order("created_at", desc=True).limit(count),
    )


def get_conversation_history(
    user_id: str | None = None,
    conversation_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if not user_id:
        return []

    def configure(q):
        q = q.eq("user_id", user_id)
        if conversation_id:
            q = q.eq("conversation_id", conversation_id)
        return q.order("created_at", desc=True).limit(limit)

    return list(
        reversed(_safe_rows("chat_messages", "role,content,created_at", configure))
    )


def get_user_preferences(user_id: str | None, limit: int = 20) -> list[dict[str, Any]]:
    if not user_id:
        return []
    return _safe_rows(
        "user_preferences",
        "preference_type,preference_value,confidence",
        lambda q: q.eq("user_id", user_id).order("confidence", desc=True).limit(limit),
    )


def get_profile(user_id: str | None) -> dict[str, Any] | None:
    if not user_id:
        return None
    rows = _safe_rows(
        "profiles",
        "display_name,bio,role,memory_enabled",
        lambda q: q.eq("user_id", user_id).limit(1),
    )
    return rows[0] if rows else None


def get_training_corpus(
    message: str = "", limit: int | None = None
) -> list[dict[str, Any]]:
    count = limit or settings.training_limit
    terms = [
        word for word in re.findall(r"[a-zA-Z0-9]+", message.lower()) if len(word) >= 4
    ][:6]

    def configure(q):
        if terms:
            filters = ",".join(
                f"content.ilike.%{term}%,topic.ilike.%{term}%" for term in terms
            )
            q = q.or_(filters)
        return q.limit(count)

    return _safe_rows("cridergpt_training_corpus", "category,topic,content", configure)


def get_user_training(
    user_id: str | None, message: str = "", limit: int = 5
) -> list[dict[str, Any]]:
    if not user_id:
        return []
    terms = [
        word for word in re.findall(r"[a-zA-Z0-9]+", message.lower()) if len(word) >= 4
    ][:6]

    def configure(q):
        q = q.eq("user_id", user_id).eq("is_active", True)
        if terms:
            q = q.or_(",".join(f"content.ilike.%{term}%" for term in terms))
        return q.limit(limit)

    return _safe_rows("training_inputs", "category,content", configure)


def build_context(
    user_id: str | None, conversation_id: str | None, message: str
) -> tuple[str, int]:
    sections: list[str] = []
    rows_used = 0

    core_profile = get_core_profile(user_id)
    if core_profile:
        rows_used += 1
        sections.append(core_profile.to_prompt_section())

    projects = get_project_knowledge(user_id, message)
    if projects:
        rows_used += len(projects)
        sections.append(
            "RELEVANT PROJECT KNOWLEDGE:\n"
            + "\n".join(item.to_prompt_line() for item in projects)
        )

    samples = get_writing_samples()
    if samples:
        rows_used += len(samples)
        sections.append(
            "WRITING STYLE SAMPLES:\n"
            + "\n".join(s.get("content", "")[:1200] for s in samples)
        )

    profile = get_profile(user_id)
    if profile:
        rows_used += 1
        sections.append(
            "LEGACY USER PROFILE:\n"
            + "\n".join(
                f"{key}: {value}"
                for key, value in profile.items()
                if value not in (None, "")
            )
        )

    memories = get_ai_memory(user_id)
    if memories:
        rows_used += len(memories)
        sections.append(
            "USER MEMORY:\n"
            + "\n".join(
                f"[{m.get('category', 'general')}] {m.get('topic', '')}: "
                f"{m.get('details') or m.get('content') or ''}"
                for m in memories
            )
        )

    preferences = get_user_preferences(user_id)
    if preferences:
        rows_used += len(preferences)
        sections.append(
            "USER PREFERENCES:\n"
            + "\n".join(
                f"{p.get('preference_type')}: {p.get('preference_value')}"
                for p in preferences
            )
        )

    corpus = get_training_corpus(message)
    user_training = get_user_training(user_id, message)
    training = corpus + user_training
    if training:
        rows_used += len(training)
        sections.append(
            "CRIDERGPT TRAINING CONTEXT:\n"
            + "\n".join(
                f"[{t.get('category', 'general')}] {t.get('topic', '')}: "
                f"{t.get('content', '')}"
                for t in training
            )
        )

    history = get_conversation_history(user_id, conversation_id)
    if history:
        rows_used += len(history)
        sections.append(
            "RECENT CONVERSATION:\n"
            + "\n".join(
                f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history
            )
        )

    return "\n\n".join(sections), rows_used


# Compatibility helpers used by the old test/module surface.
def get_profiles(user_id: str | None = None) -> list[dict[str, Any]]:
    profile = get_profile(user_id)
    return [profile] if profile else []


def get_chat_history(user_id: str | None = None) -> list[dict[str, Any]]:
    return get_conversation_history(user_id)
