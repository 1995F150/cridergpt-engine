"""Layer 2: structured, user-scoped project knowledge.

Project knowledge stores durable technical and business context separately from
short-lived conversation memory. Retrieval is intentionally bounded and uses a
small keyword score until the semantic RAG layer is introduced later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from memory.memory_store import get_supabase

PROJECT_TABLE = "project_knowledge"
MAX_RESULTS = 8
MAX_CONTENT_CHARS = 6000


@dataclass(slots=True)
class ProjectKnowledge:
    id: str
    user_id: str
    project_key: str
    project_name: str
    category: str
    title: str
    content: str
    status: str
    priority: int
    source: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ProjectKnowledge":
        return cls(
            id=str(row.get("id") or ""),
            user_id=str(row.get("user_id") or ""),
            project_key=str(row.get("project_key") or "general")[:100],
            project_name=str(row.get("project_name") or "Unnamed project")[:200],
            category=str(row.get("category") or "general")[:100],
            title=str(row.get("title") or "Untitled")[:300],
            content=str(row.get("content") or "")[:MAX_CONTENT_CHARS],
            status=str(row.get("status") or "active")[:30],
            priority=max(0, min(100, int(row.get("priority") or 0))),
            source=(str(row.get("source"))[:500] if row.get("source") else None),
            updated_at=(str(row.get("updated_at")) if row.get("updated_at") else None),
        )

    def to_prompt_line(self) -> str:
        return (
            f"[{self.project_name} / {self.category} / {self.status}] "
            f"{self.title}: {self.content}"
        )


def get_project_knowledge(
    user_id: str | None,
    message: str,
    *,
    limit: int = MAX_RESULTS,
) -> list[ProjectKnowledge]:
    if not user_id:
        return []
    client = get_supabase()
    if client is None:
        return []

    try:
        rows = (
            client.table(PROJECT_TABLE)
            .select(
                "id,user_id,project_key,project_name,category,title,content,"
                "status,priority,source,updated_at"
            )
            .eq("user_id", user_id)
            .eq("is_active", True)
            .order("priority", desc=True)
            .order("updated_at", desc=True)
            .limit(50)
            .execute()
            .data
            or []
        )
    except Exception:
        return []

    terms = _terms(message)
    ranked = sorted(
        (ProjectKnowledge.from_row(row) for row in rows),
        key=lambda item: (_score(item, terms), item.priority, item.updated_at or ""),
        reverse=True,
    )
    if terms:
        ranked = [item for item in ranked if _score(item, terms) > 0]
    return ranked[: max(1, min(limit, MAX_RESULTS))]


def _terms(message: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-zA-Z0-9_-]+", (message or "").lower())
        if len(word) >= 3
    }


def _score(item: ProjectKnowledge, terms: set[str]) -> int:
    if not terms:
        return 1
    project = f"{item.project_key} {item.project_name}".lower()
    heading = f"{item.category} {item.title}".lower()
    body = item.content.lower()
    return sum(
        6 if term in project else 3 if term in heading else 1 if term in body else 0
        for term in terms
    )
