"""Layer 3: durable, user-scoped structured memory facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory.memory_store import get_supabase

TABLE = "memory_facts"
ALLOWED_STATUS = {"active", "historical", "superseded", "disputed"}
ALLOWED_SENSITIVITY = {"public", "internal", "private", "highly_sensitive"}


@dataclass(slots=True)
class MemoryFact:
    id: str
    user_id: str
    subject: str
    predicate: str
    value: str
    category: str
    status: str
    sensitivity: str
    confidence: float
    valid_from: str | None
    valid_until: str | None
    source: str | None
    source_date: str | None
    supersedes_id: str | None
    last_verified_at: str | None
    review_after: str | None
    created_at: str | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryFact":
        return cls(
            id=str(row.get("id") or ""),
            user_id=str(row.get("user_id") or ""),
            subject=str(row.get("subject") or "").strip(),
            predicate=str(row.get("predicate") or "").strip(),
            value=str(row.get("value") or "").strip(),
            category=str(row.get("category") or "general").strip(),
            status=str(row.get("status") or "active").strip(),
            sensitivity=str(row.get("sensitivity") or "private").strip(),
            confidence=max(0.0, min(1.0, float(row.get("confidence") or 0.0))),
            valid_from=row.get("valid_from"),
            valid_until=row.get("valid_until"),
            source=row.get("source"),
            source_date=row.get("source_date"),
            supersedes_id=row.get("supersedes_id"),
            last_verified_at=row.get("last_verified_at"),
            review_after=row.get("review_after"),
            created_at=row.get("created_at"),
        )

    def is_current(self, now: datetime | None = None) -> bool:
        if self.status != "active":
            return False
        now = now or datetime.now(timezone.utc)
        if self.valid_until:
            try:
                if datetime.fromisoformat(self.valid_until.replace("Z", "+00:00")) < now:
                    return False
            except ValueError:
                pass
        return True

    def to_prompt_line(self) -> str:
        confidence = round(self.confidence, 2)
        timing = ""
        if self.valid_from or self.valid_until:
            timing = f" valid={self.valid_from or '?'}..{self.valid_until or 'open'}"
        return (
            f"[{self.category}] {self.subject} {self.predicate} {self.value} "
            f"(confidence={confidence}, status={self.status}{timing})"
        )


def _terms(message: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9]+", message.lower()) if len(w) >= 3][:8]


def get_structured_memories(
    user_id: str | None,
    message: str = "",
    *,
    limit: int = 12,
    include_historical: bool = False,
) -> list[MemoryFact]:
    if not user_id:
        return []
    client = get_supabase()
    if client is None:
        return []

    query = (
        client.table(TABLE)
        .select(
            "id,user_id,subject,predicate,value,category,status,sensitivity,confidence,"
            "valid_from,valid_until,source,source_date,supersedes_id,last_verified_at,"
            "review_after,created_at"
        )
        .eq("user_id", user_id)
        .neq("sensitivity", "highly_sensitive")
    )
    if not include_historical:
        query = query.eq("status", "active")
    terms = _terms(message)
    if terms:
        filters = ",".join(
            f"subject.ilike.%{t}%,predicate.ilike.%{t}%,value.ilike.%{t}%,category.ilike.%{t}%"
            for t in terms
        )
        query = query.or_(filters)

    rows = query.order("confidence", desc=True).order("created_at", desc=True).limit(limit * 3).execute().data or []
    facts = [MemoryFact.from_row(row) for row in rows]
    current = [fact for fact in facts if include_historical or fact.is_current()]
    current.sort(key=lambda fact: (fact.confidence, fact.created_at or ""), reverse=True)
    return current[:limit]


def upsert_memory_fact(user_id: str, payload: dict[str, Any]) -> MemoryFact:
    if not user_id:
        raise ValueError("user_id is required")
    subject = str(payload.get("subject") or "").strip()
    predicate = str(payload.get("predicate") or "").strip()
    value = str(payload.get("value") or "").strip()
    if not subject or not predicate or not value:
        raise ValueError("subject, predicate, and value are required")

    status = str(payload.get("status") or "active")
    sensitivity = str(payload.get("sensitivity") or "private")
    if status not in ALLOWED_STATUS:
        raise ValueError("invalid status")
    if sensitivity not in ALLOWED_SENSITIVITY:
        raise ValueError("invalid sensitivity")

    client = get_supabase()
    if client is None:
        raise RuntimeError("Supabase is not configured")

    row = {
        "user_id": user_id,
        "subject": subject[:300],
        "predicate": predicate[:120],
        "value": value[:8000],
        "category": str(payload.get("category") or "general")[:120],
        "status": status,
        "sensitivity": sensitivity,
        "confidence": max(0.0, min(1.0, float(payload.get("confidence", 1.0)))),
        "valid_from": payload.get("valid_from"),
        "valid_until": payload.get("valid_until"),
        "source": payload.get("source"),
        "source_date": payload.get("source_date"),
        "supersedes_id": payload.get("supersedes_id"),
        "last_verified_at": payload.get("last_verified_at"),
        "review_after": payload.get("review_after"),
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    }
    result = client.table(TABLE).insert(row).execute()
    saved = (result.data or [row])[0]

    supersedes_id = row.get("supersedes_id")
    if supersedes_id:
        client.table(TABLE).update({"status": "superseded"}).eq("id", supersedes_id).eq("user_id", user_id).execute()

    return MemoryFact.from_row(saved)
