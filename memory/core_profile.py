"""Layer 1: stable, user-scoped core profile support.

The core profile is intentionally small and structured. It stores durable facts,
preferences, goals, communication guidance, and project summaries without
pulling full conversation history into every request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from memory.memory_store import get_supabase

CORE_PROFILE_TABLE = "core_profiles"
MAX_LIST_ITEMS = 50
MAX_TEXT_LENGTH = 4000


@dataclass(slots=True)
class CoreProfile:
    user_id: str
    display_name: str | None = None
    location: str | None = None
    bio: str | None = None
    stable_facts: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    communication: dict[str, Any] = field(default_factory=dict)
    projects: list[dict[str, Any]] = field(default_factory=list)
    privacy: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CoreProfile":
        return cls(
            user_id=str(row.get("user_id") or ""),
            display_name=_clean_text(row.get("display_name")),
            location=_clean_text(row.get("location")),
            bio=_clean_text(row.get("bio")),
            stable_facts=_clean_text_list(row.get("stable_facts")),
            preferences=_clean_mapping(row.get("preferences")),
            goals=_clean_text_list(row.get("goals")),
            communication=_clean_mapping(row.get("communication")),
            projects=_clean_projects(row.get("projects")),
            privacy=_clean_mapping(row.get("privacy")),
            version=max(1, int(row.get("version") or 1)),
            updated_at=_clean_text(row.get("updated_at")),
        )

    def to_prompt_section(self) -> str:
        lines = ["CORE USER PROFILE:"]
        if self.display_name:
            lines.append(f"display_name: {self.display_name}")
        if self.location:
            lines.append(f"location: {self.location}")
        if self.bio:
            lines.append(f"bio: {self.bio}")
        if self.stable_facts:
            lines.append("stable_facts:")
            lines.extend(f"- {item}" for item in self.stable_facts)
        if self.preferences:
            lines.append("preferences:")
            lines.extend(f"- {key}: {value}" for key, value in self.preferences.items())
        if self.goals:
            lines.append("goals:")
            lines.extend(f"- {item}" for item in self.goals)
        if self.communication:
            lines.append("communication:")
            lines.extend(f"- {key}: {value}" for key, value in self.communication.items())
        if self.projects:
            lines.append("projects:")
            for project in self.projects:
                name = project.get("name", "Unnamed project")
                status = project.get("status")
                summary = project.get("summary")
                suffix = ", ".join(part for part in (status, summary) if part)
                lines.append(f"- {name}" + (f": {suffix}" if suffix else ""))
        return "\n".join(lines)


def get_core_profile(user_id: str | None) -> CoreProfile | None:
    if not user_id:
        return None
    client = get_supabase()
    if client is None:
        return None
    try:
        rows = (
            client.table(CORE_PROFILE_TABLE)
            .select(
                "user_id,display_name,location,bio,stable_facts,preferences,goals,"
                "communication,projects,privacy,version,updated_at"
            )
            .eq("user_id", user_id)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception:
        return None
    return CoreProfile.from_row(rows[0]) if rows else None


def upsert_core_profile(user_id: str, payload: dict[str, Any]) -> CoreProfile:
    if not user_id:
        raise ValueError("user_id is required")
    client = get_supabase()
    if client is None:
        raise RuntimeError("Supabase is not configured")

    current = get_core_profile(user_id)
    merged = _merge_payload(current, user_id, payload)
    row = {
        "user_id": merged.user_id,
        "display_name": merged.display_name,
        "location": merged.location,
        "bio": merged.bio,
        "stable_facts": merged.stable_facts,
        "preferences": merged.preferences,
        "goals": merged.goals,
        "communication": merged.communication,
        "projects": merged.projects,
        "privacy": merged.privacy,
        "version": merged.version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    result = client.table(CORE_PROFILE_TABLE).upsert(row, on_conflict="user_id").execute()
    saved = (result.data or [row])[0]
    return CoreProfile.from_row(saved)


def _merge_payload(
    current: CoreProfile | None, user_id: str, payload: dict[str, Any]
) -> CoreProfile:
    base = current or CoreProfile(user_id=user_id)
    return CoreProfile(
        user_id=user_id,
        display_name=_clean_text(payload.get("display_name", base.display_name)),
        location=_clean_text(payload.get("location", base.location)),
        bio=_clean_text(payload.get("bio", base.bio)),
        stable_facts=_clean_text_list(payload.get("stable_facts", base.stable_facts)),
        preferences={**base.preferences, **_clean_mapping(payload.get("preferences"))},
        goals=_clean_text_list(payload.get("goals", base.goals)),
        communication={
            **base.communication,
            **_clean_mapping(payload.get("communication")),
        },
        projects=_clean_projects(payload.get("projects", base.projects)),
        privacy={**base.privacy, **_clean_mapping(payload.get("privacy"))},
        version=base.version + 1 if current else 1,
        updated_at=base.updated_at,
    )


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:MAX_TEXT_LENGTH] or None


def _clean_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value[:MAX_LIST_ITEMS]:
        text = _clean_text(item)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _clean_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key)[:100]: item for key, item in list(value.items())[:MAX_LIST_ITEMS]}


def _clean_projects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    projects: list[dict[str, Any]] = []
    for item in value[:MAX_LIST_ITEMS]:
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("name"))
        if not name:
            continue
        projects.append(
            {
                "name": name,
                "status": _clean_text(item.get("status")),
                "summary": _clean_text(item.get("summary")),
            }
        )
    return projects
