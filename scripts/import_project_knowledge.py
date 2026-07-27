"""Import Layer 2 project knowledge from a JSON array.

Usage:
    .venv/bin/python scripts/import_project_knowledge.py data/projects.json USER_UUID
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from memory.memory_store import get_supabase


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: import_project_knowledge.py <json-file> <user-id>")
        return 2

    path = Path(sys.argv[1])
    user_id = sys.argv[2].strip()
    if not user_id:
        raise ValueError("user-id is required")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("JSON root must be an array")

    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        required = ("project_key", "project_name", "title", "content")
        missing = [key for key in required if not str(item.get(key) or "").strip()]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        rows.append(
            {
                "user_id": user_id,
                "project_key": str(item["project_key"])[:100],
                "project_name": str(item["project_name"])[:200],
                "category": str(item.get("category") or "general")[:100],
                "title": str(item["title"])[:300],
                "content": str(item["content"])[:12000],
                "status": str(item.get("status") or "active")[:30],
                "priority": max(0, min(100, int(item.get("priority") or 50))),
                "source": str(item.get("source") or "json-import")[:500],
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                "is_active": bool(item.get("is_active", True)),
            }
        )

    client = get_supabase()
    if client is None:
        raise RuntimeError("Supabase is not configured")
    if rows:
        client.table("project_knowledge").insert(rows).execute()
    print(f"Imported {len(rows)} project knowledge rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
