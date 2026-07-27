"""Chunk, embed, and upload a text or Markdown document to Layer 4 RAG."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memory.memory_store import get_supabase
from memory.semantic_rag import embed_text


def chunk_text(text: str, size: int = 1800, overlap: int = 250) -> list[str]:
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        if end < len(cleaned):
            boundary = max(cleaned.rfind("\n\n", start, end), cleaned.rfind(". ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


def ingest(user_id: str, path: Path, title: str | None, source_type: str, source_uri: str | None) -> str:
    client = get_supabase()
    if client is None:
        raise RuntimeError("Supabase is not configured")
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("document is empty")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source = client.table("rag_sources").insert({
        "user_id": user_id,
        "title": title or path.stem,
        "source_type": source_type,
        "source_uri": source_uri or str(path),
        "content_hash": digest,
        "metadata": {"filename": path.name, "chunk_count": len(chunks)},
    }).execute().data[0]
    source_id = source["id"]
    rows = []
    for index, chunk in enumerate(chunks):
        rows.append({
            "source_id": source_id,
            "user_id": user_id,
            "chunk_index": index,
            "content": chunk,
            "token_estimate": max(1, len(chunk) // 4),
            "embedding": embed_text(chunk),
        })
    client.table("rag_chunks").insert(rows).execute()
    return str(source_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id")
    parser.add_argument("path", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--source-type", default="document")
    parser.add_argument("--source-uri")
    args = parser.parse_args()
    source_id = ingest(args.user_id, args.path, args.title, args.source_type, args.source_uri)
    print(f"Ingested source {source_id}")


if __name__ == "__main__":
    main()
