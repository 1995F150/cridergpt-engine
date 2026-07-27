"""Layer 4 semantic retrieval backed by Ollama embeddings and pgvector."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from config import settings
from memory.memory_store import get_supabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagChunk:
    source_id: str
    title: str
    content: str
    source_type: str
    source_uri: str | None
    similarity: float

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "RagChunk":
        return cls(
            source_id=str(row.get("source_id") or ""),
            title=str(row.get("title") or "Untitled"),
            content=str(row.get("content") or ""),
            source_type=str(row.get("source_type") or "document"),
            source_uri=row.get("source_uri"),
            similarity=float(row.get("similarity") or 0.0),
        )

    def to_prompt_line(self) -> str:
        origin = f" ({self.source_uri})" if self.source_uri else ""
        return f"[{self.source_type}] {self.title}{origin}\n{self.content}"


def embed_text(text: str) -> list[float]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text is required")
    response = requests.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={"model": settings.embedding_model, "prompt": cleaned[: settings.rag_embedding_input_chars]},
        timeout=(5, settings.embedding_timeout_seconds),
    )
    response.raise_for_status()
    embedding = response.json().get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("embedding backend returned no vector")
    return [float(value) for value in embedding]


def get_semantic_context(
    user_id: str | None,
    query: str,
    *,
    limit: int | None = None,
    threshold: float | None = None,
) -> list[RagChunk]:
    if not user_id or not query.strip():
        return []
    client = get_supabase()
    if client is None:
        return []
    try:
        vector = embed_text(query)
        result = client.rpc(
            "match_rag_chunks",
            {
                "query_embedding": vector,
                "match_user_id": user_id,
                "match_count": limit or settings.rag_context_limit,
                "match_threshold": threshold if threshold is not None else settings.rag_similarity_threshold,
            },
        ).execute()
        rows = result.data or []
        chunks = [RagChunk.from_row(row) for row in rows]
        total = 0
        bounded: list[RagChunk] = []
        for chunk in chunks:
            content = chunk.content[: settings.rag_chunk_prompt_chars]
            if total + len(content) > settings.rag_total_prompt_chars:
                break
            bounded.append(RagChunk(chunk.source_id, chunk.title, content, chunk.source_type, chunk.source_uri, chunk.similarity))
            total += len(content)
        return bounded
    except Exception as exc:
        logger.warning("Semantic RAG unavailable: %s", exc)
        return []
