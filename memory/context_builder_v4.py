"""Layer 4 context builder: structured memory plus semantic retrieval."""

from __future__ import annotations

from memory.context_builder_v3 import build_context as build_v3_context
from memory.semantic_rag import get_semantic_context


def build_context(
    user_id: str | None, conversation_id: str | None, message: str
) -> tuple[str, int]:
    base_context, rows_used = build_v3_context(user_id, conversation_id, message)
    chunks = get_semantic_context(user_id, message)
    if not chunks:
        return base_context, rows_used

    rag_context = "SEMANTIC KNOWLEDGE RETRIEVAL:\n" + "\n\n".join(
        chunk.to_prompt_line() for chunk in chunks
    )
    combined = "\n\n".join(part for part in (rag_context, base_context) if part)
    return combined, rows_used + len(chunks)
