"""Layer 3 context wrapper.

Keeps the existing Layer 1/2 context builder stable while placing structured,
current facts ahead of legacy free-form memory.
"""

from __future__ import annotations

from memory.memory_loader import build_context as build_legacy_context
from memory.structured_memory import get_structured_memories


def build_context(
    user_id: str | None, conversation_id: str | None, message: str
) -> tuple[str, int]:
    legacy_context, legacy_rows = build_legacy_context(user_id, conversation_id, message)
    facts = get_structured_memories(user_id, message)
    sections: list[str] = []
    if facts:
        sections.append(
            "CURRENT STRUCTURED MEMORY (prefer these dated facts over conflicting legacy memory):\n"
            + "\n".join(fact.to_prompt_line() for fact in facts)
        )
    if legacy_context:
        sections.append(legacy_context)
    return "\n\n".join(sections), legacy_rows + len(facts)
