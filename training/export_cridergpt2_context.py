"""Export CriderGPT writing samples and optional founder memory for 2.0 training.

The default output lives under data/, which is gitignored. This keeps live/private
context out of normal Git history while still allowing a local training build to
consume it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory.core_profile import get_core_profile
from memory.memory_loader import (
    get_ai_memory,
    get_conversation_history,
    get_profile,
    get_user_preferences,
    get_user_training,
    get_writing_samples,
)
from memory.project_knowledge import get_project_knowledge

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "training" / "cridergpt2"


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--writing-limit", type=int, default=100)
    parser.add_argument("--user-id")
    parser.add_argument(
        "--include-private-memory",
        action="store_true",
        help="Explicitly export scoped ai_memory for --user-id into a local gitignored dataset.",
    )
    parser.add_argument(
        "--include-memory-system",
        action="store_true",
        help="Export a scoped snapshot of the retrievable CriderGPT memory system for --user-id.",
    )
    parser.add_argument("--memory-history-limit", type=int, default=200)
    parser.add_argument("--memory-ai-limit", type=int, default=500)
    args = parser.parse_args()

    samples = get_writing_samples(args.writing_limit)
    sample_rows = []
    for sample in samples:
        content = str(sample.get("content") or "").strip()
        if content:
            sample_rows.append({"text": content, "source": "writing_samples"})
    writing_path = args.output_dir / "writing_samples.jsonl"
    writing_count = write_jsonl(writing_path, sample_rows)
    print(f"Wrote {writing_count} writing samples to {writing_path}")

    if args.include_private_memory:
        if not args.user_id:
            raise SystemExit("--include-private-memory requires --user-id")
        memories = get_ai_memory(args.user_id, limit=500)
        memory_rows = []
        for item in memories:
            content = str(item.get("details") or item.get("content") or "").strip()
            topic = str(item.get("topic") or "").strip()
            category = str(item.get("category") or "general").strip()
            if content:
                text = f"[{category}]"
                if topic:
                    text += f" {topic}:"
                text += f" {content}"
                memory_rows.append({"text": text, "source": "ai_memory"})
        memory_path = args.output_dir / "founder_memory.jsonl"
        memory_count = write_jsonl(memory_path, memory_rows)
        print(f"Wrote {memory_count} private memory rows to {memory_path}")
        print("Keep this file local; data/ is gitignored.")

    if args.include_memory_system:
        if not args.user_id:
            raise SystemExit("--include-memory-system requires --user-id")

        memory_rows: list[dict] = []

        core = get_core_profile(args.user_id)
        if core:
            memory_rows.append(
                {
                    "text": core.to_prompt_section(),
                    "source": "core_profile",
                    "memory_layer": "core_profile",
                }
            )

        projects = get_project_knowledge(args.user_id, "", limit=8)
        for item in projects:
            memory_rows.append(
                {
                    "text": item.to_prompt_line(),
                    "source": "project_knowledge",
                    "memory_layer": "project_knowledge",
                }
            )

        for item in get_ai_memory(args.user_id, limit=args.memory_ai_limit):
            content = str(item.get("details") or item.get("content") or "").strip()
            topic = str(item.get("topic") or "").strip()
            category = str(item.get("category") or "general").strip()
            if content:
                text = f"[{category}]"
                if topic:
                    text += f" {topic}:"
                text += f" {content}"
                memory_rows.append(
                    {
                        "text": text,
                        "source": "ai_memory",
                        "memory_layer": "long_term_memory",
                    }
                )

        for item in get_user_preferences(args.user_id, limit=100):
            ptype = str(item.get("preference_type") or "").strip()
            pvalue = str(item.get("preference_value") or "").strip()
            if ptype or pvalue:
                memory_rows.append(
                    {
                        "text": f"User preference: {ptype}: {pvalue}",
                        "source": "user_preferences",
                        "memory_layer": "preferences",
                    }
                )

        legacy = get_profile(args.user_id)
        if legacy:
            pairs = [
                f"{key}: {value}"
                for key, value in legacy.items()
                if value not in (None, "")
            ]
            if pairs:
                memory_rows.append(
                    {
                        "text": "Legacy user profile: " + "; ".join(pairs),
                        "source": "profiles",
                        "memory_layer": "legacy_profile",
                    }
                )

        for item in get_user_training(args.user_id, "", limit=100):
            content = str(item.get("content") or "").strip()
            if content:
                memory_rows.append(
                    {
                        "text": content,
                        "source": "training_inputs",
                        "memory_layer": "user_training",
                    }
                )

        history = get_conversation_history(
            args.user_id, None, limit=max(1, args.memory_history_limit)
        )
        for item in history:
            role = str(item.get("role") or "user").strip()
            content = str(item.get("content") or "").strip()
            if content:
                memory_rows.append(
                    {
                        "text": f"{role}: {content}",
                        "source": "chat_messages",
                        "memory_layer": "short_term_history",
                    }
                )

        memory_path = args.output_dir / "memory_system.jsonl"
        memory_count = write_jsonl(memory_path, memory_rows)
        print(f"Wrote {memory_count} memory-system rows to {memory_path}")
        print("Keep this file local; data/ is gitignored.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
