"""Export CriderGPT writing samples and optional founder memory for 2.0 training.

The default output lives under data/, which is gitignored. This keeps live/private
context out of normal Git history while still allowing a local training build to
consume it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory.memory_loader import get_ai_memory, get_writing_samples

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
