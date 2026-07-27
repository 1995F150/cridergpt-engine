#!/usr/bin/env python3
"""Import Layer 3 memory facts from a JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory.structured_memory import upsert_memory_fact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id")
    parser.add_argument("json_file", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.json_file.read_text(encoding="utf-8"))
    facts = payload.get("facts", payload) if isinstance(payload, dict) else payload
    if not isinstance(facts, list):
        raise SystemExit("JSON must contain a list or a top-level 'facts' list")

    imported = 0
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        saved = upsert_memory_fact(args.user_id, fact)
        imported += 1
        print(f"Imported {saved.subject} {saved.predicate}")
    print(f"Imported {imported} structured memory facts")


if __name__ == "__main__":
    main()
