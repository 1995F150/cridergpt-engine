#!/usr/bin/env python3
"""Prepare OpenAssistant OASST1 Parquet files for CriderGPT Native training.

This tool never modifies the source Parquet files. It:
- reads train/validation parquet
- filters to English by default
- reconstructs root-to-leaf assistant conversations
- validates role alternation/content
- removes exact duplicate conversation paths
- writes JSONL in a simple messages[] format
- records source hashes and processing statistics

Raw datasets should remain outside normal Git history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {"message_id", "parent_id", "role", "text", "lang"}
ROLE_MAP = {"prompter": "user", "assistant": "assistant"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r", "\n").split())


@dataclass
class Stats:
    raw_rows: int = 0
    language_rows: int = 0
    roots: int = 0
    leaves: int = 0
    emitted: int = 0
    duplicate_paths: int = 0
    skipped_broken_parent: int = 0
    skipped_invalid_roles: int = 0
    skipped_empty_text: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def reconstruct(frame: pd.DataFrame, language: str) -> tuple[list[dict], Stats]:
    stats = Stats(raw_rows=len(frame))
    frame = frame[frame["lang"].astype(str).eq(language)].copy()
    stats.language_rows = len(frame)

    records: dict[str, dict] = {}
    children: dict[str, list[str]] = {}

    for row in frame.to_dict(orient="records"):
        message_id = str(row["message_id"])
        records[message_id] = row

    for message_id, row in records.items():
        parent = row.get("parent_id")
        if parent is None or (isinstance(parent, float) and pd.isna(parent)):
            continue
        parent_id = str(parent)
        if parent_id in records:
            children.setdefault(parent_id, []).append(message_id)

    roots = [
        mid for mid, row in records.items()
        if row.get("parent_id") is None
        or (isinstance(row.get("parent_id"), float) and pd.isna(row.get("parent_id")))
        or str(row.get("parent_id")) not in records
    ]
    stats.roots = len(roots)
    leaves = [mid for mid in records if mid not in children]
    stats.leaves = len(leaves)

    conversations: list[dict] = []
    fingerprints: set[str] = set()

    for leaf in leaves:
        chain: list[dict] = []
        current = leaf
        seen: set[str] = set()
        broken = False

        while current in records and current not in seen:
            seen.add(current)
            row = records[current]
            chain.append(row)
            parent = row.get("parent_id")
            if parent is None or (isinstance(parent, float) and pd.isna(parent)):
                break
            parent_id = str(parent)
            if parent_id not in records:
                broken = True
                break
            current = parent_id

        if broken:
            stats.skipped_broken_parent += 1
            continue

        chain.reverse()
        messages: list[dict[str, str]] = []
        invalid_role = False
        empty = False

        for row in chain:
            role = ROLE_MAP.get(str(row.get("role")))
            if role is None:
                invalid_role = True
                break
            content = normalize_text(row.get("text"))
            if not content:
                empty = True
                break
            messages.append({"role": role, "content": content})

        if invalid_role:
            stats.skipped_invalid_roles += 1
            continue
        if empty:
            stats.skipped_empty_text += 1
            continue
        if len(messages) < 2 or messages[0]["role"] != "user":
            stats.skipped_invalid_roles += 1
            continue

        # Require strict user/assistant alternation.
        expected = "user"
        valid = True
        for msg in messages:
            if msg["role"] != expected:
                valid = False
                break
            expected = "assistant" if expected == "user" else "user"
        if not valid or messages[-1]["role"] != "assistant":
            stats.skipped_invalid_roles += 1
            continue

        canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        fp = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if fp in fingerprints:
            stats.duplicate_paths += 1
            continue
        fingerprints.add(fp)

        conversations.append({
            "messages": messages,
            "source": "OpenAssistant/oasst1",
            "language": language,
            "conversation_sha256": fp,
        })

    stats.emitted = len(conversations)
    return conversations, stats


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_one(source: Path, output: Path, language: str) -> dict:
    frame = load_parquet(source)
    rows, stats = reconstruct(frame, language)
    write_jsonl(output, rows)
    return {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "output_path": str(output),
        "output_records": len(rows),
        "stats": stats.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare OASST1 for CriderGPT training")
    parser.add_argument("--train", type=Path, required=True, help="OASST1 train parquet")
    parser.add_argument("--validation", type=Path, required=True, help="OASST1 validation parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/oasst1"))
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "train_conversations.jsonl"
    val_out = args.output_dir / "validation_conversations.jsonl"

    report = {
        "dataset": "OpenAssistant/oasst1",
        "language": args.language,
        "format": "messages-jsonl",
        "train": prepare_one(args.train, train_out, args.language),
        "validation": prepare_one(args.validation, val_out, args.language),
    }
    report_path = args.output_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
