"""Build a cleaned CriderGPT-native v0.1 corpus from CSV exports.

The builder deliberately excludes personal/private categories by default,
deduplicates normalized text, and emits deterministic train/validation JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd


def normalize(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


def add_record(records: list[dict], text: str, source: str, kind: str) -> None:
    text = normalize(text)
    if len(text) < 20:
        return
    records.append({"text": text, "source": source, "kind": kind, "hash": fingerprint(text)})


def build(args: argparse.Namespace) -> dict:
    records: list[dict] = []

    writing = pd.read_csv(args.writing_samples).fillna("")
    for _, row in writing.iterrows():
        title = normalize(row.get("title"))
        category = normalize(row.get("category")).lower()
        if not args.include_personal and (
            category in {"personal", "text message"}
            or any(word in title.lower() for word in ("paisley", "savanaa", "life story"))
        ):
            continue
        add_record(records, f"Title: {title}\n\n{normalize(row.get('content'))}", "writing_samples", "style")

    memory = pd.read_csv(args.ai_memory).fillna("")
    for _, row in memory.iterrows():
        question = normalize(row.get("topic"))
        answer = normalize(row.get("details")) or normalize(row.get("content"))
        if question and answer:
            add_record(records, f"User: {question}\nAssistant: {answer}", "ai_memory", "instruction")

    training = pd.read_csv(args.training_data).fillna("")
    for _, row in training.iterrows():
        category = normalize(row.get("category")).lower()
        name = normalize(row.get("dataset_name"))
        if not args.include_personal and (category == "personal" or "life story" in name.lower()):
            continue
        add_record(
            records,
            f"Dataset: {name}\nCategory: {category}\n\n{normalize(row.get('content'))}",
            "training_data",
            "knowledge",
        )

    corpus = pd.read_csv(args.training_corpus).fillna("")
    for _, row in corpus.iterrows():
        topic = normalize(row.get("topic"))
        category = normalize(row.get("category")).lower()
        add_record(
            records,
            f"Topic: {topic}\nCategory: {category}\n\n{normalize(row.get('content'))}",
            "training_corpus",
            "knowledge",
        )

    unique: dict[str, dict] = {}
    for record in records:
        unique.setdefault(record["hash"], record)
    records = list(unique.values())

    rng = random.Random(args.seed)
    rng.shuffle(records)
    validation_count = max(1, round(len(records) * args.validation_ratio)) if len(records) > 1 else 0
    validation = records[:validation_count]
    train = records[validation_count:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "cridergpt-native-v0.1",
        "seed": args.seed,
        "records_total": len(records),
        "train_records": len(train),
        "validation_records": len(validation),
        "validation_ratio": args.validation_ratio,
        "personal_data_included": bool(args.include_personal),
        "deduplication": "sha256(normalized lowercase text)",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--writing-samples", type=Path, required=True)
    p.add_argument("--ai-memory", type=Path, required=True)
    p.add_argument("--training-data", type=Path, required=True)
    p.add_argument("--training-corpus", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/cridergpt-native-v0.1"))
    p.add_argument("--validation-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-personal", action="store_true", help="Explicitly opt in to personal/private rows.")
    return p


def main() -> None:
    args = parser().parse_args()
    if not 0 <= args.validation_ratio < 1:
        raise SystemExit("--validation-ratio must be >= 0 and < 1")
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
