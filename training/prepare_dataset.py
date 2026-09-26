"""Prepare plain-text or JSONL corpora for CriderGPT model training.

This utility intentionally uses only the Python standard library so dataset
preparation can run before installing the heavier training dependencies.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def iter_text_records(path: Path, text_field: str) -> Iterable[str]:
    """Yield non-empty training strings from .txt or .jsonl input."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".text"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield text
        return

    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc

                value = row.get(text_field)
                if not isinstance(value, str):
                    raise ValueError(
                        f"{path}:{line_number}: field {text_field!r} must be a string"
                    )
                text = value.strip()
                if text:
                    yield text
        return

    raise ValueError(f"Unsupported input type: {path}. Use .txt or .jsonl.")


def write_jsonl(path: Path, records: Iterable[str]) -> int:
    """Write strings as {"text": ...} JSONL and return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for text in records:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help=".txt or .jsonl files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.02,
        help="Fraction held out for validation (default: 0.02)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not 0 <= args.validation_ratio < 1:
        raise SystemExit("--validation-ratio must be >= 0 and < 1")

    records: list[str] = []
    for path in args.inputs:
        records.extend(iter_text_records(path, args.text_field))

    if not records:
        raise SystemExit("No usable training text was found.")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    validation_count = int(len(records) * args.validation_ratio)
    if args.validation_ratio > 0 and validation_count == 0 and len(records) > 1:
        validation_count = 1

    validation = records[:validation_count]
    train = records[validation_count:]

    train_count = write_jsonl(args.output_dir / "train.jsonl", train)
    val_count = write_jsonl(args.output_dir / "validation.jsonl", validation)

    manifest = {
        "inputs": [str(path) for path in args.inputs],
        "train_records": train_count,
        "validation_records": val_count,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {train_count} training records and {val_count} validation records")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
