"""Validate CriderGPT 2.1 Nova JSONL training datasets.

Checks message schema, empty content, exact/near duplicate prompts, suspiciously
short answers, role order, and simple answer-repetition signals. This tool does
not rewrite training data; it reports problems so a bad dataset is not silently
accepted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def repeated_phrase_score(text: str, size: int = 4) -> float:
    tokens = words(text)
    if len(tokens) < size * 2:
        return 0.0
    grams = [tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def validate_file(path: Path) -> tuple[int, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    prompts: dict[str, int] = {}
    records = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue

            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                errors.append(f"{path}:{line_no}: expected exactly two messages")
                continue

            expected = ("user", "assistant")
            valid = True
            for index, role in enumerate(expected):
                message = messages[index]
                if not isinstance(message, dict):
                    errors.append(f"{path}:{line_no}: message {index + 1} is not an object")
                    valid = False
                    continue
                if message.get("role") != role:
                    errors.append(
                        f"{path}:{line_no}: expected role {role!r} at position {index + 1}"
                    )
                    valid = False
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    errors.append(
                        f"{path}:{line_no}: {role} content must be a non-empty string"
                    )
                    valid = False
            if not valid:
                continue

            prompt = messages[0]["content"].strip()
            answer = messages[1]["content"].strip()
            key = normalize(prompt)
            if key in prompts:
                errors.append(
                    f"{path}:{line_no}: duplicate prompt; first seen on line {prompts[key]}"
                )
            else:
                prompts[key] = line_no

            if len(words(answer)) < 3:
                warnings.append(f"{path}:{line_no}: unusually short answer")
            if repeated_phrase_score(answer) >= 0.20:
                warnings.append(f"{path}:{line_no}: answer may contain repetitive phrasing")
            if "<|user|>" in answer or "<|assistant|>" in answer:
                errors.append(f"{path}:{line_no}: leaked serialized role marker in answer")

    return records, errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSONL files to validate; defaults to training_data/cridergpt21/*.jsonl",
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args()

    paths = args.paths or sorted(
        (Path(__file__).resolve().parents[1] / "training_data" / "cridergpt21").glob("*.jsonl")
    )
    if not paths:
        print("No Nova JSONL datasets found.", file=sys.stderr)
        return 2

    total = 0
    all_errors: list[str] = []
    all_warnings: list[str] = []
    for path in paths:
        if not path.exists():
            all_errors.append(f"{path}: file does not exist")
            continue
        count, errors, warnings = validate_file(path)
        total += count
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        print(f"{path}: {count} records, {len(errors)} errors, {len(warnings)} warnings")

    for item in all_errors:
        print(f"ERROR: {item}", file=sys.stderr)
    for item in all_warnings:
        print(f"WARNING: {item}", file=sys.stderr)

    print(
        f"Nova validation summary: {total} records, "
        f"{len(all_errors)} errors, {len(all_warnings)} warnings"
    )
    if all_errors or (args.strict and all_warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
