#!/usr/bin/env python3
"""CriderGPT Image Stage 2 dataset preparation.

Input manifest: JSONL, one record per line:
{"image":"relative/or/absolute/path.png","caption":"description","source":"...","license":"..."}

The pipeline validates provenance, images, captions, hashes exact duplicates,
creates deterministic train/validation splits, and writes a reproducible JSONL
manifest. It never downloads training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required: python -m pip install Pillow") from exc

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_record(raw: dict[str, Any], base: Path, min_size: int) -> tuple[dict[str, Any] | None, str | None]:
    for key in ("image", "caption", "source", "license"):
        if not str(raw.get(key, "")).strip():
            return None, f"missing_{key}"

    image_path = Path(raw["image"])
    if not image_path.is_absolute():
        image_path = (base / image_path).resolve()
    if not image_path.is_file():
        return None, "missing_image"

    caption = " ".join(str(raw["caption"]).split())
    if len(caption) < 3:
        return None, "caption_too_short"

    try:
        with Image.open(image_path) as im:
            im.verify()
        with Image.open(image_path) as im:
            width, height = im.size
            fmt = (im.format or "").upper()
    except Exception:
        return None, "invalid_image"

    if fmt not in ALLOWED_FORMATS:
        return None, "unsupported_format"
    if min(width, height) < min_size:
        return None, "image_too_small"

    digest = sha256_file(image_path)
    return {
        "image": str(image_path),
        "caption": caption,
        "source": str(raw["source"]).strip(),
        "license": str(raw["license"]).strip(),
        "sha256": digest,
        "width": width,
        "height": height,
        "format": fmt,
    }, None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, default=Path("dataset_manifest.jsonl"))
    p.add_argument("--report", type=Path, default=Path("dataset_report.json"))
    p.add_argument("--validation-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=427)
    p.add_argument("--min-size", type=int, default=256)
    args = p.parse_args()

    if not 0 < args.validation_ratio < 1:
        p.error("--validation-ratio must be between 0 and 1")
    if not args.manifest.is_file():
        p.error(f"manifest not found: {args.manifest}")

    accepted: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    seen_hashes: set[str] = set()

    with args.manifest.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                rejected["invalid_json"] = rejected.get("invalid_json", 0) + 1
                continue
            record, reason = validate_record(raw, args.manifest.parent, args.min_size)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            assert record is not None
            if record["sha256"] in seen_hashes:
                rejected["exact_duplicate"] = rejected.get("exact_duplicate", 0) + 1
                continue
            seen_hashes.add(record["sha256"])
            record["source_line"] = line_no
            accepted.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    val_count = max(1, round(len(accepted) * args.validation_ratio)) if len(accepted) > 1 else 0
    for index, record in enumerate(accepted):
        record["split"] = "validation" if index < val_count else "train"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in accepted:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "stage": 2,
        "accepted": len(accepted),
        "train": sum(r["split"] == "train" for r in accepted),
        "validation": sum(r["split"] == "validation" for r in accepted),
        "rejected": rejected,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "min_size": args.min_size,
        "manifest_sha256": sha256_file(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
