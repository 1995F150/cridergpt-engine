"""Build CriderGPT 2.0 without overwriting the CriderGPT 1.0 checkpoint.

By default this continues training from model/checkpoint (CriderGPT 1.0), reuses
its tokenizer, combines prepared OASST1 conversations with curated CriderGPT
identity/behavior data, and optionally includes locally exported writing samples
and founder memory. The 2.0 checkpoint is written separately to
model/cridergpt-2.0/checkpoint.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "training_data" / "cridergpt2"
DEFAULT_OASST1 = ROOT / "data" / "training" / "oasst1" / "train_conversations.jsonl"
DEFAULT_LOCAL = ROOT / "data" / "training" / "cridergpt2"
DEFAULT_BASE = ROOT / "model" / "checkpoint"
DEFAULT_OUTPUT = ROOT / "model" / "cridergpt-2.0" / "checkpoint"
DEFAULT_WORK = ROOT / "artifacts" / "cridergpt2_build"


def run(cmd: list[str]) -> None:
    print("\n>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def format_row(row: dict) -> str | None:
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    messages = row.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                parts.append(f"<|{role}|>\n{content.strip()}")
        if parts:
            return "\n".join(parts) + "\n<|eos|>"
    return None


def append_jsonl(source: Path, dest, repeat: int = 1) -> int:
    if not source.exists():
        return 0
    records: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{source}:{line_no}: invalid JSON: {exc}") from exc
            text = format_row(row)
            if text:
                records.append(text)
    for _ in range(max(1, repeat)):
        for text in records:
            dest.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return len(records) * max(1, repeat)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oasst1", type=Path, default=DEFAULT_OASST1)
    p.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    p.add_argument("--identity-weight", type=int, default=20)
    p.add_argument("--behavior-weight", type=int, default=10)
    p.add_argument("--writing-samples", type=Path, default=DEFAULT_LOCAL / "writing_samples.jsonl")
    p.add_argument("--founder-memory", type=Path, default=DEFAULT_LOCAL / "founder_memory.jsonl")
    p.add_argument("--include-founder-memory", action="store_true")
    p.add_argument("--validation-ratio", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--overwrite-2.0", action="store_true")
    args = p.parse_args()

    if not args.base_model.exists():
        raise SystemExit(
            f"CriderGPT 1.0 checkpoint not found: {args.base_model}. "
            "2.0 defaults to continuing from the trained 1.0 model."
        )
    if not args.oasst1.exists():
        raise SystemExit(f"Prepared OASST1 data not found: {args.oasst1}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_2_0:
        raise SystemExit(
            f"{args.output_dir} already contains a 2.0 checkpoint. "
            "Use --overwrite-2.0 only if replacement is intentional."
        )

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)
    combined = args.work_dir / "combined.jsonl"

    counts: dict[str, int] = {}
    with combined.open("w", encoding="utf-8") as out:
        counts["oasst1"] = append_jsonl(args.oasst1, out, 1)
        counts["identity"] = append_jsonl(SEED_DIR / "identity.jsonl", out, args.identity_weight)
        counts["behavior"] = append_jsonl(SEED_DIR / "behavior.jsonl", out, args.behavior_weight)
        counts["writing_samples"] = append_jsonl(args.writing_samples, out, 1)
        if args.include_founder_memory:
            counts["founder_memory"] = append_jsonl(args.founder_memory, out, 1)

    print("Combined training rows:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    prepared = args.work_dir / "prepared"
    run([
        sys.executable, "-m", "training.prepare_dataset", str(combined),
        "--output-dir", str(prepared),
        "--text-field", "text",
        "--validation-ratio", str(args.validation_ratio),
        "--seed", "42",
    ])

    trained = args.work_dir / "trained"
    run([
        sys.executable, "-m", "training.train_causal_lm",
        "--train-file", str(prepared / "train.jsonl"),
        "--validation-file", str(prepared / "validation.jsonl"),
        "--tokenizer", str(args.base_model),
        "--base-model", str(args.base_model),
        "--output-dir", str(trained),
        "--block-size", "256",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation", str(args.gradient_accumulation),
        "--learning-rate", str(args.learning_rate),
        "--seed", "42",
    ])

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.copytree(trained, args.output_dir)

    manifest = {
        "name": "CriderGPT 2.0",
        "family": "CriderGPT Native",
        "version": "2.0.0",
        "base_checkpoint": str(args.base_model.resolve()),
        "checkpoint": str(args.output_dir.resolve()),
        "training_rows": counts,
        "identity_weight": args.identity_weight,
        "behavior_weight": args.behavior_weight,
        "includes_private_founder_memory": bool(args.include_founder_memory),
    }
    (args.output_dir / "cridergpt_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nCriderGPT 2.0 checkpoint saved to {args.output_dir}")
    print("CriderGPT 1.0 remains unchanged at model/checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
