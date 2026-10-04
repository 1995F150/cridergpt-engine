"""Build or continue CriderGPT 2.0 while preserving prior checkpoints.

This builder uses rehearsal/replay: every continued-training run mixes the broad
OASST1 corpus with CriderGPT identity, behavior, optional real conversation
exports, writing samples, and optional founder memory. That reduces catastrophic
forgetting compared with training only on the newest data.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "training_data" / "cridergpt2"
DEFAULT_OASST1 = ROOT / "data" / "training" / "oasst1" / "train_conversations.jsonl"
DEFAULT_LOCAL = ROOT / "data" / "training" / "cridergpt2"
V1_CHECKPOINT = ROOT / "model" / "checkpoint"
DEFAULT_OUTPUT = ROOT / "model" / "cridergpt-2.0" / "checkpoint"
DEFAULT_HISTORY = ROOT / "model" / "cridergpt-2.0" / "history"
DEFAULT_WORK = ROOT / "artifacts" / "cridergpt2_build"


def run(cmd: list[str]) -> None:
    print("\n>", " ".join(map(str, cmd)), flush=True)
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


def load_records(source: Path) -> list[str]:
    if not source.exists():
        return []
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
    return records


def append_records(records: list[str], dest, repeat: int = 1) -> int:
    if repeat < 0:
        raise ValueError("source weights cannot be negative")
    for _ in range(repeat):
        for text in records:
            dest.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return len(records) * repeat


def append_jsonl(source: Path, dest, repeat: int = 1) -> int:
    """Append a weighted JSONL source while preserving the original public helper."""
    return append_records(load_records(source), dest, repeat)


def has_checkpoint(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def validate_replacement(base: Path, output: Path, overwrite: bool) -> None:
    """Require an explicit opt-in when replacing 2.0 from another checkpoint."""
    if has_checkpoint(output) and base.resolve() != output.resolve() and not overwrite:
        raise SystemExit(
            f"{output} already contains a checkpoint. Use --overwrite-2.0 only "
            "if replacement is intentional."
        )


def backup_checkpoint(path: Path, history_dir: Path) -> Path | None:
    if not path.exists() or not any(path.iterdir()):
        return None
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = history_dir / f"checkpoint-{stamp}"
    shutil.copytree(path, backup)
    return backup


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oasst1", type=Path, default=DEFAULT_OASST1)
    p.add_argument(
        "--conversation-data",
        type=Path,
        default=DEFAULT_LOCAL / "conversations.jsonl",
        help="Clean local conversation JSONL produced by export_cridergpt2_dataset.py",
    )
    p.add_argument("--conversation-weight", type=int, default=1)
    p.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Explicit starting checkpoint; default is existing 2.0, otherwise 1.0",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    p.add_argument("--identity-weight", type=int, default=100)
    p.add_argument("--behavior-weight", type=int, default=25)
    p.add_argument("--writing-samples", type=Path, default=DEFAULT_LOCAL / "writing_samples.jsonl")
    p.add_argument("--writing-weight", type=int, default=2)
    p.add_argument("--founder-memory", type=Path, default=DEFAULT_LOCAL / "founder_memory.jsonl")
    p.add_argument("--founder-memory-weight", type=int, default=2)
    p.add_argument("--include-founder-memory", action="store_true")
    p.add_argument("--validation-ratio", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument(
        "--overwrite-2.0",
        action="store_true",
        help="Allow replacing the current 2.0 checkpoint when starting from a different base",
    )
    args = p.parse_args()

    base = args.base_model or (
        DEFAULT_OUTPUT if has_checkpoint(DEFAULT_OUTPUT) else V1_CHECKPOINT
    )
    if not base.exists():
        raise SystemExit(f"Base checkpoint not found: {base}")
    if not args.oasst1.exists():
        raise SystemExit(f"Prepared OASST1 data not found: {args.oasst1}")
    validate_replacement(base, args.output_dir, args.overwrite_2_0)

    identity_records = load_records(SEED_DIR / "identity.jsonl")
    if not identity_records:
        raise SystemExit("Identity dataset is empty; refusing to train.")

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    sources = {
        "oasst1": (load_records(args.oasst1), 1),
        "conversations": (load_records(args.conversation_data), args.conversation_weight),
        "identity": (identity_records, args.identity_weight),
        "behavior": (load_records(SEED_DIR / "behavior.jsonl"), args.behavior_weight),
        "writing_samples": (load_records(args.writing_samples), args.writing_weight),
        "founder_memory": (
            load_records(args.founder_memory) if args.include_founder_memory else [],
            args.founder_memory_weight,
        ),
    }

    combined = args.work_dir / "combined.jsonl"
    counts: dict[str, int] = {}
    with combined.open("w", encoding="utf-8") as out:
        for name, (records, weight) in sources.items():
            counts[name] = append_records(records, out, weight) if records else 0

    total = sum(counts.values())
    print(f"Starting checkpoint: {base}")
    print("Combined training rows:")
    for name, count in counts.items():
        pct = (count / total * 100.0) if total else 0.0
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    print(f"  TOTAL: {total:,}")

    if counts["identity"] == 0:
        raise SystemExit("Identity weight must be greater than zero; refusing to train.")
    if total == 0:
        raise SystemExit("No training records found.")

    prepared = args.work_dir / "prepared"
    run([
        sys.executable,
        "-m",
        "training.prepare_dataset",
        str(combined),
        "--output-dir",
        str(prepared),
        "--text-field",
        "text",
        "--validation-ratio",
        str(args.validation_ratio),
        "--seed",
        "42",
    ])

    trained = args.work_dir / "trained"
    run([
        sys.executable,
        "-m",
        "training.train_causal_lm",
        "--train-file",
        str(prepared / "train.jsonl"),
        "--validation-file",
        str(prepared / "validation.jsonl"),
        "--tokenizer",
        str(base),
        "--base-model",
        str(base),
        "--output-dir",
        str(trained),
        "--block-size",
        "256",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation",
        str(args.gradient_accumulation),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        "42",
    ])
    if not has_checkpoint(trained):
        raise SystemExit(f"Training completed without a usable checkpoint: {trained}")

    backup = None
    if has_checkpoint(args.output_dir):
        backup = backup_checkpoint(args.output_dir, args.history_dir)
        print(f"Preserved previous 2.0 checkpoint at: {backup}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.copytree(trained, args.output_dir)

    manifest = {
        "name": "CriderGPT 2.0",
        "family": "CriderGPT Native",
        "version": "2.0.0",
        "continued_from": str(base.resolve()),
        "checkpoint": str(args.output_dir.resolve()),
        "previous_checkpoint_backup": str(backup.resolve()) if backup else None,
        "training_rows": counts,
        "conversation_weight": args.conversation_weight,
        "identity_weight": args.identity_weight,
        "behavior_weight": args.behavior_weight,
        "writing_weight": args.writing_weight,
        "founder_memory_weight": args.founder_memory_weight,
        "includes_private_founder_memory": bool(args.include_founder_memory),
    }
    (args.output_dir / "cridergpt_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nCriderGPT 2.0 checkpoint saved to {args.output_dir}")
    print("CriderGPT 1.0 remains unchanged at model/checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
