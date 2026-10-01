"""One-command builder for the CriderGPT Native ~4.27M parameter text model.

This orchestrates the existing CriderGPT training utilities:
1. prepare owned/licensed text/JSONL into train/validation splits
2. train the CriderGPT byte-level BPE tokenizer
3. train a decoder-only GPT-style language model from scratch
4. save the Hugging Face-compatible checkpoint to model/checkpoint

The architecture defaults are intentionally chosen to match the original
CriderGPT Native target: 4 layers, 256 hidden size, 4 heads, 4096 vocabulary,
and a 256-token context window (~4.27M trainable parameters).

This script does NOT download another model and does NOT use a cloud AI API.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = ROOT / "artifacts" / "native_build"
DEFAULT_CHECKPOINT = ROOT / "model" / "checkpoint"


def run(cmd: list[str]) -> None:
    print("\n>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Owned/licensed .txt or .jsonl training files.",
    )
    p.add_argument("--text-field", default="text")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--validation-ratio", type=float, default=0.05)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="Allow replacing model/checkpoint after training succeeds.",
    )
    return p


def main() -> int:
    args = parser().parse_args()

    missing = [str(p) for p in args.inputs if not p.exists()]
    if missing:
        raise SystemExit("Training input(s) not found: " + ", ".join(missing))
    if args.hidden_size % args.heads:
        raise SystemExit("--hidden-size must be divisible by --heads")
    if args.checkpoint_dir.exists() and any(args.checkpoint_dir.iterdir()) and not args.overwrite_checkpoint:
        raise SystemExit(
            f"{args.checkpoint_dir} already contains files. "
            "Use --overwrite-checkpoint only if you intentionally want to replace it."
        )

    prepared = args.work_dir / "prepared"
    tokenizer = args.work_dir / "tokenizer"
    trained = args.work_dir / "trained"

    # Never silently reuse stale generated artifacts.
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    run([
        python, "-m", "training.prepare_dataset",
        *[str(p) for p in args.inputs],
        "--output-dir", str(prepared),
        "--text-field", args.text_field,
        "--validation-ratio", str(args.validation_ratio),
        "--seed", str(args.seed),
    ])

    run([
        python, "-m", "training.train_tokenizer",
        str(prepared / "train.jsonl"),
        "--output-dir", str(tokenizer),
        "--vocab-size", str(args.vocab_size),
        "--min-frequency", "2",
    ])

    train_cmd = [
        python, "-m", "training.train_causal_lm",
        "--train-file", str(prepared / "train.jsonl"),
        "--validation-file", str(prepared / "validation.jsonl"),
        "--tokenizer", str(tokenizer),
        "--output-dir", str(trained),
        "--block-size", str(args.context),
        "--layers", str(args.layers),
        "--heads", str(args.heads),
        "--hidden-size", str(args.hidden_size),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation", str(args.gradient_accumulation),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
    ]
    run(train_cmd)

    # Verify that the actual trained artifact can be loaded locally before deployment.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(trained), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(trained), local_files_only=True)
    parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.checkpoint_dir.exists():
        shutil.rmtree(args.checkpoint_dir)
    shutil.copytree(trained, args.checkpoint_dir)

    manifest = {
        "name": "CriderGPT Native",
        "architecture": {
            "type": "decoder-only GPT-style transformer",
            "layers": args.layers,
            "hidden_size": args.hidden_size,
            "attention_heads": args.heads,
            "vocab_size": len(tok),
            "context_length": args.context,
        },
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "inputs": [str(p.resolve()) for p in args.inputs],
        },
        "checkpoint": str(args.checkpoint_dir.resolve()),
    }
    (args.checkpoint_dir / "cridergpt_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nCriderGPT Native build complete.")
    print(f"Parameters: {parameters:,}")
    print(f"Tokenizer vocabulary: {len(tok):,}")
    print(f"Checkpoint: {args.checkpoint_dir.resolve()}")
    print("\nRun it with:")
    print(f'  "{python}" cridergpt_stage5\\inference.py --chat')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
