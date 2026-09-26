"""Train a CriderGPT byte-level BPE tokenizer from owned/licensed text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]


def iter_corpus(paths: list[Path], text_field: str) -> Iterable[str]:
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".text"}:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if text:
                        yield text
            continue

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
            continue

        raise ValueError(f"Unsupported input type: {path}")


def build_tokenizer(vocab_size: int, min_frequency: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer._crider_trainer = trainer  # type: ignore[attr-defined]
    return tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help=".txt or .jsonl corpora")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.vocab_size < len(SPECIAL_TOKENS) + 256:
        raise SystemExit("--vocab-size is too small for a useful byte-level tokenizer")
    if args.min_frequency < 1:
        raise SystemExit("--min-frequency must be at least 1")

    tokenizer = build_tokenizer(args.vocab_size, args.min_frequency)
    trainer = tokenizer._crider_trainer  # type: ignore[attr-defined]
    tokenizer.train_from_iterator(
        iter_corpus(args.inputs, args.text_field),
        trainer=trainer,
        length=None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = args.output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        model_max_length=4096,
    )
    hf_tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "type": "byte-level-bpe",
        "vocab_size": len(hf_tokenizer),
        "special_tokens": SPECIAL_TOKENS,
        "source_files": [str(path) for path in args.inputs],
    }
    (args.output_dir / "cridergpt_tokenizer.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Tokenizer saved to {args.output_dir.resolve()}")
    print(f"Vocabulary size: {len(hf_tokenizer)}")


if __name__ == "__main__":
    main()
