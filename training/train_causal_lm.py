"""Train or fine-tune a decoder-only CriderGPT language model.

Two modes are supported:
1. From scratch: initialize a small GPT-style transformer using the CriderGPT tokenizer.
2. Fine-tune: load an existing causal LM and continue training on CriderGPT data.

This is a development utility, not part of the production FastAPI process.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument(
        "--base-model",
        help="Optional Hugging Face model name/path. Omit to train from scratch.",
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # From-scratch architecture controls.
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=512)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str)
    return parser


def load_training_data(train_file: Path, validation_file: Path | None):
    files = {"train": str(train_file)}
    if validation_file:
        files["validation"] = str(validation_file)

    def kind(path: Path) -> str:
        return "json" if path.suffix.lower() in {".json", ".jsonl"} else "text"

    train_kind = kind(train_file)
    if validation_file and kind(validation_file) != train_kind:
        raise ValueError("Train and validation files must use the same file format")

    return load_dataset(train_kind, data_files=files)


def build_model(args, tokenizer):
    if args.base_model:
        model = AutoModelForCausalLM.from_pretrained(args.base_model)
        model.resize_token_embeddings(len(tokenizer))
        return model

    if args.hidden_size % args.heads != 0:
        raise ValueError("--hidden-size must be divisible by --heads")

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.hidden_size,
        n_layer=args.layers,
        n_head=args.heads,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return GPT2LMHeadModel(config)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    if args.fp16 and args.bf16:
        raise SystemExit("Choose only one of --fp16 or --bf16")
    if args.block_size < 32:
        raise SystemExit("--block-size must be at least 32")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_training_data(args.train_file, args.validation_file)

    text_column = "text"
    if text_column not in raw["train"].column_names:
        raise SystemExit("Training data must contain a 'text' column")

    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=args.block_size,
            add_special_tokens=True,
        )

    remove_columns = raw["train"].column_names
    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing",
    )

    model = build_model(args, tokenizer)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    has_validation = "validation" in tokenized
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if has_validation else "no",
        eval_steps=args.save_steps if has_validation else None,
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate() if has_validation else {}
    trainer.save_metrics("eval", metrics)

    print(f"Model saved to {args.output_dir.resolve()}")
    if metrics:
        print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
