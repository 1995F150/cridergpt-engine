# CriderGPT tokenizer and model training

This directory adds the first model-development pipeline to CriderGPT Engine.
It is intentionally isolated from the production FastAPI dependencies so the
server does not need PyTorch/Transformers just to serve Ollama-backed traffic.

## What is included

- `prepare_dataset.py` — converts owned/licensed text or JSONL into deterministic train/validation JSONL splits.
- `train_tokenizer.py` — trains a byte-level BPE tokenizer and exports a Hugging Face-compatible tokenizer directory.
- `train_causal_lm.py` — either trains a small decoder-only transformer from scratch or fine-tunes an existing causal LM.
- `../requirements-training.txt` — optional training-only dependencies.

This is a starter training stack, not a claim that the resulting model will be
competitive with large commercial foundation models. Model quality depends on
data quality/quantity, architecture size, compute, evaluation, and training time.

## 1. Create a training environment

Do this on a development/training machine, not necessarily the production API host.

```bash
python3 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

For CUDA, install the PyTorch build appropriate for the machine/GPU before or
after this step according to PyTorch's platform instructions.

## 2. Prepare data

Plain text:

```bash
python -m training.prepare_dataset \
  data/source/my_corpus.txt \
  --output-dir data/training
```

JSONL input defaults to the `text` field:

```json
{"text":"Example training document."}
```

Only train on data CriderGPT owns, has permission to use, or is otherwise
properly licensed for model training.

## 3. Train the tokenizer

```bash
python -m training.train_tokenizer \
  data/training/train.jsonl \
  --output-dir artifacts/tokenizer \
  --vocab-size 32000
```

The output contains `tokenizer.json` plus Hugging Face tokenizer metadata.

## 4A. Train a small model from scratch

The default architecture is deliberately modest (8 layers, 512 hidden width,
8 attention heads) so the pipeline can be validated before scaling up.

```bash
python -m training.train_causal_lm \
  --train-file data/training/train.jsonl \
  --validation-file data/training/validation.jsonl \
  --tokenizer artifacts/tokenizer \
  --output-dir artifacts/cridergpt-small \
  --block-size 512 \
  --epochs 1
```

Scale `--layers`, `--heads`, and `--hidden-size` only after measuring memory,
throughput, loss, and evaluation quality.

## 4B. Fine-tune an existing model

```bash
python -m training.train_causal_lm \
  --train-file data/training/train.jsonl \
  --validation-file data/training/validation.jsonl \
  --tokenizer artifacts/tokenizer \
  --base-model /path/to/compatible/base-model \
  --output-dir artifacts/cridergpt-finetuned \
  --learning-rate 2e-5
```

A custom tokenizer can change vocabulary size. The script resizes the base
model's embedding table, but a newly trained tokenizer is generally most
appropriate for from-scratch training. For ordinary fine-tuning, prefer the
base model's original tokenizer unless there is a measured reason to replace it.

## Next engineering steps

Before calling this production-ready model training, add:

1. dataset provenance/version manifests and deduplication,
2. tokenizer fertility/coverage evaluation,
3. held-out perplexity plus task-specific evals,
4. checkpoint/model registry with reproducible config snapshots,
5. distributed training (FSDP/DeepSpeed) for larger models,
6. LoRA/QLoRA for efficient fine-tuning,
7. instruction/chat formatting and supervised fine-tuning,
8. safety/quality regression suites,
9. export/quantization and an Ollama/llama.cpp serving path,
10. GPU telemetry and cost/throughput benchmarks.
