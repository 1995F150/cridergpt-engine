# OASST1 → CriderGPT training data

This directory contains the preprocessing pipeline for the OpenAssistant OASST1
conversation dataset.

## Why Parquet?

OASST1 is distributed as Parquet, a compact columnar dataset format. The source
files are read with `pandas` + `pyarrow` and are never modified in place.

## Keep raw datasets out of normal Git history

Do **not** commit the large source `.parquet` files into this repository.
Store them on the training machine (or another dataset store) and point this
script at their paths.

Example layout on the server:

```text
/opt/cridergpt-datasets/oasst1/
├── train-00000-of-00001-b42a775f407cee45.parquet
└── validation-00000-of-00001-134b8fd0c89408b6.parquet
```

## Install the converter

```bash
cd /opt/cridergpt-engine
python3 -m venv .venv-training
.venv-training/bin/pip install -r training_data/oasst1/requirements.txt
```

## Convert OASST1

```bash
.venv-training/bin/python training_data/oasst1/prepare_oasst1.py \
  --train /opt/cridergpt-datasets/oasst1/train-00000-of-00001-b42a775f407cee45.parquet \
  --validation /opt/cridergpt-datasets/oasst1/validation-00000-of-00001-134b8fd0c89408b6.parquet \
  --output-dir data/training/oasst1 \
  --language en
```

The output is:

```text
data/training/oasst1/
├── train_conversations.jsonl
├── validation_conversations.jsonl
└── prepare_report.json
```

Each JSONL row uses a neutral conversation representation:

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ],
  "source": "OpenAssistant/oasst1",
  "language": "en",
  "conversation_sha256": "..."
}
```

## What the converter validates

- required OASST1 columns exist
- selected language only (English by default)
- parent/child conversation links
- strict user/assistant role alternation
- non-empty text
- conversation ends with an assistant response
- duplicate conversation paths are removed
- SHA-256 hashes are recorded for the raw source files

## Important

This step prepares conversations only. It intentionally does **not** replace,
retrain, or alter the existing CriderGPT tokenizer. The next training step must
serialize these messages with the tokenizer/special-token contract used by the
actual CriderGPT Native text checkpoint.
