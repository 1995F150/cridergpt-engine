# CriderGPT Native Model v0.1

This track trains CriderGPT-owned model weights from scratch. It does **not** download a pretrained language model when `--base-model` is omitted.

## Current goal

v0.1 is an engineering proof-of-concept: prove the complete path from CriderGPT-owned/permitted data -> tokenizer -> initialized transformer -> trained weights -> saved checkpoint -> generated text.

The initial dataset is intentionally small. Do not expect general-purpose assistant quality yet.

## 1. Install training dependencies

```bash
python -m pip install -r requirements-training.txt
python -m pip install pandas
```

## 2. Build the dataset

Place the four CSV exports outside Git (they may contain private information), then run:

```bash
python -m training.build_cridergpt_dataset \
  --writing-samples /path/writing_samples_rows.csv \
  --ai-memory /path/ai_memory_rows.csv \
  --training-data /path/cridergpt_training_data_rows.csv \
  --training-corpus /path/cridergpt_training_corpus_rows.csv \
  --output-dir data/cridergpt-native-v0.1
```

Personal/text-message/life-story rows are excluded by default. Only use `--include-personal` after deliberately reviewing what will be memorized by the model.

## 3. Train the tokenizer

For this small proof-of-concept corpus, start with a small vocabulary instead of 32K:

```bash
python -m training.train_tokenizer \
  data/cridergpt-native-v0.1/train.jsonl \
  --output-dir artifacts/cridergpt-native-v0.1-tokenizer \
  --vocab-size 4096 \
  --min-frequency 1
```

## 4. Train native weights from scratch

Start tiny so the full pipeline can be debugged cheaply:

```bash
python -m training.train_causal_lm \
  --train-file data/cridergpt-native-v0.1/train.jsonl \
  --validation-file data/cridergpt-native-v0.1/validation.jsonl \
  --tokenizer artifacts/cridergpt-native-v0.1-tokenizer \
  --output-dir artifacts/cridergpt-native-v0.1 \
  --block-size 256 \
  --layers 4 \
  --heads 4 \
  --hidden-size 256 \
  --epochs 10 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --learning-rate 0.0005 \
  --save-steps 50 \
  --logging-steps 5
```

There is intentionally no `--base-model` argument in this command. `train_causal_lm.py` therefore initializes a new GPT-style transformer with random weights.

## 5. Generate text

```bash
python -m training.generate_native \
  --model artifacts/cridergpt-native-v0.1 \
  --prompt "User: What is CriderGPT?\nAssistant:"
```

## v0.1 exit criteria

- dataset builder runs reproducibly
- tokenizer saves and reloads
- model initializes from random weights
- training loss is recorded
- checkpoint saves and reloads
- inference produces tokens without crashing
- validation loss/perplexity is recorded

Do not make v0.1 the production default. After the pipeline passes, grow and improve the permitted dataset, add evaluation, and train progressively larger versions before Engine integration.
