# CriderGPT 2.0 training upgrade

CriderGPT 2.0 continues from the local CriderGPT 1.0 checkpoint without
overwriting it.

## Versioned checkpoints

- CriderGPT 1.0: `model/checkpoint`
- CriderGPT 2.0: `model/cridergpt-2.0/checkpoint`

## Curated tracked data

`training_data/cridergpt2/identity.jsonl` contains stable model/founder identity
facts already present in the engine source.

`training_data/cridergpt2/behavior.jsonl` contains the engine's established
privacy, honesty, and writing-sample behavior.

Changing/private memory is deliberately not committed to Git.

## Export live writing samples locally

With Supabase configured:

```bash
python training/export_cridergpt2_context.py
```

This writes:

```text
data/training/cridergpt2/writing_samples.jsonl
```

The `data/` tree is gitignored.

Private scoped memory is opt-in only:

```bash
python training/export_cridergpt2_context.py --user-id USER_ID --include-private-memory
```

That produces `founder_memory.jsonl` locally. The 2.0 builder will not use it
unless `--include-founder-memory` is explicitly passed.

## Prepare only

No training starts just by merging this code. To inspect the command first:

```bash
python training/build_cridergpt2.py --help
```

When intentionally ready to train:

```bash
python training/build_cridergpt2.py
```

The builder expects the already-prepared OASST1 file at
`data/training/oasst1/train_conversations.jsonl` and the CriderGPT 1.0
checkpoint at `model/checkpoint`.
