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

## Emily 15K context dataset

The builder now loads this local dataset by default:

```text
data/training/cridergpt2/emily_context_15000.jsonl
```

It is mixed into rehearsal training as the `emily_context` source. The default
weight remains configurable with `--emily-context-weight`, and the file can be
overridden with `--emily-context PATH`.

The dataset contains private relationship context, so it stays under the
Git-ignored `data/` tree rather than being committed to this public repository.
Copy the generated `emily_context_15000.jsonl` into that path before training.


## Scoped memory-system training

To export the current retrievable memory system into a local, gitignored training
snapshot:

```bash
python training/export_cridergpt2_context.py --user-id USER_ID --include-memory-system
```

This writes:

```text
data/training/cridergpt2/memory_system.jsonl
```

The snapshot can include the core profile, project knowledge, long-term `ai_memory`,
user preferences, the legacy profile, user training inputs, and bounded recent
chat history. The builder loads this file automatically as the `memory_system`
source with weight 1.

The 15K Emily dataset remains the main Emily source, and
`data/training/cridergpt2/emily_context.jsonl` is also loaded as a smaller
`emily_context_updates` source so newly confirmed details are not lost.
