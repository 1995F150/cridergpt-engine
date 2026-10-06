# CriderGPT 2.1 "Nova"

Nova is the quality-focused successor to CriderGPT 2.0. The 2.0 checkpoint remains the baseline and must not be overwritten.

## Goals

1. Improve general instruction following and basic reasoning.
2. Reduce repetitive or degenerate generation.
3. Preserve CriderGPT identity and creator attribution.
4. Train personalized/context data as actual user/assistant conversations instead of relying on plain fact-continuation text.
5. Add held-out capability evaluations so a checkpoint is promoted only when it improves on 2.0.
6. Keep private/live memory separate from model weights unless a scoped export is explicitly selected.

## Dataset curriculum

Nova datasets should be versioned by category and validated before training. Initial categories:

- general instruction following
- reasoning and arithmetic
- coding
- Linux, servers, and networking
- computer hardware and troubleshooting
- cybersecurity fundamentals and defensive administration
- conversational quality
- CriderGPT identity/behavior
- explicitly selected personalized context

Synthetic datasets are allowed, but generated records must pass schema validation, deduplication, repetition checks, and held-out evaluation. Generated answers should not be accepted merely because they were generated.

## Required chat format

Instruction examples should use structured messages:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

The builder may serialize these to the model's role markers. Plain factual paragraphs are not a substitute for Q&A/instruction examples when the intended behavior is question answering.

## Evaluation gates

Compare Nova against the frozen CriderGPT 2.0 baseline on held-out prompts covering:

- identity and creator
- arithmetic
- instruction following
- coding
- networking/Linux
- factual question answering
- personalized-context recall where explicitly included
- repetition/degeneration

A Nova candidate should not replace the baseline merely because training loss decreases.

## Version paths

```text
model/cridergpt-2.0/checkpoint       frozen 2.0 baseline
model/cridergpt-2.1-nova/checkpoint  Nova candidate/final
artifacts/cridergpt21_build/         temporary Nova build artifacts
```

## First implementation work

- inspect `training.train_causal_lm` tokenization, labels, and truncation
- add Nova-specific identity records
- convert selected personalized facts into explicit Q&A records
- build category-balanced dataset manifests
- add held-out capability tests
- tune generation defaults only after training/evaluation issues are understood
- do not start a long training run until the dataset and evaluation pipeline pass validation
