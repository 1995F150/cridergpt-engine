# Stage 2 — Dataset Pipeline

Stage 2 prepares auditable image/caption pairs for CriderGPT Image training. The repository does not bundle or download third-party training images.

## Input

Create a JSONL file with one record per image:

```json
{"image":"images/chicken.png","caption":"A red chicken standing in green grass.","source":"CriderGPT-owned photo collection","license":"owned"}
```

Every record must include `image`, `caption`, `source`, and `license`. Only include material you are authorized to use for model training.

## Run

```bash
python3 cridergpt_image/dataset/prepare_dataset.py data/raw.jsonl \
  --output data/processed/dataset_manifest.jsonl \
  --report data/processed/dataset_report.json
```

Install Pillow on the development/training machine if it is not already available.

## What the pipeline checks

- required caption and provenance metadata
- readable JPEG/PNG/WebP images
- minimum image dimensions (256 px by default)
- SHA-256 of every accepted image
- exact binary duplicate rejection
- deterministic train/validation split
- reproducible dataset-manifest SHA-256

Stage 2 intentionally does not perform AI-generated captioning or download datasets, because doing either would introduce an external-model/data dependency without an explicit provenance decision.

## Stage 2 completion criteria

The code pipeline is ready when it can process a real authorized image collection and produce a clean manifest/report. The data stage itself is complete only after a real dataset has been supplied, processed, reviewed, and frozen by manifest hash.
