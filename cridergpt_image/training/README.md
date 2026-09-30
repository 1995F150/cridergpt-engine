# CriderGPT Image Stage 6 — Generator Training

Stage 6 connects the prior components into the first end-to-end training loop.

Required inputs:
- Stage 2 manifest with `image` + `caption`
- Stage 3 tokenizer and trained text-encoder checkpoint
- Stage 4 trained autoencoder checkpoint
- Stage 5 generator architecture

The Stage 3 and Stage 4 networks are frozen during Stage 6. The Stage 5 generator learns to predict noise added to Stage 4 latents while being conditioned by Stage 3 prompt embeddings.

Example:

```bash
python3 -m cridergpt_image.training.train_generator \
  --manifest data/processed/dataset_manifest.jsonl \
  --tokenizer checkpoints/stage3/tokenizer.json \
  --text-checkpoint checkpoints/stage3/text_encoder_best.pt \
  --autoencoder-checkpoint checkpoints/stage4/autoencoder_best.pt \
  --output checkpoints/stage6 \
  --epochs 50 \
  --batch-size 4 \
  --device auto
```

Outputs:
- `training_config.json`
- `generator_latest.pt`
- `generator_best.pt`
- `training_summary.json`

This code does not mean the generator is trained yet. A valid Stage 3 checkpoint, Stage 4 checkpoint, and real image/caption dataset are prerequisites to producing learned Stage 6 weights.
