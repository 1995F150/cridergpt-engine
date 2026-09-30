# CriderGPT Image Stage 7 — Full Training and Evaluation

Stage 7 is where experimental code becomes a measured model-development process.

## Goals

1. Train Stage 3 and Stage 4 prerequisites on real authorized data.
2. Train the Stage 5 generator through the Stage 6 loop.
3. Keep a validation split that is never used for optimizer updates.
4. Measure validation noise-prediction MSE.
5. Compare checkpoints/runs instead of declaring quality from training loss alone.
6. Maintain a fixed canonical prompt suite for later Stage 8 image-generation comparisons.

## Validation example

```bash
python3 -m cridergpt_image.evaluation.evaluate_generator \
  --manifest data/processed/validation_manifest.jsonl \
  --tokenizer checkpoints/stage3/tokenizer.json \
  --text-checkpoint checkpoints/stage3/text_encoder_best.pt \
  --autoencoder-checkpoint checkpoints/stage4/autoencoder_best.pt \
  --generator-checkpoint checkpoints/stage6/generator_best.pt \
  --output checkpoints/stage7/evaluation.json \
  --device auto
```

## Stage 7 exit criteria

Stage 7 is NOT complete merely because training finishes.

Before moving to production-quality inference, record:
- exact dataset manifest/version
- train/validation sizes
- tokenizer hash/version
- Stage 3/4/5 checkpoint hashes
- training hyperparameters
- best training loss
- validation noise MSE
- visual sample review from the fixed prompt suite once Stage 8 sampling exists
- evidence that the model is not simply memorizing the tiny overfit test batch

The current 19-image starter batch is for pipeline verification only, not general image-model training.
