# CriderGPT Image Stage 4 Training

Stage 4 trains the CriderGPT-owned latent autoencoder defined in `model.py`.

## Input

Point `--images` at a directory containing PNG/JPEG/WebP training images. Images are converted to RGB, resized/cropped to 256x256 by default, converted to tensors, and normalized to [-1, 1].

## Small overfit test

```bash
python3 cridergpt_image/autoencoder/train.py \
  --images /path/to/cridergpt_training_batch_001/images \
  --output checkpoints/stage4-batch001 \
  --epochs 100 \
  --batch-size 4 \
  --device auto
```

`auto` selects CUDA when available, then Apple MPS when available, otherwise CPU.

## Outputs

The trainer writes:

- `training_config.json` including SHA-256 for every input image
- `autoencoder_best.pt`
- periodic `autoencoder_epoch_XXXX.pt` checkpoints
- `training_summary.json`

The initial 19-image batch is intended only as a pipeline/overfit test. It is not sufficient for a general-purpose image generator.

Stage 4 is complete only after a real training run shows that reconstructions improve and the resulting checkpoint is validated. Do not mark Stage 4 trained merely because this script exists.
