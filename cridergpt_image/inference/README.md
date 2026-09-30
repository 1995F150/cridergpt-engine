# CriderGPT Image Stage 8 — Local Inference

Stage 8 is the first end-to-end local image generation entry point.

It loads only CriderGPT-owned local artifacts:

- Stage 3 tokenizer
- trained Stage 3 text encoder checkpoint
- trained Stage 4 autoencoder checkpoint
- trained Stage 6/7 generator checkpoint

No OpenAI, Gemini, Claude, Midjourney, hosted diffusion API, or remote inference service is used.

## Example

```bash
python3 -m cridergpt_image.inference.generate \
  --prompt "A futuristic computer server rack with blue lights" \
  --tokenizer checkpoints/stage3/tokenizer.json \
  --text-checkpoint checkpoints/stage3/text_encoder_best.pt \
  --autoencoder-checkpoint checkpoints/stage4/autoencoder_best.pt \
  --generator-checkpoint checkpoints/stage6/generator_best.pt \
  --output outputs/server-rack.png \
  --seed 42 \
  --inference-steps 50 \
  --device auto
```

The command writes the PNG plus a JSON sidecar recording prompt, seed, token IDs, timing, device, and checkpoint paths.

## Offline acceptance test

Once the three learned checkpoints exist:

1. Disconnect the training machine from the internet.
2. Run the command above.
3. Confirm a new PNG is generated.
4. Repeat with the same prompt + seed and verify reproducibility.
5. Change the prompt and verify the resulting generation changes.

Stage 8 code being present does **not** mean the model is already trained. The final PNG quality depends entirely on valid learned Stage 3, Stage 4, and Stage 6 generator weights.
