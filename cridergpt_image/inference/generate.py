"""Local CriderGPT Image Stage 8 inference.

Loads CriderGPT-owned tokenizer/text encoder, Stage 4 autoencoder decoder,
and Stage 6/7 generator checkpoint. Generates an image locally with no API.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from cridergpt_image.autoencoder.model import CriderImageAutoencoder
from cridergpt_image.generator.model import CriderImageLatentGenerator
from cridergpt_image.text_conditioning.encoder import CriderImageTextEncoder
from cridergpt_image.text_conditioning.tokenizer import CriderImageTokenizer
from cridergpt_image.training.noise_schedule import LinearNoiseSchedule
from cridergpt_image.training.train_generator import load_state, choose_device


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    x = x[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x, mode="RGB")


@torch.no_grad()
def sample(
    prompt: str,
    tokenizer: CriderImageTokenizer,
    text_encoder: CriderImageTextEncoder,
    autoencoder: CriderImageAutoencoder,
    generator: CriderImageLatentGenerator,
    schedule: LinearNoiseSchedule,
    device: torch.device,
    seed: int = 42,
    inference_steps: int = 50,
):
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    ids, mask = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.tensor([mask], dtype=torch.long, device=device)
    text_embeddings = text_encoder(ids, mask)

    latent = torch.randn((1, 4, 32, 32), generator=g, device=device)

    # Use a descending subset of the full training schedule.
    timesteps = torch.linspace(
        schedule.steps - 1, 0, inference_steps, device=device
    ).long().unique_consecutive()

    for i, t in enumerate(timesteps):
        tb = torch.full((1,), int(t.item()), dtype=torch.long, device=device)
        pred_noise = generator(latent, tb, text_embeddings, mask)

        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        beta = schedule.betas[t]

        mean = (latent - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) / torch.sqrt(alpha)

        if t.item() > 0:
            noise = torch.randn(latent.shape, generator=g, device=device)
            latent = mean + torch.sqrt(beta) * noise
        else:
            latent = mean

    image = autoencoder.decode(latent)
    return image, ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--text-checkpoint", type=Path, required=True)
    ap.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    ap.add_argument("--generator-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("cridergpt-output.png"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inference-steps", type=int, default=50)
    ap.add_argument("--training-steps", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = choose_device(args.device)
    tok = CriderImageTokenizer.load(args.tokenizer)

    text = CriderImageTextEncoder(vocab_size=len(tok.vocab), pad_id=tok.pad_id)
    ae = CriderImageAutoencoder()
    gen = CriderImageLatentGenerator()

    load_state(text, args.text_checkpoint)
    load_state(ae, args.autoencoder_checkpoint)
    load_state(gen, args.generator_checkpoint)

    text.to(device).eval()
    ae.to(device).eval()
    gen.to(device).eval()
    schedule = LinearNoiseSchedule(args.training_steps).to(device)

    started = time.time()
    image_tensor, ids, mask = sample(
        args.prompt, tok, text, ae, gen, schedule, device,
        seed=args.seed, inference_steps=args.inference_steps,
    )
    elapsed = time.time() - started

    image = tensor_to_pil(image_tensor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)

    metadata = {
        "prompt": args.prompt,
        "seed": args.seed,
        "inference_steps": args.inference_steps,
        "device": str(device),
        "seconds": elapsed,
        "output": str(args.output),
        "prompt_token_ids": ids[0].detach().cpu().tolist(),
        "prompt_attention_mask": mask[0].detach().cpu().tolist(),
        "tokenizer": str(args.tokenizer),
        "text_checkpoint": str(args.text_checkpoint),
        "autoencoder_checkpoint": str(args.autoencoder_checkpoint),
        "generator_checkpoint": str(args.generator_checkpoint),
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
