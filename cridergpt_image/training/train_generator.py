"""Stage 6 training loop for CriderGPT Image latent generator.

Requires:
- Stage 2 image/caption manifest
- trained Stage 3 tokenizer/text encoder checkpoint
- trained Stage 4 autoencoder checkpoint
- Stage 5 generator architecture
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from cridergpt_image.autoencoder.model import CriderImageAutoencoder
from cridergpt_image.generator.model import CriderImageLatentGenerator
from cridergpt_image.text_conditioning.encoder import CriderImageTextEncoder
from cridergpt_image.text_conditioning.tokenizer import CriderImageTokenizer
from cridergpt_image.training.noise_schedule import LinearNoiseSchedule

class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, tokenizer: CriderImageTokenizer, resolution: int = 256):
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError("manifest is empty")
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(resolution, antialias=True),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = Path(row["image"])
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        ids, mask = self.tokenizer.encode(row["caption"])
        return image, torch.tensor(ids), torch.tensor(mask)

def load_state(module, checkpoint_path: Path, key: str | None = None):
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload[key] if key and key in payload else payload.get("model", payload)
    module.load_state_dict(state)

def choose_device(name: str):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--text-checkpoint", type=Path, required=True)
    ap.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("checkpoints/stage6"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)

    tokenizer = CriderImageTokenizer.load(args.tokenizer)
    text_encoder = CriderImageTextEncoder(vocab_size=len(tokenizer.vocab), pad_id=tokenizer.pad_id)
    autoencoder = CriderImageAutoencoder()
    generator = CriderImageLatentGenerator()

    load_state(text_encoder, args.text_checkpoint)
    load_state(autoencoder, args.autoencoder_checkpoint)
    text_encoder.to(device).eval()
    autoencoder.to(device).eval()
    for p in text_encoder.parameters(): p.requires_grad_(False)
    for p in autoencoder.parameters(): p.requires_grad_(False)

    generator.to(device).train()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr)
    schedule = LinearNoiseSchedule(args.steps).to(device)
    dataset = ManifestDataset(args.manifest, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k,v in config.items()}
    config["device_resolved"] = str(device)
    config["dataset_size"] = len(dataset)
    (args.output/"training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best = float("inf"); started = time.time()
    for epoch in range(1, args.epochs+1):
        total = 0.0
        for images, ids, mask in loader:
            images, ids, mask = images.to(device), ids.to(device), mask.to(device)
            with torch.no_grad():
                clean_latents = autoencoder.encode(images)
                text_embeddings = text_encoder(ids, mask)
            noise = torch.randn_like(clean_latents)
            timesteps = torch.randint(0, args.steps, (images.size(0),), device=device)
            noisy = schedule.add_noise(clean_latents, noise, timesteps)
            predicted = generator(noisy, timesteps, text_embeddings, mask)
            loss = torch.nn.functional.mse_loss(predicted, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * images.size(0)

        avg = total / len(dataset)
        print(f"epoch={epoch:04d} diffusion_loss={avg:.6f}")
        payload = {"epoch": epoch, "model": generator.state_dict(), "optimizer": optimizer.state_dict(), "loss": avg, "config": config}
        if avg < best:
            best = avg
            torch.save(payload, args.output/"generator_best.pt")
        torch.save(payload, args.output/"generator_latest.pt")

    summary = {"best_loss": best, "seconds": time.time()-started, "best_checkpoint": str(args.output/"generator_best.pt")}
    (args.output/"training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
