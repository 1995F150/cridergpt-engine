"""Train the CriderGPT Image Stage 4 latent autoencoder on an image folder."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import CriderImageAutoencoder

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, resolution: int = 256):
        self.files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
        if not self.files:
            raise RuntimeError(f"No supported images found in {root}")
        self.transform = transforms.Compose([
            transforms.Resize(resolution, antialias=True),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested != "auto": return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("checkpoints/stage4"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    dataset = ImageFolderDataset(args.images, args.resolution)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = CriderImageAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()
    args.output.mkdir(parents=True, exist_ok=True)

    run = {
        "images": str(args.images), "image_count": len(dataset), "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.lr, "resolution": args.resolution,
        "seed": args.seed, "device": str(device),
        "dataset_sha256": {str(p): sha256(p) for p in dataset.files},
    }
    (args.output / "training_config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    best = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(images)
            loss = loss_fn(reconstruction, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * images.size(0)
        avg = total / len(dataset)
        print(f"epoch={epoch:04d} loss={avg:.6f}")
        payload = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "loss": avg, "config": run}
        if avg < best:
            best = avg; torch.save(payload, args.output / "autoencoder_best.pt")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(payload, args.output / f"autoencoder_epoch_{epoch:04d}.pt")

    summary = {"best_loss": best, "seconds": time.time() - started, "best_checkpoint": str(args.output / "autoencoder_best.pt")}
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
