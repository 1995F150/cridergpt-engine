"""Evaluate a trained CriderGPT Image Stage 5/6 generator on a validation manifest."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cridergpt_image.autoencoder.model import CriderImageAutoencoder
from cridergpt_image.generator.model import CriderImageLatentGenerator
from cridergpt_image.text_conditioning.encoder import CriderImageTextEncoder
from cridergpt_image.text_conditioning.tokenizer import CriderImageTokenizer
from cridergpt_image.training.noise_schedule import LinearNoiseSchedule
from cridergpt_image.training.train_generator import ManifestDataset, load_state, choose_device

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--tokenizer",type=Path,required=True)
    ap.add_argument("--text-checkpoint",type=Path,required=True)
    ap.add_argument("--autoencoder-checkpoint",type=Path,required=True)
    ap.add_argument("--generator-checkpoint",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="auto")
    ap.add_argument("--output",type=Path,default=Path("checkpoints/stage7/evaluation.json"))
    args=ap.parse_args()

    torch.manual_seed(args.seed)
    device=choose_device(args.device)
    tok=CriderImageTokenizer.load(args.tokenizer)
    text=CriderImageTextEncoder(vocab_size=len(tok.vocab),pad_id=tok.pad_id)
    ae=CriderImageAutoencoder()
    gen=CriderImageLatentGenerator()
    load_state(text,args.text_checkpoint)
    load_state(ae,args.autoencoder_checkpoint)
    load_state(gen,args.generator_checkpoint)
    text.to(device).eval(); ae.to(device).eval(); gen.to(device).eval()
    schedule=LinearNoiseSchedule(args.steps).to(device)
    ds=ManifestDataset(args.manifest,tok)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)

    losses=[]
    latent_abs=[]
    for images, ids, mask in dl:
        images,ids,mask=images.to(device),ids.to(device),mask.to(device)
        clean=ae.encode(images)
        emb=text(ids,mask)
        noise=torch.randn_like(clean)
        t=torch.randint(0,args.steps,(images.size(0),),device=device)
        noisy=schedule.add_noise(clean,noise,t)
        pred=gen(noisy,t,emb,mask)
        per=((pred-noise)**2).flatten(1).mean(1)
        losses.extend(per.cpu().tolist())
        latent_abs.extend(clean.abs().flatten(1).mean(1).cpu().tolist())

    report={
        "samples":len(ds),
        "seed":args.seed,
        "device":str(device),
        "noise_mse_mean":statistics.fmean(losses),
        "noise_mse_median":statistics.median(losses),
        "noise_mse_min":min(losses),
        "noise_mse_max":max(losses),
        "latent_abs_mean":statistics.fmean(latent_abs),
        "generator_checkpoint":str(args.generator_checkpoint)
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
