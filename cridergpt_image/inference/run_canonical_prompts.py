"""Run the fixed Stage 7 prompt suite through Stage 8 inference."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prompts",type=Path,default=Path("cridergpt_image/evaluation/prompts.json"))
    ap.add_argument("--tokenizer",type=Path,required=True)
    ap.add_argument("--text-checkpoint",type=Path,required=True)
    ap.add_argument("--autoencoder-checkpoint",type=Path,required=True)
    ap.add_argument("--generator-checkpoint",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,default=Path("outputs/canonical"))
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="auto")
    args=ap.parse_args()

    prompts=json.loads(args.prompts.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True,exist_ok=True)

    for item in prompts:
        out=args.output_dir/f'{item["id"]}.png'
        cmd=[
            sys.executable,"-m","cridergpt_image.inference.generate",
            "--prompt",item["prompt"],
            "--tokenizer",str(args.tokenizer),
            "--text-checkpoint",str(args.text_checkpoint),
            "--autoencoder-checkpoint",str(args.autoencoder_checkpoint),
            "--generator-checkpoint",str(args.generator_checkpoint),
            "--output",str(out),
            "--seed",str(args.seed),
            "--device",args.device,
        ]
        print("RUN"," ".join(cmd))
        subprocess.run(cmd,check=True)

if __name__=="__main__":
    main()
