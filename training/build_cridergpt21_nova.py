"""Build CriderGPT 2.1 Nova from the frozen CriderGPT 2.0 checkpoint."""
from __future__ import annotations
import argparse,json,random,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"model"/"cridergpt-2.0"/"checkpoint"
OUT=ROOT/"model"/"cridergpt-2.1-nova"/"checkpoint"
WORK=ROOT/"artifacts"/"cridergpt21_build"
SEEDS=ROOT/"training_data"/"cridergpt21"
GENERATED=ROOT/"data"/"training"/"cridergpt21"

def run(cmd):
    print("\n>"," ".join(map(str,cmd)),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)

def text_rows(path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line)
            if isinstance(row.get("messages"),list):
                parts=[]
                for m in row["messages"]:
                    if m.get("role") in ("user","assistant") and str(m.get("content","")).strip():
                        parts.append(f"<|{m['role']}|>\n{m['content'].strip()}")
                if parts: rows.append("\n".join(parts)+"\n<|eos|>")
            elif isinstance(row.get("text"),str) and row["text"].strip():
                rows.append(row["text"].strip())
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-model",type=Path,default=BASE)
    p.add_argument("--output-dir",type=Path,default=OUT)
    p.add_argument("--work-dir",type=Path,default=WORK)
    p.add_argument("--epochs",type=float,default=1.0)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--gradient-accumulation",type=int,default=4)
    p.add_argument("--learning-rate",type=float,default=2e-5)
    p.add_argument("--validation-ratio",type=float,default=.03)
    p.add_argument("--smoke-test",action="store_true")
    p.add_argument("--smoke-records",type=int,default=1200)
    args=p.parse_args()

    sources=[
      SEEDS/"identity.jsonl",
      SEEDS/"reasoning_math_seed.jsonl",
      GENERATED/"generated_reasoning_math.jsonl",
      GENERATED/"generated_coding.jsonl",
      GENERATED/"generated_networking_linux_servers.jsonl",
    ]
    missing=[str(x) for x in sources if not x.exists()]
    if missing: raise SystemExit("Missing Nova dataset(s):\n  "+"\n  ".join(missing))
    if not args.base_model.exists(): raise SystemExit(f"Frozen 2.0 checkpoint not found: {args.base_model}")
    if args.output_dir.resolve()==args.base_model.resolve(): raise SystemExit("Refusing to overwrite the 2.0 baseline.")

    run([sys.executable,"training/validate_nova_dataset.py",*map(str,sources)])
    run([sys.executable,"training/audit_nova_datasets.py",*map(str,sources[1:])])

    rows=[]
    counts={}
    for src in sources:
        part=text_rows(src); counts[str(src.relative_to(ROOT))]=len(part); rows.extend(part)
    rng=random.Random(42); rng.shuffle(rows)
    if args.smoke_test: rows=rows[:min(args.smoke_records,len(rows))]

    if args.work_dir.exists(): shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)
    combined=args.work_dir/"combined.jsonl"
    with combined.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps({"text":row},ensure_ascii=False)+"\n")

    prepared=args.work_dir/"prepared"
    run([sys.executable,"-m","training.prepare_dataset",str(combined),"--output-dir",str(prepared),
         "--validation-ratio",str(args.validation_ratio),"--seed","42"])
    trained=args.work_dir/"trained"
    run([sys.executable,"-m","training.train_causal_lm","--train-file",str(prepared/"train.jsonl"),
         "--validation-file",str(prepared/"validation.jsonl"),"--tokenizer",str(args.base_model),
         "--base-model",str(args.base_model),"--output-dir",str(trained),"--block-size","256",
         "--epochs",str(args.epochs),"--batch-size",str(args.batch_size),
         "--gradient-accumulation",str(args.gradient_accumulation),
         "--learning-rate",str(args.learning_rate),"--seed","42"])
    if args.output_dir.exists(): shutil.rmtree(args.output_dir)
    shutil.copytree(trained,args.output_dir)
    manifest={"name":"CriderGPT 2.1 Nova","version":"2.1.0","base_model":str(args.base_model),
              "smoke_test":args.smoke_test,"rows_used":len(rows),"source_rows":counts,
              "held_out_eval":"training_data/cridergpt21/eval/held_out.jsonl (never trained)"}
    (args.output_dir/"cridergpt_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"\nNova {'SMOKE TEST' if args.smoke_test else 'training'} checkpoint saved to {args.output_dir}")
    return 0
if __name__=="__main__": raise SystemExit(main())
