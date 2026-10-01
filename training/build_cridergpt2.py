"""Build or continue CriderGPT 2.0 while preserving CriderGPT 1.0.

If a trained CriderGPT 2.0 checkpoint already exists, the builder continues from it by
default. Otherwise it starts from CriderGPT 1.0. Identity and behavior examples are
upweighted so they are not drowned out by the much larger OASST1 corpus.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "training_data" / "cridergpt2"
DEFAULT_OASST1 = ROOT / "data" / "training" / "oasst1" / "train_conversations.jsonl"
DEFAULT_LOCAL = ROOT / "data" / "training" / "cridergpt2"
V1_CHECKPOINT = ROOT / "model" / "checkpoint"
DEFAULT_OUTPUT = ROOT / "model" / "cridergpt-2.0" / "checkpoint"
DEFAULT_WORK = ROOT / "artifacts" / "cridergpt2_build"

def run(cmd):
    print("\n>", " ".join(map(str, cmd)), flush=True); subprocess.run(cmd, cwd=ROOT, check=True)

def format_row(row):
    text=row.get("text")
    if isinstance(text,str) and text.strip(): return text.strip()
    messages=row.get("messages")
    if isinstance(messages,list):
        parts=[]
        for m in messages:
            if isinstance(m,dict) and m.get("role") in {"user","assistant"} and isinstance(m.get("content"),str) and m["content"].strip():
                parts.append(f'<|{m["role"]}|>\n{m["content"].strip()}')
        if parts: return "\n".join(parts)+"\n<|eos|>"
    return None

def append_jsonl(source,dest,repeat=1):
    if not source.exists(): return 0
    records=[]
    with source.open(encoding="utf-8") as h:
        for n,line in enumerate(h,1):
            if not line.strip(): continue
            try: row=json.loads(line)
            except json.JSONDecodeError as e: raise SystemExit(f"{source}:{n}: invalid JSON: {e}") from e
            text=format_row(row)
            if text: records.append(text)
    for _ in range(max(1,repeat)):
        for text in records: dest.write(json.dumps({"text":text},ensure_ascii=False)+"\n")
    return len(records)*max(1,repeat)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oasst1",type=Path,default=DEFAULT_OASST1)
    p.add_argument("--base-model",type=Path,default=None,help="Explicit starting checkpoint; default is existing 2.0, otherwise 1.0")
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--work-dir",type=Path,default=DEFAULT_WORK)
    p.add_argument("--identity-weight",type=int,default=100)
    p.add_argument("--behavior-weight",type=int,default=25)
    p.add_argument("--writing-samples",type=Path,default=DEFAULT_LOCAL/"writing_samples.jsonl")
    p.add_argument("--founder-memory",type=Path,default=DEFAULT_LOCAL/"founder_memory.jsonl")
    p.add_argument("--include-founder-memory",action="store_true")
    p.add_argument("--validation-ratio",type=float,default=.05)
    p.add_argument("--epochs",type=float,default=2.0)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--gradient-accumulation",type=int,default=4)
    p.add_argument("--learning-rate",type=float,default=5e-5)
    p.add_argument("--overwrite-2.0",action="store_true",help="Allow replacing the current 2.0 checkpoint after successful continued training")
    a=p.parse_args()
    base=a.base_model or (DEFAULT_OUTPUT if DEFAULT_OUTPUT.exists() and any(DEFAULT_OUTPUT.iterdir()) else V1_CHECKPOINT)
    if not base.exists(): raise SystemExit(f"Base checkpoint not found: {base}")
    if not a.oasst1.exists(): raise SystemExit(f"Prepared OASST1 data not found: {a.oasst1}")
    continuing = base.resolve()==a.output_dir.resolve()
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not continuing and not a.overwrite_2_0:
        raise SystemExit(f"{a.output_dir} already contains a checkpoint. Use --overwrite-2.0 only if replacement is intentional.")
    if a.work_dir.exists(): shutil.rmtree(a.work_dir)
    a.work_dir.mkdir(parents=True); combined=a.work_dir/"combined.jsonl"; counts={}
    with combined.open("w",encoding="utf-8") as out:
        counts["oasst1"]=append_jsonl(a.oasst1,out)
        counts["identity"]=append_jsonl(SEED_DIR/"identity.jsonl",out,a.identity_weight)
        counts["behavior"]=append_jsonl(SEED_DIR/"behavior.jsonl",out,a.behavior_weight)
        counts["writing_samples"]=append_jsonl(a.writing_samples,out)
        counts["founder_memory"]=append_jsonl(a.founder_memory,out) if a.include_founder_memory else 0
    print(f"Starting checkpoint: {base}"); print("Combined training rows:")
    for k,v in counts.items(): print(f"  {k}: {v:,}")
    prepared=a.work_dir/"prepared"
    run([sys.executable,"-m","training.prepare_dataset",str(combined),"--output-dir",str(prepared),"--text-field","text","--validation-ratio",str(a.validation_ratio),"--seed","42"])
    trained=a.work_dir/"trained"
    run([sys.executable,"-m","training.train_causal_lm","--train-file",str(prepared/"train.jsonl"),"--validation-file",str(prepared/"validation.jsonl"),"--tokenizer",str(base),"--base-model",str(base),"--output-dir",str(trained),"--block-size","256","--epochs",str(a.epochs),"--batch-size",str(a.batch_size),"--gradient-accumulation",str(a.gradient_accumulation),"--learning-rate",str(a.learning_rate),"--seed","42"])
    if a.output_dir.exists(): shutil.rmtree(a.output_dir)
    shutil.copytree(trained,a.output_dir)
    manifest={"name":"CriderGPT 2.0","family":"CriderGPT Native","version":"2.0.0","continued_from":str(base.resolve()),"checkpoint":str(a.output_dir.resolve()),"training_rows":counts,"identity_weight":a.identity_weight,"behavior_weight":a.behavior_weight,"includes_private_founder_memory":bool(a.include_founder_memory)}
    (a.output_dir/"cridergpt_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"\nCriderGPT 2.0 checkpoint saved to {a.output_dir}"); print("CriderGPT 1.0 remains unchanged at model/checkpoint."); return 0
if __name__=="__main__": raise SystemExit(main())
