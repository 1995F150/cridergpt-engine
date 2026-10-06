"""Run held-out CriderGPT Nova evaluations against a local checkpoint.

Reports general capability separately from version-specific identity so an older
baseline is not penalized merely for correctly identifying its own version.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

def norm(s):
    s=s.lower().strip().replace("×","x").replace("÷","/")
    s=re.sub(r"[^a-z0-9%$/\\.]+"," ",s)
    return " ".join(s.split())

def load_cases(path):
    with path.open("r",encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--eval",type=Path,default=Path("training_data/cridergpt21/eval/held_out.jsonl"))
    p.add_argument("--output",type=Path,default=Path("artifacts/nova_eval_results.json"))
    p.add_argument("--max-new-tokens",type=int,default=96)
    a=p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.checkpoint,local_files_only=True)
    model.eval()
    cases=load_cases(a.eval); results=[]; stats=defaultdict(lambda:[0,0])

    for case in cases:
        prompt=f"<|user|> {case['prompt']} <|assistant|>"
        inputs=tok(prompt,return_tensors="pt")
        with torch.no_grad():
            out=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,
                repetition_penalty=1.15,pad_token_id=tok.eos_token_id)
        generated=tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        ok=norm(case["expected_answer"]) in norm(generated)
        group="identity" if case["category"]=="identity" else "capability"
        stats[group][1]+=1; stats[group][0]+=int(ok)
        stats["overall"][1]+=1; stats["overall"][0]+=int(ok)
        results.append({**case,"generated":generated,"passed":ok,"score_group":group})
        print(f"[{'PASS' if ok else 'FAIL'}] {case['id']} {case['category']}: {generated[:160]}")

    def summary(group):
        passed,total=stats[group]
        return {"passed":passed,"total":total,"score":passed/total if total else 0}

    capability=summary("capability"); identity=summary("identity"); overall=summary("overall")
    payload={"checkpoint":a.checkpoint,"capability":capability,"identity":identity,
             "overall":overall,"results":results}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")

    print("\n--- Nova evaluation summary ---")
    print(f"General capability: {capability['passed']}/{capability['total']} ({capability['score']:.1%})")
    print(f"Version/identity:    {identity['passed']}/{identity['total']} ({identity['score']:.1%})")
    print(f"Raw overall:         {overall['passed']}/{overall['total']} ({overall['score']:.1%})")
    print("Promotion comparison should use General capability; identity is reported separately.")
    print(f"Saved: {a.output}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
