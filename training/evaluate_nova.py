"""Run held-out CriderGPT Nova evaluations against a local checkpoint.

The eval JSONL must never be mixed into training data. Scoring is deliberately
simple and reproducible: normalized expected-answer containment. Results are
written to JSON for later CriderGPT 2.0 vs 2.1 Nova comparison.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

def norm(s):
    s=s.lower().strip().replace("×","x").replace("÷","/")
    s=re.sub(r"[^a-z0-9%$/\.]+"," ",s)
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
    cases=load_cases(a.eval); results=[]; passed=0
    for case in cases:
        prompt=f"<|user|> {case['prompt']} <|assistant|>"
        inputs=tok(prompt,return_tensors="pt")
        with torch.no_grad():
            out=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,
                repetition_penalty=1.15,pad_token_id=tok.eos_token_id)
        generated=tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        ok=norm(case["expected_answer"]) in norm(generated)
        passed+=int(ok)
        results.append({**case,"generated":generated,"passed":ok})
        print(f"[{'PASS' if ok else 'FAIL'}] {case['id']} {case['category']}: {generated[:160]}")
    score=passed/len(cases) if cases else 0
    payload={"checkpoint":a.checkpoint,"passed":passed,"total":len(cases),"score":score,"results":results}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"Score: {passed}/{len(cases)} ({score:.1%})")
    print(f"Saved: {a.output}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
