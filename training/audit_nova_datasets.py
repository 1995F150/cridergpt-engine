"""Audit Nova datasets before training."""
from __future__ import annotations
import argparse,json,re
from collections import Counter
from pathlib import Path

def norm(s): return re.sub(r"\s+"," ",s.strip().lower())
def shape(s):
    s=norm(s)
    s=re.sub(r"\b\d+(?:\.\d+)?\b","<n>",s)
    s=re.sub(r"value_\d+|values\d+|add_?\d+|file_\d+|output_\d+|record-\d+|nova-service-\d+","<id>",s)
    return s

def main():
    p=argparse.ArgumentParser()
    p.add_argument("paths",nargs="+",type=Path)
    p.add_argument("--max-shape-share",type=float,default=.08)
    a=p.parse_args()
    total=0; prompts=[]; answers=[]; shapes=Counter(); per={}
    for path in a.paths:
        n=0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row=json.loads(line); msgs=row["messages"]
                q=msgs[0]["content"]; ans=msgs[1]["content"]
                prompts.append(norm(q)); answers.append(norm(ans)); shapes[shape(q)]+=1
                n+=1
        per[str(path)]=n; total+=n
    dup_prompts=total-len(set(prompts))
    answer_counts=Counter(answers)
    repeated_answers=sum(v for v in answer_counts.values() if v>1)
    top=shapes.most_common(15)
    max_share=(top[0][1]/total) if total else 0
    print(f"Records: {total:,}")
    for k,v in per.items(): print(f"  {k}: {v:,}")
    print(f"Exact duplicate prompts: {dup_prompts:,}")
    print(f"Rows whose answer text is repeated: {repeated_answers:,}")
    print(f"Most common normalized prompt shape share: {max_share:.2%}")
    print("Top prompt shapes:")
    for s,n in top: print(f"  {n:>6,}  {n/total:>7.2%}  {s[:120]}")
    failed=dup_prompts>0 or max_share>a.max_shape_share
    print("AUDIT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0
if __name__=="__main__": raise SystemExit(main())
