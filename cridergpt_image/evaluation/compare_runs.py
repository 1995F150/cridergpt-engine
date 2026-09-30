"""Compare Stage 7 evaluation JSON reports without hiding regressions."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("reports",nargs="+",type=Path)
    args=ap.parse_args()
    rows=[]
    for path in args.reports:
        r=json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "report":str(path),
            "samples":r.get("samples"),
            "noise_mse_mean":r.get("noise_mse_mean"),
            "noise_mse_median":r.get("noise_mse_median"),
            "checkpoint":r.get("generator_checkpoint")
        })
    rows.sort(key=lambda x: float("inf") if x["noise_mse_mean"] is None else x["noise_mse_mean"])
    print(json.dumps(rows,indent=2))

if __name__=="__main__":
    main()
