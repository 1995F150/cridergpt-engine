"""Expanded deterministic CriderGPT 2.1 Nova reasoning/math generator."""
from __future__ import annotations
import argparse, json, random
from fractions import Fraction
from pathlib import Path

def rec(q,a):
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")

def make(kind,r):
    if kind=="add":
        a,b=r.randint(10,9999),r.randint(10,9999); return f"What is {a} + {b}?",f"{a} + {b} = {a+b}."
    if kind=="subtract":
        a,b=sorted((r.randint(1,9999),r.randint(1,9999))); return f"What is {b} - {a}?",f"{b} - {a} = {b-a}."
    if kind=="multiply":
        a,b=r.randint(2,100),r.randint(2,50); return f"What is {a} × {b}?",f"{a} × {b} = {a*b}."
    if kind=="divide":
        d,q=r.randint(2,50),r.randint(2,100); n=d*q; return f"What is {n} ÷ {d}?",f"{n} ÷ {d} = {q}."
    if kind=="percent":
        p=r.choice((5,10,20,25,30,40,50,75)); b=r.randint(2,100)*20; v=b*p/100
        return f"What is {p}% of {b}?",f"{p}% of {b} is {fmt(v)}."
    if kind=="percent_change":
        old=r.randint(2,40)*20; p=r.choice((10,20,25,50)); new=old*(100+p)//100
        return f"A price rises from ${old} to ${new}. What is the percentage increase?",f"The increase is ${new-old}. ({new-old} ÷ {old}) × 100 = {p}%, so the increase is {p}%."
    if kind=="rate":
        h,s=r.randint(2,10),r.randint(10,80); d=h*s
        return f"A vehicle travels {d} miles in {h} hours at a constant speed. What is its average speed?",f"Average speed = distance ÷ time = {d} ÷ {h} = {s} miles per hour."
    if kind=="ratio":
        k,a,b=r.randint(2,12),r.randint(1,9),r.randint(1,9)
        return f"A mixture uses {a} cups of A for every {b} cups of B. If A is scaled to {a*k} cups, how many cups of B are needed?",f"The scale factor is {k}. Multiply B by it: {b} × {k} = {b*k} cups."
    if kind=="fraction":
        den=r.choice((4,5,8,10,12)); num=r.randint(1,den-1); total=den*r.randint(2,30); f=Fraction(num,den); v=total*f.numerator//f.denominator
        return f"What is {f.numerator}/{f.denominator} of {total}?",f"{f.numerator}/{f.denominator} × {total} = {v}."
    if kind=="linear":
        x,c,o=r.randint(-20,50),r.randint(2,9),r.randint(-20,20); result=c*x+o; sign="+" if o>=0 else "-"
        return f"Solve for x: {c}x {sign} {abs(o)} = {result}.",f"Undo the constant, then divide by {c}. x = {x}."
    if kind=="money":
        price,qty=r.randint(5,50),r.randint(2,12); total=price*qty; paid=total+r.randint(1,10)*10
        return f"You buy {qty} items at ${price} each and pay ${paid}. How much change do you receive?",f"The cost is {qty} × ${price} = ${total}. Change = ${paid} - ${total} = ${paid-total}."
    if kind=="units":
        feet=r.randint(3,20); cut=r.choice((6,12,18,24,30)); rem=feet*12-cut
        return f"A board is {feet} feet long and {cut} inches are cut off. How many feet remain?",f"{feet} feet = {feet*12} inches. Subtract {cut} to get {rem} inches, or {fmt(rem/12)} feet."
    if kind=="area":
        a,b=r.randint(2,40),r.randint(2,30); return f"A rectangle is {a} feet by {b} feet. What is its area?",f"Area = {a} × {b} = {a*b} square feet."
    if kind=="perimeter":
        a,b=r.randint(2,40),r.randint(2,30); return f"A rectangle is {a} meters by {b} meters. What is its perimeter?",f"Perimeter = 2 × ({a} + {b}) = {2*(a+b)} meters."
    if kind=="average":
        ns=[r.randint(1,100) for _ in range(r.choice((3,4,5)))]; s=sum(ns)
        return f"What is the average of {', '.join(map(str,ns))}?",f"The sum is {s}. Divide by {len(ns)} to get {fmt(s/len(ns))}."
    if kind=="probability":
        red,blue=r.randint(1,10),r.randint(1,10); total=red+blue; f=Fraction(blue,total)
        return f"A bag has {red} red balls and {blue} blue balls. What is the probability of drawing blue?",f"There are {total} balls. The probability is {f.numerator}/{f.denominator}, or about {fmt(blue/total*100)}%."
    if kind=="sequence":
        start,step=r.randint(1,30),r.randint(2,15); ns=[start+step*i for i in range(4)]
        return f"What is the next number in {', '.join(map(str,ns))}, ?",f"Each term increases by {step}, so the next number is {ns[-1]+step}."
    if kind=="geometric":
        start,k=r.randint(1,8),r.choice((2,3,4)); ns=[start*k**i for i in range(4)]
        return f"What is the next number in {', '.join(map(str,ns))}, ?",f"Each term is multiplied by {k}, so the next number is {ns[-1]*k}."
    if kind=="logic":
        return "If A is greater than B and B is greater than C, what must be true about A and C?","By transitivity, A is greater than C."
    if kind=="logic_some":
        thing=r.choice(("birds","servers","vehicles","machines")); prop=r.choice(("fast","online","quiet","expensive"))
        return f"Some {thing} are {prop}. X is one of the {thing}. Can you conclude X is {prop}?",f"No. Saying some {thing} are {prop} does not mean every member of that group is {prop}."
    if kind=="multi_rate":
        m,each,h=r.randint(2,8),r.randint(3,20),r.randint(2,8); total=m*each*h
        return f"{m} machines each make {each} parts per hour. How many parts do they make together in {h} hours?",f"Together they make {m*each} parts per hour. Over {h} hours: {m*each} × {h} = {total} parts."
    if kind=="remaining":
        total=r.randint(2,50)*20; p=r.choice((10,20,25,40,50)); used=total*p//100
        return f"A drive has {total} GB and {p}% is used. How many GB remain?",f"{p}% of {total} is {used} GB. {total} - {used} = {total-used} GB remain."
    raise ValueError(kind)

KINDS=("add","subtract","multiply","divide","percent","percent_change","rate","ratio","fraction","linear","money","units","area","perimeter","average","probability","sequence","geometric","logic","logic_some","multi_rate","remaining")

def generate(count,seed):
    r=random.Random(seed); rows=[]; seen=set(); attempts=0
    while len(rows)<count:
        attempts+=1
        if attempts>count*200: raise RuntimeError("could not generate enough unique records")
        q,a=make(KINDS[(attempts-1)%len(KINDS)],r); key=" ".join(q.lower().split())
        if key in seen: continue
        seen.add(key); rows.append(rec(q,a))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--count",type=int,default=1000); p.add_argument("--seed",type=int,default=210)
    p.add_argument("--output",type=Path,default=Path("data/training/cridergpt21/generated_reasoning_math.jsonl")); a=p.parse_args()
    if a.count<1: raise SystemExit("--count must be at least 1")
    rows=generate(a.count,a.seed); a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(f"Wrote {len(rows)} Nova records across {len(KINDS)} template families to {a.output}")
    return 0
if __name__=="__main__": raise SystemExit(main())
