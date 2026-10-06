"""Generate deterministic CriderGPT 2.1 Nova reasoning/math examples.

This generator produces verified template-based arithmetic records whose answers
are computed by code. It intentionally does not call an LLM. Generated files
should be passed through validate_nova_dataset.py before being admitted to a
training build.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def record(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def generate(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()
    templates = ("add", "subtract", "multiply", "divide", "percent", "rate")

    while len(rows) < count:
        kind = templates[len(rows) % len(templates)]
        if kind == "add":
            a, b = rng.randint(10, 999), rng.randint(10, 999)
            q = f"What is {a} + {b}?"
            atext = f"{a} + {b} = {a + b}."
        elif kind == "subtract":
            low, high = sorted((rng.randint(1, 999), rng.randint(1, 999)))
            q = f"What is {high} - {low}?"
            atext = f"{high} - {low} = {high - low}."
        elif kind == "multiply":
            a, b = rng.randint(2, 50), rng.randint(2, 25)
            q = f"What is {a} × {b}?"
            atext = f"{a} × {b} = {a * b}."
        elif kind == "divide":
            divisor, quotient = rng.randint(2, 25), rng.randint(2, 50)
            dividend = divisor * quotient
            q = f"What is {dividend} ÷ {divisor}?"
            atext = f"{dividend} ÷ {divisor} = {quotient}."
        elif kind == "percent":
            percent = rng.choice((5, 10, 20, 25, 50, 75))
            base = rng.randint(2, 100) * 20
            value = base * percent / 100
            shown = int(value) if value.is_integer() else value
            q = f"What is {percent}% of {base}?"
            atext = f"{percent}% of {base} is {shown}."
        else:
            hours = rng.randint(2, 10)
            speed = rng.randint(10, 80)
            distance = hours * speed
            q = (
                f"A vehicle travels {distance} miles in {hours} hours at a constant "
                "speed. What is its average speed?"
            )
            atext = (
                f"Average speed is distance divided by time: {distance} ÷ {hours} "
                f"= {speed} miles per hour."
            )

        if q in seen:
            continue
        seen.add(q)
        rows.append(record(q, atext))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=210)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/cridergpt21/generated_reasoning_math.jsonl"),
    )
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")

    rows = generate(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} generated Nova records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
