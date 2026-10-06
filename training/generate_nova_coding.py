"""Generate deterministic, duplicate-safe Nova coding instruction records."""
import argparse
import json
import random
from pathlib import Path

def rec(q, a):
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}

def make(kind, rng, n):
    a, b = rng.randint(2, 999), rng.randint(2, 999)
    name = f"value_{n}"

    if kind == "py_add":
        q = f"Write a Python function named add_{n} that adds {a} to its argument."
        atext = f"""\`\`\`python
def add_{n}(x):
    return x + {a}
\`\`\`"""
    elif kind == "py_even":
        q = f"In Python, store {a * 2} in a variable named number_{n} and check whether it is even."
        atext = f"""\`\`\`python
number_{n} = {a * 2}
is_even_{n} = number_{n} % 2 == 0
print(is_even_{n})
\`\`\`
This prints \`True\`."""
    elif kind == "py_list":
        q = f"In Python, create a list named values_{n} containing {a}, {b}, and {a+b}, then print its length."
        atext = f"""\`\`\`python
values_{n} = [{a}, {b}, {a+b}]
print(len(values_{n}))
\`\`\`
The length is 3."""
    elif kind == "py_dict":
        q = f"Create a Python dictionary named data_{n} with key \`{name}\` and integer value {a}."
        atext = f"""\`\`\`python
data_{n} = {{"{name}": {a}}}
\`\`\`"""
    elif kind == "js_add":
        q = f"Write a JavaScript arrow function named add{n} that adds two numbers."
        atext = f"""\`\`\`javascript
const add{n} = (a, b) => a + b;
\`\`\`"""
    elif kind == "js_array":
        q = f"In JavaScript, create an array named values{n} containing {a}, {b}, and {a+b}, then read its length."
        atext = f"""\`\`\`javascript
const values{n} = [{a}, {b}, {a+b}];
const length{n} = values{n}.length;
\`\`\`
The length is 3."""
    elif kind == "sql_select":
        q = f"For query exercise {n}, write SQL selecting id and name from users where id equals {a}."
        atext = f"""\`\`\`sql
SELECT id, name FROM users WHERE id = {a};
\`\`\`"""
    elif kind == "sql_param":
        q = f"For SQL exercise {n}, why should user input such as ID {a} be passed as a parameter instead of concatenated into SQL?"
        atext = "Parameterized queries keep data separate from SQL syntax, reducing SQL-injection risk and handling values safely."
    elif kind == "json":
        q = f"Write valid JSON for request {n} with keys \`id\`={a}, \`request_id\`={n}, and \`active\`=true."
        atext = f"""\`\`\`json
{{"id": {a}, "request_id": {n}, "active": true}}
\`\`\`"""
    elif kind == "http":
        status = rng.choice((400, 401, 403, 404, 500, 503))
        q = f"For API scenario {n}, an endpoint returns HTTP {status}. Is it a 4xx client-error response or a 5xx server-error response?"
        cls = "4xx client-error" if status < 500 else "5xx server-error"
        atext = f"HTTP {status} is a {cls} response."
    elif kind == "debug":
        q = f"In debugging scenario {n}, Python raises NameError for variable \`{name}\`. What should you inspect?"
        atext = "Check that the variable is spelled correctly, assigned before use, and available in the current scope."
    elif kind == "loop":
        q = f"For Python exercise {n}, print every integer from {a} through {a+4} inclusive."
        atext = f"""\`\`\`python
for number in range({a}, {a+5}):
    print(number)
\`\`\`"""
    elif kind == "test":
        q = f"For test case {n}, write a Python assertion verifying that {a} + {b} equals {a+b}."
        atext = f"""\`\`\`python
assert {a} + {b} == {a+b}
\`\`\`"""
    elif kind == "file":
        q = f"Show Python that writes the text \`record-{n}\` to \`output_{n}.txt\` using UTF-8."
        atext = f"""\`\`\`python
with open("output_{n}.txt", "w", encoding="utf-8") as file:
    file.write("record-{n}")
\`\`\`"""
    elif kind == "git":
        q = f"What Git command stages a file named \`file_{n}.py\`?"
        atext = f"\`git add file_{n}.py\` stages that file."
    else:
        raise ValueError(kind)

    return q, atext

KINDS = (
    "py_add", "py_even", "py_list", "py_dict", "js_add", "js_array",
    "sql_select", "sql_param", "json", "http", "debug", "loop",
    "test", "file", "git",
)

def generate(count, seed):
    rng = random.Random(seed)
    rows = []
    seen = set()
    for index in range(count):
        q, a = make(KINDS[index % len(KINDS)], rng, index + 1)
        key = " ".join(q.lower().split())
        if key in seen:
            raise RuntimeError(f"duplicate prompt generated at record {index + 1}: {q}")
        seen.add(key)
        rows.append(rec(q, a))
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=10000)
    p.add_argument("--seed", type=int, default=211)
    p.add_argument("--output", type=Path, default=Path("data/training/cridergpt21/generated_coding.jsonl"))
    args = p.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    rows = generate(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} Nova coding records across {len(KINDS)} template families to {args.output}")

if __name__ == "__main__":
    main()
