"""Generate deterministic Nova coding instruction records."""
import argparse,json,random
from pathlib import Path
LANGS=("Python","JavaScript")
def rec(q,a): return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}
def make(k,r,n):
    a,b=r.randint(2,99),r.randint(2,99); name=f"value_{n}"
    if k=="py_add": return f"Write a Python function named add_{n} that returns the sum of {a} and its argument.",f"\`\`\`python\ndef add_{n}(x):\n    return x + {a}\n\`\`\`"
    if k=="py_even": return f"Write Python code that checks whether the integer {a*2} is even.",f"\`\`\`python\nnumber = {a*2}\nis_even = number % 2 == 0\nprint(is_even)\n\`\`\`\nThis prints \`True\`."
    if k=="py_list": return f"In Python, create a list containing {a}, {b}, and {a+b}, then print its length.",f"\`\`\`python\nvalues = [{a}, {b}, {a+b}]\nprint(len(values))\n\`\`\`\nThe length is 3."
    if k=="py_dict": return f"Create a Python dictionary with key \`{name}\` and integer value {a}.",f"\`\`\`python\ndata = {{\"{name}\": {a}}}\n\`\`\`"
    if k=="js_add": return f"Write a JavaScript arrow function named add{n} that adds two numbers.",f"\`\`\`javascript\nconst add{n} = (a, b) => a + b;\n\`\`\`"
    if k=="js_array": return f"In JavaScript, create an array containing {a}, {b}, and {a+b} and read its length.",f"\`\`\`javascript\nconst values = [{a}, {b}, {a+b}];\nconst length = values.length;\n\`\`\`\nThe length is 3."
    if k=="sql_select": return f"Write SQL to select id and name from users where id equals {a}.",f"\`\`\`sql\nSELECT id, name FROM users WHERE id = {a};\n\`\`\`"
    if k=="sql_param": return f"Why should user input such as ID {a} be passed as a SQL parameter instead of concatenated into SQL?","Parameterized queries keep data separate from SQL syntax, reducing SQL-injection risk and handling values safely."
    if k=="json": return f"Write valid JSON with keys \`id\`={a} and \`active\`=true.",f"\`\`\`json\n{{\"id\":{a},\"active\":true}}\n\`\`\`"
    if k=="http": return f"An API returns HTTP {r.choice((400,401,403,404,500,503))}. Explain what class of response it is.","HTTP 4xx codes indicate client-side request errors, while 5xx codes indicate server-side failures. Check the exact status code and server response for the specific cause."
    if k=="debug": return f"Python raises NameError for variable \`{name}\`. What should you inspect?","Check that the variable is spelled correctly, assigned before use, and available in the current scope."
    if k=="loop": return f"Write Python that prints integers from {a} through {a+4} inclusive.",f"\`\`\`python\nfor number in range({a}, {a+5}):\n    print(number)\n\`\`\`"
    if k=="test": return f"Write a Python assertion verifying that {a} + {b} equals {a+b}.",f"\`\`\`python\nassert {a} + {b} == {a+b}\n\`\`\`"
    if k=="file": return f"Show Python that writes the text \`record-{n}\` to \`output_{n}.txt\` using UTF-8.",f"\`\`\`python\nwith open(\"output_{n}.txt\", \"w\", encoding=\"utf-8\") as file:\n    file.write(\"record-{n}\")\n\`\`\`"
    if k=="git": return f"What Git command stages a file named \`file_{n}.py\`? ",f"\`git add file_{n}.py\` stages that file."
    raise ValueError(k)
KINDS=("py_add","py_even","py_list","py_dict","js_add","js_array","sql_select","sql_param","json","http","debug","loop","test","file","git")
def generate(count,seed):
    r=random.Random(seed); rows=[]
    for n in range(count): rows.append(rec(*make(KINDS[n%len(KINDS)],r,n+1)))
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument("--count",type=int,default=10000);p.add_argument("--seed",type=int,default=211);p.add_argument("--output",type=Path,default=Path("data/training/cridergpt21/generated_coding.jsonl"));a=p.parse_args()
    if a.count<1: raise SystemExit("--count must be at least 1")
    rows=generate(a.count,a.seed);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8") as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print(f"Wrote {len(rows)} Nova coding records across {len(KINDS)} template families to {a.output}")
if __name__=="__main__":main()
