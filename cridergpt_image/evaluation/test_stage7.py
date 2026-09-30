import json
from pathlib import Path

def test_prompt_suite_exists():
    prompts=json.loads((Path(__file__).parent/"prompts.json").read_text(encoding="utf-8"))
    assert len(prompts) >= 5
    assert all("id" in p and "prompt" in p and p["prompt"].strip() for p in prompts)
    assert len({p["id"] for p in prompts}) == len(prompts)
