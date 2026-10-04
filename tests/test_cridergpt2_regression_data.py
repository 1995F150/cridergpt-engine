import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "training_data" / "cridergpt2" / "regression_cases.json"


def test_regression_cases_cover_identity_and_current_pricing():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    prompts = {row["prompt"]: row["must_include"] for row in cases}
    assert "Who are you?" in prompts
    assert "Who created you?" in prompts
    assert "Jessie Crider" in prompts["Who created you?"]
    assert "$3" in prompts["How much is CriderGPT Plus?"]
    assert "$7" in prompts["How much is CriderGPT Pro?"]


def test_regression_cases_have_nonempty_expectations():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    assert cases
    assert all(row.get("prompt") and row.get("must_include") for row in cases)
