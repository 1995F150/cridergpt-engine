import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training_data" / "cridergpt2"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flatten(records):
    return "\n".join(
        message["content"]
        for record in records
        for message in record.get("messages", [])
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )


def test_identity_dataset_contains_model_and_founder_identity():
    text = flatten(read_jsonl(DATA / "identity.jsonl"))
    assert "CriderGPT 2.0" in text
    assert "Jessie Crider" in text
    assert "founder and owner" in text


def test_behavior_dataset_preserves_privacy_rules():
    text = flatten(read_jsonl(DATA / "behavior.jsonl"))
    assert "Private user memory" in text
    assert "credentials" in text
    assert "do not invent" in text
