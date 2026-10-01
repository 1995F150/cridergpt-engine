from pathlib import Path

from training.build_cridergpt2 import append_jsonl


def test_append_jsonl_weights_seed_data(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"messages":[{"role":"user","content":"Who are you?"},{"role":"assistant","content":"CriderGPT 2.0"}]}\n',
        encoding="utf-8",
    )
    output = tmp_path / "combined.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        count = append_jsonl(source, handle, repeat=3)
    assert count == 3
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all("<|assistant|>" in line for line in lines)
