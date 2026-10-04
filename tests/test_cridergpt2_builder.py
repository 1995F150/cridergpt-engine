from pathlib import Path

import pytest

from training.build_cridergpt2 import append_jsonl, append_records, validate_replacement


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


def test_zero_weight_excludes_optional_source(tmp_path: Path):
    output = tmp_path / "combined.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        count = append_records(["optional replay row"], handle, repeat=0)
    assert count == 0
    assert output.read_text(encoding="utf-8") == ""


def test_negative_weight_is_rejected(tmp_path: Path):
    output = tmp_path / "combined.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        with pytest.raises(ValueError, match="cannot be negative"):
            append_records(["row"], handle, repeat=-1)


def test_replacing_checkpoint_from_different_base_requires_opt_in(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="--overwrite-2.0"):
        validate_replacement(base, output, overwrite=False)
    validate_replacement(base, output, overwrite=True)


def test_continuing_from_current_checkpoint_does_not_require_overwrite(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")
    validate_replacement(output, output, overwrite=False)
