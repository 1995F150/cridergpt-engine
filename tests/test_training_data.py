from pathlib import Path

import pytest

from training.prepare_dataset import iter_text_records, write_jsonl


def test_iter_text_records_reads_txt(tmp_path: Path):
    source = tmp_path / "corpus.txt"
    source.write_text("hello\n\nworld\n", encoding="utf-8")

    assert list(iter_text_records(source, "text")) == ["hello", "world"]


def test_iter_text_records_reads_jsonl_field(tmp_path: Path):
    source = tmp_path / "corpus.jsonl"
    source.write_text(
        '{"text":"first"}\n{"text":"second"}\n',
        encoding="utf-8",
    )

    assert list(iter_text_records(source, "text")) == ["first", "second"]


def test_iter_text_records_rejects_missing_jsonl_field(tmp_path: Path):
    source = tmp_path / "corpus.jsonl"
    source.write_text('{"content":"missing"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="field 'text' must be a string"):
        list(iter_text_records(source, "text"))


def test_write_jsonl_emits_text_records(tmp_path: Path):
    destination = tmp_path / "train.jsonl"

    count = write_jsonl(destination, ["alpha", "beta"])

    assert count == 2
    assert destination.read_text(encoding="utf-8").splitlines() == [
        '{"text": "alpha"}',
        '{"text": "beta"}',
    ]
