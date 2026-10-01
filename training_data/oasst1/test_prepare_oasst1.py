from __future__ import annotations

import pandas as pd

from prepare_oasst1 import reconstruct


def test_reconstruct_simple_conversation():
    df = pd.DataFrame([
        {"message_id": "u1", "parent_id": None, "role": "prompter", "text": "Hello", "lang": "en"},
        {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "Hi there", "lang": "en"},
    ])
    rows, stats = reconstruct(df, "en")
    assert stats.emitted == 1
    assert rows[0]["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_language_filtering():
    df = pd.DataFrame([
        {"message_id": "u1", "parent_id": None, "role": "prompter", "text": "Hello", "lang": "en"},
        {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "Hi", "lang": "en"},
        {"message_id": "u2", "parent_id": None, "role": "prompter", "text": "Hallo", "lang": "de"},
        {"message_id": "a2", "parent_id": "u2", "role": "assistant", "text": "Guten Tag", "lang": "de"},
    ])
    rows, _ = reconstruct(df, "en")
    assert len(rows) == 1


def test_rejects_non_alternating_roles():
    df = pd.DataFrame([
        {"message_id": "u1", "parent_id": None, "role": "prompter", "text": "Question", "lang": "en"},
        {"message_id": "u2", "parent_id": "u1", "role": "prompter", "text": "More", "lang": "en"},
        {"message_id": "a1", "parent_id": "u2", "role": "assistant", "text": "Answer", "lang": "en"},
    ])
    rows, stats = reconstruct(df, "en")
    assert rows == []
    assert stats.skipped_invalid_roles >= 1
