import json
from pathlib import Path

from training.export_cridergpt2_dataset import normalize_conversation, redact


def test_active_branch_preserves_user_assistant_order():
    conv = {
        "current_node": "a1",
        "mapping": {
            "root": {"parent": None, "message": None},
            "u1": {
                "parent": "root",
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Who are you?"]},
                    "create_time": 1,
                },
            },
            "a1": {
                "parent": "u1",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["I am CriderGPT 2.0."]},
                    "create_time": 2,
                },
            },
            "unused": {
                "parent": "u1",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["Wrong branch"]},
                    "create_time": 3,
                },
            },
        },
    }
    rendered = normalize_conversation(conv, 12000)
    assert rendered == (
        "<|user|>\nWho are you?\n"
        "<|assistant|>\nI am CriderGPT 2.0.\n<|eos|>"
    )
    assert "Wrong branch" not in rendered


def test_redacts_obvious_credentials():
    text = redact("password=supersecret123 and api_key: abcdefghijklmnop")
    assert "supersecret123" not in text
    assert "abcdefghijklmnop" not in text
    assert text.count("[REDACTED]") == 2


def test_dangling_user_turn_is_removed():
    conv = {
        "current_node": "u2",
        "mapping": {
            "u1": {
                "parent": None,
                "message": {"author": {"role": "user"}, "content": {"parts": ["Hello"]}},
            },
            "a1": {
                "parent": "u1",
                "message": {"author": {"role": "assistant"}, "content": {"parts": ["Hi there"]}},
            },
            "u2": {
                "parent": "a1",
                "message": {"author": {"role": "user"}, "content": {"parts": ["Unanswered"]}},
            },
        },
    }
    rendered = normalize_conversation(conv, 12000)
    assert "Hello" in rendered
    assert "Hi there" in rendered
    assert "Unanswered" not in rendered
