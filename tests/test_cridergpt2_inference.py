from pathlib import Path

from cridergpt_stage5.inference import (
    ASSISTANT_TAG,
    EOS_TAG,
    USER_TAG,
    checkpoint_path,
    clean_generated_text,
    format_chat_prompt,
)


def test_chat_prompt_uses_role_markers():
    rendered = format_chat_prompt("Hello")
    assert rendered == f"{USER_TAG}\nHello\n{ASSISTANT_TAG}\n"


def test_generated_text_stops_before_next_user_turn():
    raw = f"{ASSISTANT_TAG}\nI am CriderGPT 2.0.\n{USER_TAG}\nWho made you?"
    assert clean_generated_text(raw) == "I am CriderGPT 2.0."


def test_generated_text_stops_at_eos_marker():
    assert clean_generated_text(f"Hello there{EOS_TAG}junk") == "Hello there"


def test_cridergpt2_checkpoint_is_separate():
    path = checkpoint_path({"checkpoint_path": "model/cridergpt-2.0/checkpoint"})
    assert path.parts[-3:] == ("model", "cridergpt-2.0", "checkpoint")
