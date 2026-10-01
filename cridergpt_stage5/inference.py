#!/usr/bin/env python3
"""CriderGPT local inference runtime for versioned native checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
MODEL_CONFIG = REPO_ROOT / "model" / "config.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "model" / "cridergpt-2.0" / "checkpoint"
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
EOS_TAG = "<|eos|>"


class Backend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class EchoBackend:
    reason: str = "trained checkpoint not available"

    def generate(self, prompt: str) -> str:
        return f"CriderGPT model runtime received: {prompt}"


def format_chat_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt cannot be empty")
    return f"{USER_TAG}\n{prompt}\n{ASSISTANT_TAG}\n"


def clean_generated_text(text: str) -> str:
    """Return only the assistant turn and remove leaked conversation markers."""
    cleaned = text.strip()
    if cleaned.startswith(ASSISTANT_TAG):
        cleaned = cleaned[len(ASSISTANT_TAG):].lstrip()
    stop_markers = (USER_TAG, ASSISTANT_TAG, EOS_TAG, "<|bos|>", "<|pad|>")
    positions = [cleaned.find(marker) for marker in stop_markers if cleaned.find(marker) >= 0]
    if positions:
        cleaned = cleaned[: min(positions)]
    return cleaned.strip()


class TransformersBackend:
    """Local Hugging Face Transformers causal-language-model backend."""

    def __init__(self, checkpoint: Path, max_new_tokens: int = 128) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> str:
        rendered = format_chat_prompt(prompt)
        inputs = self.tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][input_length:]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        return clean_generated_text(decoded)


def load_config() -> dict:
    if MODEL_CONFIG.exists():
        return json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    return {
        "name": "CriderGPT 2.0",
        "family": "CriderGPT Native",
        "version": "2.0.0",
        "stage": 6,
        "checkpoint_path": "model/cridergpt-2.0/checkpoint",
    }


def checkpoint_path(config: dict) -> Path:
    configured = os.environ.get("CRIDERGPT_CHECKPOINT") or config.get("checkpoint_path")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path
    return DEFAULT_CHECKPOINT


def load_backend(config: dict, max_new_tokens: int = 128) -> Backend:
    checkpoint = checkpoint_path(config)
    if not checkpoint.exists():
        print(f"Backend: smoke-test (checkpoint not found: {checkpoint})")
        return EchoBackend("checkpoint directory not found")

    try:
        backend = TransformersBackend(checkpoint, max_new_tokens=max_new_tokens)
        print(f"Backend: local Transformers model ({checkpoint})")
        print(f"Device: {backend.device}")
        return backend
    except Exception as exc:
        print(f"Backend: smoke-test (checkpoint failed to load: {exc})")
        return EchoBackend("checkpoint failed to load")


def generate(prompt: str, backend: Backend | None = None) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt cannot be empty")
    return (backend or EchoBackend()).generate(prompt)


def interactive_chat(backend: Backend, model_name: str) -> None:
    print(f"\n{model_name} Local AI")
    print("Type /exit to quit.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            return
        try:
            answer = generate(prompt, backend)
            print(f"CriderGPT: {answer or '[empty response]'}\n")
        except Exception as exc:
            print(f"CriderGPT error: {exc}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CriderGPT locally")
    parser.add_argument("prompt", nargs="*", help="One prompt to send to the model")
    parser.add_argument("--chat", action="store_true", help="Start interactive terminal chat")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    config = load_config()
    model_name = config.get("name", "CriderGPT 2.0")
    print(f"CriderGPT runtime: {model_name}")
    print(f"Stage: {config.get('stage', 6)}")
    backend = load_backend(config, max_new_tokens=args.max_new_tokens)

    prompt = " ".join(args.prompt).strip()
    if args.chat or not prompt:
        interactive_chat(backend, model_name)
    else:
        print(f"CriderGPT: {generate(prompt, backend)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
