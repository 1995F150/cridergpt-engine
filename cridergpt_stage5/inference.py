#!/usr/bin/env python3
"""CriderGPT Stage 5 inference smoke-test runtime."""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent
MODEL_CONFIG = ROOT.parent / "model" / "config.json"

class Backend(Protocol):
    def generate(self, prompt: str) -> str: ...

@dataclass
class EchoBackend:
    """Smoke-test backend used until trained weights are deployed."""
    def generate(self, prompt: str) -> str:
        return f"CriderGPT model runtime received: {prompt}"

def load_config() -> dict:
    if MODEL_CONFIG.exists():
        return json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    return {"name": "CriderGPT", "stage": 5, "backend": "smoke-test"}

def generate(prompt: str, backend: Backend | None = None) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt cannot be empty")
    return (backend or EchoBackend()).generate(prompt)

def main() -> int:
    parser = argparse.ArgumentParser(description="Test the CriderGPT model runtime")
    parser.add_argument("prompt", nargs="*", help="Prompt to send to the model")
    args = parser.parse_args()
    config = load_config()
    print(f"CriderGPT runtime: {config.get('name', 'CriderGPT')}")
    print(f"Stage: {config.get('stage', 5)}")
    prompt = " ".join(args.prompt).strip() or input("Prompt: ").strip()
    print(generate(prompt))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
