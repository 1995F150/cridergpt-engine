"""Native tokenizer foundation for CriderGPT Image prompt conditioning.

Stage 3 intentionally does not import a pretrained tokenizer. The vocabulary is
built from the authorized caption corpus produced by Stage 2.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class CriderImageTokenizer:
    def __init__(self, vocab: dict[str, int], max_length: int = 64):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.max_length = max_length
        for token in SPECIAL_TOKENS:
            if token not in vocab:
                raise ValueError(f"Missing required special token: {token}")

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return TOKEN_RE.findall(cls.normalize(text))

    @classmethod
    def train(cls, captions: Iterable[str], vocab_size: int = 8192, max_length: int = 64):
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("vocab_size is too small")
        counts: Counter[str] = Counter()
        for caption in captions:
            counts.update(cls.tokenize(caption))
        vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if len(vocab) >= vocab_size:
                break
            if token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab, max_length=max_length)

    @property
    def pad_id(self) -> int: return self.vocab["<pad>"]
    @property
    def unk_id(self) -> int: return self.vocab["<unk>"]
    @property
    def bos_id(self) -> int: return self.vocab["<bos>"]
    @property
    def eos_id(self) -> int: return self.vocab["<eos>"]

    def encode(self, text: str, pad_to_max: bool = True) -> tuple[list[int], list[int]]:
        body = [self.vocab.get(t, self.unk_id) for t in self.tokenize(text)]
        ids = [self.bos_id] + body[: self.max_length - 2] + [self.eos_id]
        mask = [1] * len(ids)
        if pad_to_max:
            missing = self.max_length - len(ids)
            ids += [self.pad_id] * missing
            mask += [0] * missing
        return ids, mask

    def decode(self, ids: Iterable[int]) -> str:
        tokens = []
        for token_id in ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if token == "<eos>":
                break
            if token not in ("<pad>", "<bos>"):
                tokens.append(token)
        text = " ".join(tokens)
        return re.sub(r"\s+([^\w\s])", r"\1", text)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"max_length": self.max_length, "vocab": self.vocab}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocab"], max_length=int(payload["max_length"]))
