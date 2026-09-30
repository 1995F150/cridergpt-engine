"""Transformer text encoder for CriderGPT Image Stage 3."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class CriderImageTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 64,
        embedding_dim: int = 256,
        layers: int = 4,
        heads: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads")
        self.max_length = max_length
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, seq = input_ids.shape
        if seq > self.max_length:
            raise ValueError(f"sequence length {seq} exceeds max_length {self.max_length}")
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, seq)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        if attention_mask is None:
            padding_mask = input_ids.eq(self.pad_id)
        else:
            padding_mask = attention_mask.eq(0)
        return self.final_norm(self.encoder(hidden, src_key_padding_mask=padding_mask))


def build_from_config(config_path: str | Path, actual_vocab_size: int) -> CriderImageTextEncoder:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return CriderImageTextEncoder(
        vocab_size=actual_vocab_size,
        max_length=cfg["max_prompt_tokens"],
        embedding_dim=cfg["embedding_dim"],
        layers=cfg["encoder_layers"],
        heads=cfg["attention_heads"],
        feedforward_dim=cfg["feedforward_dim"],
        dropout=cfg["dropout"],
    )
