"""CriderGPT Image Stage 5 text-conditioned latent denoiser."""
from __future__ import annotations
import math
import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float()[:, None] * freq[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2: emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class SpatialTextAttention(nn.Module):
    def __init__(self, channels: int, text_dim: int, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.text_proj = nn.Linear(text_dim, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x, text, text_mask=None):
        b, c, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)
        qn = self.norm(q)
        kv = self.text_proj(text)
        key_padding_mask = None if text_mask is None else text_mask.eq(0)
        out, _ = self.attn(qn, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return (q + out).transpose(1, 2).reshape(b, c, h, w)


class CriderImageLatentGenerator(nn.Module):
    """Predict noise for a [B,4,32,32] latent conditioned on Stage 3 text embeddings."""
    def __init__(self, latent_channels=4, base=128, text_dim=256, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.input = nn.Conv2d(latent_channels, base, 3, padding=1)
        self.block1 = ResBlock(base, time_dim)
        self.attn1 = SpatialTextAttention(base, text_dim)
        self.block2 = ResBlock(base, time_dim)
        self.attn2 = SpatialTextAttention(base, text_dim)
        self.out_norm = nn.GroupNorm(32, base)
        self.out = nn.Conv2d(base, latent_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, noisy_latent, timesteps, text_embeddings, text_mask=None):
        temb = self.time_mlp(timestep_embedding(timesteps, self.time_dim))
        h = self.input(noisy_latent)
        h = self.block1(h, temb)
        h = self.attn1(h, text_embeddings, text_mask)
        h = self.block2(h, temb)
        h = self.attn2(h, text_embeddings, text_mask)
        return self.out(self.act(self.out_norm(h)))
