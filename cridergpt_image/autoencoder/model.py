"""CriderGPT Image Stage 4 latent autoencoder.

This module defines an untrained, CriderGPT-owned convolutional encoder/decoder.
It does not load or substitute third-party pretrained image-model weights.
"""
from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder(nn.Module):
    """256x256 RGB -> 4x32x32 latent by default."""
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            ResidualBlock(base),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            ResidualBlock(base * 2),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            ResidualBlock(base * 4),
            nn.Conv2d(base * 4, base * 4, 4, stride=2, padding=1),
            ResidualBlock(base * 4),
            nn.GroupNorm(_groups(base * 4), base * 4),
            nn.SiLU(),
            nn.Conv2d(base * 4, latent_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """4x32x32 latent -> 256x256 RGB by default."""
    def __init__(self, out_channels: int = 3, latent_channels: int = 4, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, base * 4, 3, padding=1),
            ResidualBlock(base * 4),
            nn.ConvTranspose2d(base * 4, base * 4, 4, stride=2, padding=1),
            ResidualBlock(base * 4),
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            ResidualBlock(base * 2),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            ResidualBlock(base),
            nn.GroupNorm(_groups(base), base),
            nn.SiLU(),
            nn.Conv2d(base, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CriderImageAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, base)
        self.decoder = Decoder(in_channels, latent_channels, base)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(images)
        reconstruction = self.decode(latent)
        return reconstruction, latent
