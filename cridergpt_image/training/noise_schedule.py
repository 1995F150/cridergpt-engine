"""Diffusion/noise schedule utilities for CriderGPT Image Stage 6."""
from __future__ import annotations
import torch

class LinearNoiseSchedule:
    def __init__(self, steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        if steps < 2:
            raise ValueError("steps must be >= 2")
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[timesteps].view(-1, 1, 1, 1)
        return a_bar.sqrt() * clean + (1.0 - a_bar).sqrt() * noise
