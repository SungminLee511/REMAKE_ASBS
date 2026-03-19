"""Sinusoidal time embedding."""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar time t ∈ [0, 1]."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [batch] scalar times
        Returns: [batch, embed_dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)  # [batch, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [batch, embed_dim]
