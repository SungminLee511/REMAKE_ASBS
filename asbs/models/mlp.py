"""Time-dependent MLP for drift and corrector networks."""

import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbedding


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm + SiLU."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TimeDependentMLP(nn.Module):
    """
    MLP that takes (t, x) as input and outputs a vector of same dim as x.

    Architecture:
    - Sinusoidal time embedding
    - Concatenate [time_embed, x]
    - Multiple residual blocks
    - Output layer projecting to dim(x)
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        self.input_proj = nn.Linear(x_dim + time_embed_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        t: [batch] or scalar
        x: [batch, x_dim]
        Returns: [batch, x_dim]
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        t_emb = self.time_embed(t)  # [batch, time_embed_dim]
        h = torch.cat([t_emb, x], dim=-1)  # [batch, time_embed_dim + x_dim]
        h = self.input_proj(h)  # [batch, hidden_dim]

        for block in self.blocks:
            h = block(h)

        return self.output_proj(h)  # [batch, x_dim]


class CorrectorMLP(nn.Module):
    """
    Corrector network h_ϕ(x) — no time dependence (only acts at t=1).
    Zero-initialized last layer so h^(0) = 0.
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.x_dim = x_dim

        self.input_proj = nn.Linear(x_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, x_dim)

        # Zero-init last layer so h^(0) ≡ 0
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, x_dim] -> [batch, x_dim]"""
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)
