from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    """Transformer feed-forward network applied independently to each token."""

    def __init__(self, d_model: int, expansion_factor: int = 4) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")

        hidden_dim = expansion_factor * d_model
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must be shaped [B, T, D]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, got {x.shape[-1]}")

        return self.net(x)
