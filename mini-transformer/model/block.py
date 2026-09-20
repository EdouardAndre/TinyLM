from __future__ import annotations

import torch
from torch import nn

from .attention import KVCache, MultiHeadCausalSelfAttention
from .mlp import FeedForward
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm decoder Transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")

        self.d_model = d_model
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = FeedForward(
            d_model=d_model,
            expansion_factor=mlp_expansion_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        if x.ndim != 3:
            raise ValueError("x must be shaped [B, T, D]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, got {x.shape[-1]}")

        if use_cache:
            attn_output, new_cache = self.attn(
                self.attn_norm(x),
                cache=cache,
                use_cache=True,
            )
            x = x + attn_output
            x = x + self.mlp(self.mlp_norm(x))
            return x, new_cache

        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
