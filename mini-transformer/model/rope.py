from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """Applies RoPE to query/key tensors.

    Expected input shape:
      [B, T, Dh] or [B, H, T, Dh]

    The last dimension is rotated in pairs, so `head_dim` must be even.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        start_position: int = 0,
    ) -> torch.Tensor:
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected last dimension {self.head_dim}, got {x.shape[-1]}"
            )
        if x.ndim not in (3, 4):
            raise ValueError("RoPE expects input shaped [B, T, Dh] or [B, H, T, Dh]")

        sequence_length = x.shape[-2]
        if positions is None:
            positions = torch.arange(
                start_position,
                start_position + sequence_length,
                device=x.device,
            )
        elif positions.shape != torch.Size([sequence_length]):
            raise ValueError(
                f"positions must have shape [{sequence_length}], got {tuple(positions.shape)}"
            )

        angles = torch.outer(positions.to(self.inv_freq.dtype), self.inv_freq)
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)

        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos
        return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryPositionalEmbeddings,
    *,
    positions: torch.Tensor | None = None,
    start_position: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies the same RoPE positions to queries and keys."""

    return (
        rope(q, positions=positions, start_position=start_position),
        rope(k, positions=positions, start_position=start_position),
    )
