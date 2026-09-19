from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rope import RotaryPositionalEmbeddings, apply_rotary_pos_emb


def causal_mask(sequence_length: int, device: torch.device | None = None) -> torch.Tensor:
    """Returns a [T, T] mask where True means the position is allowed to attend."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device).tril()


class SingleHeadCausalSelfAttention(nn.Module):
    """Single-head causal self-attention with RoPE applied to Q and K."""

    def __init__(self, d_model: int, head_dim: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        self.d_model = d_model
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(head_dim=head_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("x must be shaped [B, T, D]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, got {x.shape[-1]}")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k = apply_rotary_pos_emb(q, k, self.rope)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = causal_mask(x.shape[1], device=x.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ v

        if return_attention:
            return output, attention_weights
        return output


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE applied inside each head."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbeddings(head_dim=self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("x must be shaped [B, T, D]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, got {x.shape[-1]}")

        batch_size, sequence_length, _ = x.shape
        q = self._split_heads(self.q_proj(x), batch_size, sequence_length)
        k = self._split_heads(self.k_proj(x), batch_size, sequence_length)
        v = self._split_heads(self.v_proj(x), batch_size, sequence_length)
        q, k = apply_rotary_pos_emb(q, k, self.rope)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = causal_mask(sequence_length, device=x.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, sequence_length, self.d_model)
        output = self.out_proj(output)

        if return_attention:
            return output, attention_weights
        return output

    def _split_heads(
        self,
        x: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        x = x.view(batch_size, sequence_length, self.n_heads, self.head_dim)
        return x.transpose(1, 2)
