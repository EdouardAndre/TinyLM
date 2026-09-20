from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rope import RotaryPositionalEmbeddings, apply_rotary_pos_emb


KVCache = tuple[torch.Tensor, torch.Tensor]


def causal_mask(
    query_length: int | None = None,
    key_length: int | None = None,
    *,
    past_length: int = 0,
    device: torch.device | None = None,
    sequence_length: int | None = None,
) -> torch.Tensor:
    """Returns a mask where True means the query position may attend to the key."""

    if query_length is None:
        query_length = sequence_length
    if query_length is None:
        raise ValueError("query_length must be provided")
    if query_length <= 0:
        raise ValueError("query_length must be positive")
    if key_length is None:
        key_length = query_length
    if key_length <= 0:
        raise ValueError("key_length must be positive")

    query_positions = torch.arange(
        past_length,
        past_length + query_length,
        device=device,
    ).unsqueeze(-1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    return key_positions <= query_positions


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
        cache: KVCache | None = None,
        use_cache: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache] | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("x must be shaped [B, T, D]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, got {x.shape[-1]}")

        batch_size, query_length, _ = x.shape
        past_length = 0 if cache is None else cache[0].shape[-2]
        q = self._split_heads(self.q_proj(x), batch_size, query_length)
        k = self._split_heads(self.k_proj(x), batch_size, query_length)
        v = self._split_heads(self.v_proj(x), batch_size, query_length)
        q, k = apply_rotary_pos_emb(q, k, self.rope, start_position=past_length)

        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=-2)
            v = torch.cat([cached_v, v], dim=-2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = causal_mask(
            query_length,
            key_length=k.shape[-2],
            past_length=past_length,
            device=x.device,
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, query_length, self.d_model)
        output = self.out_proj(output)

        if use_cache:
            new_cache = (k, v)
            return output, new_cache
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
