from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attention import KVCache
from .block import TransformerBlock
from .norm import RMSNorm


class TokenEmbeddings(nn.Module):
    """Maps integer token IDs into learned D-dimensional vectors."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be a torch.long tensor of token IDs")

        return self.embedding(input_ids)


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_expansion_factor: int = 4
    tie_embeddings: bool = True


class MiniTransformerLM(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if config.d_model <= 0:
            raise ValueError("d_model must be positive")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if config.n_heads <= 0:
            raise ValueError("n_heads must be positive")

        self.config = config
        self.token_embeddings = TokenEmbeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_expansion_factor=config.mlp_expansion_factor,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        caches: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, list[KVCache]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shaped [B, T]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be a torch.long tensor of token IDs")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if use_cache and targets is not None:
            raise ValueError("targets are not supported when use_cache=True")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("caches must have one entry per Transformer block")

        x = self.token_embeddings(input_ids)
        new_caches: list[KVCache] = []
        for block_idx, block in enumerate(self.blocks):
            if use_cache:
                block_cache = None if caches is None else caches[block_idx]
                x, new_cache = block(x, cache=block_cache, use_cache=True)
                new_caches.append(new_cache)
            else:
                x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_caches
        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.reshape(-1),
        )
        return logits, loss
