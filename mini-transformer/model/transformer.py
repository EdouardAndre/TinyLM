from __future__ import annotations

import torch
from torch import nn


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
