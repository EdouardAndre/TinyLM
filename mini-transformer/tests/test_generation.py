import torch
from torch import nn

from generate import generate_token_ids, generate_token_ids_with_cache, sample_next_token
from model import MiniTransformerLM, TransformerConfig


class IncrementingDummyModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_lengths = []

    def forward(self, input_ids):
        self.seen_lengths.append(input_ids.shape[1])
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(batch_size, sequence_length, self.vocab_size)
        next_ids = (input_ids[:, -1] + 1) % self.vocab_size
        logits[:, -1, :] = -100
        logits[torch.arange(batch_size), -1, next_ids] = 100
        return logits


def test_sample_next_token_greedy_chooses_highest_logit():
    logits = torch.tensor([[0.1, 2.0, 0.3]])

    next_token = sample_next_token(logits, greedy=True)

    assert next_token.tolist() == [[1]]


def test_sample_next_token_respects_top_k_candidates():
    logits = torch.tensor([[10.0, 9.0, -100.0]])
    generator = torch.Generator().manual_seed(0)

    next_token = sample_next_token(logits, top_k=2, generator=generator)

    assert next_token.item() in {0, 1}


def test_generate_token_ids_appends_new_tokens_greedily():
    model = IncrementingDummyModel(vocab_size=10)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    output = generate_token_ids(
        model,
        input_ids,
        max_new_tokens=3,
        greedy=True,
    )

    assert output.tolist() == [[1, 2, 3, 4, 5]]


def test_generate_token_ids_crops_to_context_length():
    model = IncrementingDummyModel(vocab_size=10)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    generate_token_ids(
        model,
        input_ids,
        max_new_tokens=2,
        context_length=3,
        greedy=True,
    )

    assert model.seen_lengths == [3, 3]


def test_cached_greedy_generation_matches_uncached_greedy_generation():
    torch.manual_seed(0)
    model = MiniTransformerLM(
        TransformerConfig(
            vocab_size=16,
            d_model=8,
            n_layers=2,
            n_heads=2,
        )
    )
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    uncached = generate_token_ids(
        model,
        input_ids,
        max_new_tokens=4,
        greedy=True,
    )
    cached = generate_token_ids_with_cache(
        model,
        input_ids,
        max_new_tokens=4,
        greedy=True,
    )

    assert torch.equal(cached, uncached)
