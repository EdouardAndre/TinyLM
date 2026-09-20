import torch
from torch import nn

from eval.benchmark import benchmark_cached_generation, benchmark_naive_generation


class ConstantNextTokenModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, input_ids):
        self.calls += 1
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(batch_size, sequence_length, self.vocab_size)
        logits[:, -1, 1] = 10
        return logits


class ConstantNextTokenCacheModel(ConstantNextTokenModel):
    def forward(self, input_ids, *, caches=None, use_cache=False):
        logits = super().forward(input_ids)
        if use_cache:
            batch_size, sequence_length = input_ids.shape
            past_length = 0 if caches is None else caches[0][0].shape[-2]
            cache = (
                torch.zeros(batch_size, 1, past_length + sequence_length, 1),
                torch.zeros(batch_size, 1, past_length + sequence_length, 1),
            )
            return logits, [cache]
        return logits


def test_benchmark_naive_generation_reports_timing_metrics():
    model = ConstantNextTokenModel(vocab_size=4)
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)

    result = benchmark_naive_generation(
        model,
        input_ids,
        max_new_tokens=3,
    )

    assert result.generated_tokens == 3
    assert result.total_seconds > 0
    assert result.time_to_first_token_seconds > 0
    assert result.average_decode_latency_seconds > 0
    assert result.tokens_per_second > 0
    assert model.calls == 3


def test_benchmark_naive_generation_requires_positive_new_tokens():
    model = ConstantNextTokenModel(vocab_size=4)
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)

    try:
        benchmark_naive_generation(model, input_ids, max_new_tokens=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_benchmark_cached_generation_reports_timing_metrics():
    model = ConstantNextTokenCacheModel(vocab_size=4)
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)

    result = benchmark_cached_generation(
        model,
        input_ids,
        max_new_tokens=3,
    )

    assert result.generated_tokens == 3
    assert result.total_seconds > 0
    assert result.time_to_first_token_seconds > 0
    assert result.average_decode_latency_seconds > 0
    assert result.tokens_per_second > 0
    assert model.calls == 3
