import pytest
import torch
from torch import nn

from model import FeedForward


def test_feed_forward_preserves_batch_time_and_model_dimensions():
    batch_size = 2
    sequence_length = 5
    d_model = 8
    x = torch.randn(batch_size, sequence_length, d_model)
    mlp = FeedForward(d_model=d_model)

    output = mlp(x)

    assert output.shape == torch.Size([batch_size, sequence_length, d_model])


def test_feed_forward_expands_hidden_dimension_by_default():
    d_model = 8
    mlp = FeedForward(d_model=d_model)

    first_linear = mlp.net[0]
    second_linear = mlp.net[2]

    assert isinstance(first_linear, nn.Linear)
    assert isinstance(second_linear, nn.Linear)
    assert first_linear.in_features == d_model
    assert first_linear.out_features == 4 * d_model
    assert second_linear.in_features == 4 * d_model
    assert second_linear.out_features == d_model


def test_feed_forward_uses_gelu_activation():
    mlp = FeedForward(d_model=8)

    assert isinstance(mlp.net[1], nn.GELU)


def test_feed_forward_supports_custom_expansion_factor():
    mlp = FeedForward(d_model=8, expansion_factor=2)

    assert mlp.hidden_dim == 16


def test_feed_forward_validates_input_shape():
    mlp = FeedForward(d_model=8)

    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        mlp(torch.randn(2, 8))


def test_feed_forward_validates_model_dimension():
    mlp = FeedForward(d_model=8)

    with pytest.raises(ValueError, match="last dimension"):
        mlp(torch.randn(2, 5, 6))
