import pytest
import torch

from model import SingleHeadCausalSelfAttention, causal_mask


def test_causal_mask_allows_current_and_previous_positions_only():
    mask = causal_mask(sequence_length=4)

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_single_head_attention_maps_model_dim_to_head_dim():
    batch_size = 2
    sequence_length = 5
    d_model = 8
    head_dim = 4
    x = torch.randn(batch_size, sequence_length, d_model)
    attention = SingleHeadCausalSelfAttention(d_model=d_model, head_dim=head_dim)

    output = attention(x)

    assert output.shape == torch.Size([batch_size, sequence_length, head_dim])


def test_single_head_attention_can_return_attention_weights():
    batch_size = 2
    sequence_length = 5
    d_model = 8
    head_dim = 4
    x = torch.randn(batch_size, sequence_length, d_model)
    attention = SingleHeadCausalSelfAttention(d_model=d_model, head_dim=head_dim)

    output, attention_weights = attention(x, return_attention=True)

    assert output.shape == torch.Size([batch_size, sequence_length, head_dim])
    assert attention_weights.shape == torch.Size([batch_size, sequence_length, sequence_length])


def test_single_head_attention_masks_future_positions():
    sequence_length = 5
    x = torch.randn(2, sequence_length, 8)
    attention = SingleHeadCausalSelfAttention(d_model=8, head_dim=4)

    _, attention_weights = attention(x, return_attention=True)

    future_positions = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool),
        diagonal=1,
    )
    assert torch.all(attention_weights[:, future_positions] == 0)


def test_single_head_attention_weights_sum_to_one_over_allowed_positions():
    x = torch.randn(2, 5, 8)
    attention = SingleHeadCausalSelfAttention(d_model=8, head_dim=4)

    _, attention_weights = attention(x, return_attention=True)

    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(2, 5))


def test_single_head_attention_validates_input_shape():
    attention = SingleHeadCausalSelfAttention(d_model=8, head_dim=4)

    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        attention(torch.randn(2, 8))
