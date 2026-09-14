import pytest
import torch

from model import RotaryPositionalEmbeddings, apply_rotary_pos_emb


def test_rope_preserves_single_head_shape():
    batch_size = 2
    sequence_length = 4
    head_dim = 8
    x = torch.randn(batch_size, sequence_length, head_dim)
    rope = RotaryPositionalEmbeddings(head_dim=head_dim)

    output = rope(x)

    assert output.shape == torch.Size([batch_size, sequence_length, head_dim])


def test_rope_preserves_multi_head_shape():
    batch_size = 2
    n_heads = 3
    sequence_length = 4
    head_dim = 8
    x = torch.randn(batch_size, n_heads, sequence_length, head_dim)
    rope = RotaryPositionalEmbeddings(head_dim=head_dim)

    output = rope(x)

    assert output.shape == torch.Size([batch_size, n_heads, sequence_length, head_dim])


def test_rope_rotates_same_vector_differently_at_different_positions():
    head_dim = 4
    token_vector = torch.tensor([1.0, 0.0, 1.0, 0.0])
    x = token_vector.repeat(1, 3, 1)
    rope = RotaryPositionalEmbeddings(head_dim=head_dim)

    output = rope(x)

    assert torch.allclose(output[0, 0], token_vector)
    assert not torch.allclose(output[0, 1], output[0, 2])


def test_rope_preserves_pairwise_vector_norms():
    x = torch.randn(2, 5, 8)
    rope = RotaryPositionalEmbeddings(head_dim=8)

    output = rope(x)

    input_pair_norms = x.reshape(2, 5, 4, 2).norm(dim=-1)
    output_pair_norms = output.reshape(2, 5, 4, 2).norm(dim=-1)
    assert torch.allclose(output_pair_norms, input_pair_norms, atol=1e-6)


def test_apply_rotary_pos_emb_rotates_queries_and_keys():
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    rope = RotaryPositionalEmbeddings(head_dim=4)

    q_rotated, k_rotated = apply_rotary_pos_emb(q, k, rope)

    assert q_rotated.shape == q.shape
    assert k_rotated.shape == k.shape


def test_rope_requires_even_head_dim():
    with pytest.raises(ValueError, match="even"):
        RotaryPositionalEmbeddings(head_dim=3)
