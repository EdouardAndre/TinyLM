import pytest
import torch

from model import TransformerBlock
from model.attention import MultiHeadCausalSelfAttention
from model.mlp import FeedForward
from model.norm import RMSNorm


def test_transformer_block_preserves_shape():
    x = torch.randn(2, 5, 8)
    block = TransformerBlock(d_model=8, n_heads=2)

    output = block(x)

    assert output.shape == torch.Size([2, 5, 8])


def test_transformer_block_uses_expected_sublayers():
    block = TransformerBlock(d_model=8, n_heads=2)

    assert isinstance(block.attn_norm, RMSNorm)
    assert isinstance(block.attn, MultiHeadCausalSelfAttention)
    assert isinstance(block.mlp_norm, RMSNorm)
    assert isinstance(block.mlp, FeedForward)


def test_transformer_block_changes_values_but_not_shape():
    x = torch.randn(2, 5, 8)
    block = TransformerBlock(d_model=8, n_heads=2)

    output = block(x)

    assert output.shape == x.shape
    assert not torch.allclose(output, x)


def test_transformer_block_supports_backpropagation():
    x = torch.randn(2, 5, 8, requires_grad=True)
    block = TransformerBlock(d_model=8, n_heads=2)

    loss = block(x).sum()
    loss.backward()

    assert x.grad is not None
    assert block.attn.q_proj.weight.grad is not None
    assert block.mlp.net[0].weight.grad is not None


def test_transformer_block_validates_input_shape():
    block = TransformerBlock(d_model=8, n_heads=2)

    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        block(torch.randn(2, 8))


def test_transformer_block_validates_model_dimension():
    block = TransformerBlock(d_model=8, n_heads=2)

    with pytest.raises(ValueError, match="last dimension"):
        block(torch.randn(2, 5, 6))
