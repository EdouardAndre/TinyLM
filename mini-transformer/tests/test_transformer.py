import pytest
import torch

from model import MiniTransformerLM, TransformerBlock, TransformerConfig


def test_mini_transformer_lm_maps_token_ids_to_logits():
    config = TransformerConfig(vocab_size=32, d_model=8, n_layers=2, n_heads=2)
    model = MiniTransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 5))

    logits = model(input_ids)

    assert logits.shape == torch.Size([3, 5, config.vocab_size])


def test_mini_transformer_lm_builds_requested_number_of_blocks():
    config = TransformerConfig(vocab_size=32, d_model=8, n_layers=3, n_heads=2)
    model = MiniTransformerLM(config)

    assert len(model.blocks) == 3
    assert all(isinstance(block, TransformerBlock) for block in model.blocks)


def test_mini_transformer_lm_can_compute_next_token_loss():
    config = TransformerConfig(vocab_size=32, d_model=8, n_layers=2, n_heads=2)
    model = MiniTransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 5))
    targets = torch.randint(0, config.vocab_size, (3, 5))

    logits, loss = model(input_ids, targets=targets)

    assert logits.shape == torch.Size([3, 5, config.vocab_size])
    assert loss.ndim == 0


def test_mini_transformer_lm_ties_embedding_and_lm_head_weights_by_default():
    config = TransformerConfig(vocab_size=32, d_model=8, n_layers=1, n_heads=2)
    model = MiniTransformerLM(config)

    assert model.lm_head.weight is model.token_embeddings.embedding.weight


def test_mini_transformer_lm_can_disable_weight_tying():
    config = TransformerConfig(
        vocab_size=32,
        d_model=8,
        n_layers=1,
        n_heads=2,
        tie_embeddings=False,
    )
    model = MiniTransformerLM(config)

    assert model.lm_head.weight is not model.token_embeddings.embedding.weight


def test_mini_transformer_lm_validates_input_shape():
    config = TransformerConfig(vocab_size=32, d_model=8, n_layers=1, n_heads=2)
    model = MiniTransformerLM(config)

    with pytest.raises(ValueError, match=r"\[B, T\]"):
        model(torch.randint(0, config.vocab_size, (2, 3, 4)))


def test_mini_transformer_lm_validates_target_shape():
    config = TransformerConfig(vocab_size=32, d_model=8, n_layers=1, n_heads=2)
    model = MiniTransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 3))

    with pytest.raises(ValueError, match="same shape"):
        model(input_ids, targets=targets)
