import pytest
import torch

from model import TokenEmbeddings


def test_token_embeddings_map_batch_by_time_to_batch_by_time_by_model_dim():
    batch_size = 2
    sequence_length = 4
    vocab_size = 10
    d_model = 8
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    embeddings = TokenEmbeddings(vocab_size=vocab_size, d_model=d_model)

    output = embeddings(input_ids)

    assert output.shape == torch.Size([batch_size, sequence_length, d_model])


def test_same_token_id_uses_same_embedding_vector():
    embeddings = TokenEmbeddings(vocab_size=5, d_model=3)
    input_ids = torch.tensor([[2, 2]], dtype=torch.long)

    output = embeddings(input_ids)

    assert torch.equal(output[0, 0], output[0, 1])


def test_token_embeddings_require_integer_token_ids():
    embeddings = TokenEmbeddings(vocab_size=5, d_model=3)
    input_ids = torch.tensor([[1.0, 2.0]])

    with pytest.raises(TypeError, match="torch.long"):
        embeddings(input_ids)
