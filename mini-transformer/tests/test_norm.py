import pytest
import torch

from model import RMSNorm


def test_rms_norm_preserves_shape():
    x = torch.randn(2, 5, 8)
    norm = RMSNorm(d_model=8)

    output = norm(x)

    assert output.shape == torch.Size([2, 5, 8])


def test_rms_norm_sets_unit_rms_when_weight_is_one():
    x = torch.randn(2, 5, 8)
    norm = RMSNorm(d_model=8, eps=1e-12)

    output = norm(x)

    rms = torch.sqrt(output.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rms_norm_has_learned_scale_parameter():
    norm = RMSNorm(d_model=8)

    assert norm.weight.shape == torch.Size([8])
    assert norm.weight.requires_grad


def test_rms_norm_scale_parameter_changes_output():
    x = torch.ones(1, 1, 4)
    norm = RMSNorm(d_model=4, eps=1e-12)

    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    output = norm(x)

    assert torch.allclose(output, torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))


def test_rms_norm_validates_input_shape():
    norm = RMSNorm(d_model=8)

    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        norm(torch.randn(2, 8))


def test_rms_norm_validates_model_dimension():
    norm = RMSNorm(d_model=8)

    with pytest.raises(ValueError, match="last dimension"):
        norm(torch.randn(2, 5, 6))
