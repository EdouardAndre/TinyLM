from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import MiniTransformerLM, TransformerConfig
from training.trainer import TrainingConfig, evaluate, train


def _tiny_loader(vocab_size=16, batch_size=2):
    input_ids = torch.randint(0, vocab_size, (8, 4))
    targets = torch.randint(0, vocab_size, (8, 4))
    return DataLoader(TensorDataset(input_ids, targets), batch_size=batch_size)


def _tiny_model(vocab_size=16):
    return MiniTransformerLM(
        TransformerConfig(
            vocab_size=vocab_size,
            d_model=8,
            n_layers=1,
            n_heads=2,
        )
    )


def test_evaluate_returns_scalar_validation_loss():
    model = _tiny_model()
    loader = _tiny_loader()

    loss = evaluate(model, loader, torch.device("cpu"), max_batches=2)

    assert isinstance(loss, float)
    assert loss > 0


def test_train_runs_requested_number_of_steps():
    model = _tiny_model()
    train_loader = _tiny_loader()
    validation_loader = _tiny_loader()

    result = train(
        model,
        train_loader,
        validation_loader,
        TrainingConfig(
            max_steps=2,
            eval_interval=1,
            checkpoint_dir=None,
            device="cpu",
        ),
    )

    assert result.step == 2
    assert result.train_loss > 0
    assert result.validation_loss is not None


def test_train_writes_checkpoints(tmp_path):
    model = _tiny_model()
    train_loader = _tiny_loader()

    result = train(
        model,
        train_loader,
        None,
        TrainingConfig(
            max_steps=2,
            eval_interval=10,
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=2,
            device="cpu",
        ),
    )

    checkpoint_path = Path(tmp_path) / "step_000002.pt"
    assert result.step == 2
    assert checkpoint_path.exists()


def test_train_checkpoint_includes_extra_state(tmp_path):
    model = _tiny_model()
    train_loader = _tiny_loader()

    train(
        model,
        train_loader,
        None,
        TrainingConfig(
            max_steps=1,
            eval_interval=10,
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=1,
            device="cpu",
        ),
        checkpoint_extra_state={"tokenizer": {"type": "test"}},
    )

    checkpoint = torch.load(Path(tmp_path) / "step_000001.pt", map_location="cpu")
    assert checkpoint["tokenizer"] == {"type": "test"}


def test_train_supports_gradient_accumulation():
    model = _tiny_model()
    train_loader = _tiny_loader()

    result = train(
        model,
        train_loader,
        None,
        TrainingConfig(
            max_steps=2,
            eval_interval=10,
            checkpoint_dir=None,
            device="cpu",
            grad_accumulation_steps=2,
        ),
    )

    assert result.step == 2
    assert result.train_loss > 0


def test_train_validates_precision():
    model = _tiny_model()
    train_loader = _tiny_loader()

    try:
        train(
            model,
            train_loader,
            None,
            TrainingConfig(
                max_steps=1,
                checkpoint_dir=None,
                device="cpu",
                precision="fp16",
            ),
        )
    except ValueError as error:
        assert "precision" in str(error)
    else:
        raise AssertionError("Expected ValueError")
