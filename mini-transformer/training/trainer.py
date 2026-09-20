from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 10_000
    eval_interval: int = 500
    grad_clip: float = 1.0
    seed: int = 1337
    checkpoint_dir: str | None = "checkpoints"
    checkpoint_interval: int | None = None
    device: str | None = None


@dataclass
class TrainResult:
    step: int
    train_loss: float
    validation_loss: float | None
    tokens_per_second: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 20,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        _, loss = model(input_ids, targets)
        losses.append(loss.item())

    model.train()
    if not losses:
        raise ValueError("validation dataloader produced no batches")
    return sum(losses) / len(losses)


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_loss: float,
    validation_loss: float | None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        checkpoint_path,
    )
    return checkpoint_path


def train(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    config: TrainingConfig,
) -> TrainResult:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")

    set_seed(config.seed)
    device = torch.device(config.device) if config.device else get_default_device()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    step = 0
    last_loss = 0.0
    validation_loss: float | None = None
    start_time = perf_counter()

    while step < config.max_steps:
        for input_ids, targets in train_loader:
            if step >= config.max_steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, targets)
            loss.backward()

            if config.grad_clip is not None and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

            step += 1
            last_loss = loss.item()

            should_eval = validation_loader is not None and step % config.eval_interval == 0
            if should_eval:
                validation_loss = evaluate(model, validation_loader, device)

            should_checkpoint = (
                config.checkpoint_dir is not None
                and config.checkpoint_interval is not None
                and config.checkpoint_interval > 0
                and step % config.checkpoint_interval == 0
            )
            if should_checkpoint:
                save_checkpoint(
                    config.checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=last_loss,
                    validation_loss=validation_loss,
                )

        else:
            continue
        break

    elapsed = max(perf_counter() - start_time, 1e-9)
    tokens_seen = _count_tokens_seen(train_loader, steps=step)
    return TrainResult(
        step=step,
        train_loss=last_loss,
        validation_loss=validation_loss,
        tokens_per_second=tokens_seen / elapsed,
    )


def _count_tokens_seen(train_loader: DataLoader, *, steps: int) -> int:
    batch_size = train_loader.batch_size
    dataset = train_loader.dataset
    if batch_size is None or not hasattr(dataset, "context_length"):
        return 0
    return int(batch_size) * int(dataset.context_length) * steps
