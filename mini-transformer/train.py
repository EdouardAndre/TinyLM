from __future__ import annotations

import argparse
from pathlib import Path

from data import TinyStoriesDataModule
from model import MiniTransformerLM, TransformerConfig
from training.trainer import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the mini Transformer LM.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to the YAML training config.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    project_root = args.config.resolve().parents[1]

    data_config = config["dataset"]
    data = TinyStoriesDataModule(
        train_path=_resolve_path(project_root, data_config["train_path"]),
        validation_path=_resolve_path(project_root, data_config["validation_path"]),
        text_column=data_config["text_column"],
        tokenizer_type=data_config["tokenizer"],
        vocab_size=data_config["vocab_size"],
        min_pair_frequency=data_config["min_pair_frequency"],
        context_length=data_config["context_length"],
        batch_size=data_config["batch_size"],
        max_train_chars=data_config.get("max_train_chars"),
        max_validation_chars=data_config.get("max_validation_chars"),
    )
    data.setup()

    model_config = config["model"]
    model = MiniTransformerLM(
        TransformerConfig(
            vocab_size=data.tokenizer.vocab_size,
            d_model=model_config["d_model"],
            n_layers=model_config["n_layers"],
            n_heads=model_config["n_heads"],
        )
    )

    training_config = config["training"]
    result = train(
        model,
        data.train_dataloader(),
        data.validation_dataloader(),
        TrainingConfig(
            seed=training_config["seed"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            max_steps=training_config["max_steps"],
            eval_interval=training_config["eval_interval"],
            grad_clip=training_config["grad_clip"],
            checkpoint_dir=training_config.get("checkpoint_dir", "checkpoints"),
            checkpoint_interval=training_config.get("checkpoint_interval"),
            device=training_config.get("device"),
            grad_accumulation_steps=training_config.get("grad_accumulation_steps", 1),
            precision=training_config.get("precision", "fp32"),
        ),
        checkpoint_extra_state={
            "tokenizer": data.tokenizer.state_dict(),
            "transformer_config": dict(model.config.__dict__),
            "data_config": data_config,
            "model_config": model_config,
        },
    )

    print(
        "training complete: "
        f"step={result.step} "
        f"train_loss={result.train_loss:.4f} "
        f"validation_loss={_format_optional_loss(result.validation_loss)} "
        f"tokens_per_second={result.tokens_per_second:.1f}"
    )


def _load_config(config_path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_yaml(config_path)

    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_simple_yaml(config_path: Path) -> dict:
    config: dict[str, dict] = {}
    current_section: str | None = None

    with config_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not line.startswith(" "):
                key = line.removesuffix(":")
                config[key] = {}
                current_section = key
                continue
            if current_section is None:
                raise ValueError(f"Config value without section: {raw_line.rstrip()}")

            key, value = line.strip().split(":", 1)
            config[current_section][key] = _parse_scalar(value.strip())

    return config


def _parse_scalar(value: str) -> object:
    if value in {"null", "None", ""}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _resolve_path(project_root: Path, path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (project_root / path_obj).resolve()


def _format_optional_loss(loss: float | None) -> str:
    if loss is None:
        return "n/a"
    return f"{loss:.4f}"


if __name__ == "__main__":
    main()
