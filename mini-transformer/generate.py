from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from data import TinyStoriesDataModule
from model import MiniTransformerLM, TransformerConfig
from train import _load_config, _resolve_path


@torch.no_grad()
def generate_token_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_length: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be shaped [B, T]")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must be a torch.long tensor of token IDs")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0 and not greedy:
        raise ValueError("temperature must be positive unless greedy=True")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    model.eval()
    generated = input_ids

    for _ in range(max_new_tokens):
        model_input = generated
        if context_length is not None:
            model_input = model_input[:, -context_length:]

        logits = model(model_input)
        next_token_logits = logits[:, -1, :]
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
            generator=generator,
        )
        generated = torch.cat([generated, next_token], dim=1)

    return generated


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must be shaped [B, V]")

    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        top_values, top_indices = torch.topk(logits, k=k, dim=-1)
        probabilities = F.softmax(top_values, dim=-1)
        sampled = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
        return top_indices.gather(dim=-1, index=sampled)

    probabilities = F.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the mini Transformer LM.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    project_root = args.config.resolve().parents[1]
    data_config = config["dataset"]
    model_config = config["model"]

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

    device = torch.device(args.device or config["training"].get("device") or "cpu")
    model = MiniTransformerLM(
        TransformerConfig(
            vocab_size=data.tokenizer.vocab_size,
            d_model=model_config["d_model"],
            n_layers=model_config["n_layers"],
            n_heads=model_config["n_heads"],
        )
    ).to(device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])

    input_ids = torch.tensor([data.tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output_ids = generate_token_ids(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=model_config.get("context_length"),
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )
    print(data.tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
