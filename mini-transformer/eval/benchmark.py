from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import TinyStoriesDataModule
from generate import sample_next_token
from model import MiniTransformerLM, TransformerConfig
from train import _load_config, _resolve_path


@dataclass(frozen=True)
class GenerationBenchmarkResult:
    generated_tokens: int
    total_seconds: float
    time_to_first_token_seconds: float
    average_decode_latency_seconds: float
    tokens_per_second: float
    peak_memory_bytes: int | None


@torch.no_grad()
def benchmark_naive_generation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_length: int | None = None,
    greedy: bool = True,
) -> GenerationBenchmarkResult:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be shaped [B, T]")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must be a torch.long tensor of token IDs")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    device = input_ids.device
    model.eval()
    generated = input_ids
    decode_latencies: list[float] = []

    _reset_peak_memory(device)
    _synchronize(device)
    total_start = perf_counter()

    first_token_start = perf_counter()
    generated = _decode_one_token(
        model,
        generated,
        context_length=context_length,
        greedy=greedy,
    )
    _synchronize(device)
    time_to_first_token = perf_counter() - first_token_start

    for _ in range(max_new_tokens - 1):
        decode_start = perf_counter()
        generated = _decode_one_token(
            model,
            generated,
            context_length=context_length,
            greedy=greedy,
        )
        _synchronize(device)
        decode_latencies.append(perf_counter() - decode_start)

    total_seconds = perf_counter() - total_start
    average_decode_latency = (
        sum(decode_latencies) / len(decode_latencies) if decode_latencies else 0.0
    )
    return GenerationBenchmarkResult(
        generated_tokens=max_new_tokens,
        total_seconds=total_seconds,
        time_to_first_token_seconds=time_to_first_token,
        average_decode_latency_seconds=average_decode_latency,
        tokens_per_second=max_new_tokens / max(total_seconds, 1e-9),
        peak_memory_bytes=_peak_memory(device),
    )


@torch.no_grad()
def benchmark_cached_generation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_length: int | None = None,
    greedy: bool = True,
) -> GenerationBenchmarkResult:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be shaped [B, T]")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must be a torch.long tensor of token IDs")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    device = input_ids.device
    model.eval()
    generated = input_ids
    model_input = generated if context_length is None else generated[:, -context_length:]
    decode_latencies: list[float] = []

    _reset_peak_memory(device)
    _synchronize(device)
    total_start = perf_counter()

    first_token_start = perf_counter()
    logits, caches = model(model_input, use_cache=True)
    next_token = sample_next_token(logits[:, -1, :], greedy=greedy)
    generated = torch.cat([generated, next_token], dim=1)
    _synchronize(device)
    time_to_first_token = perf_counter() - first_token_start

    for _ in range(max_new_tokens - 1):
        if context_length is not None:
            caches = _crop_caches(caches, max_length=max(context_length - 1, 0))

        decode_start = perf_counter()
        logits, caches = model(next_token, caches=caches, use_cache=True)
        next_token = sample_next_token(logits[:, -1, :], greedy=greedy)
        generated = torch.cat([generated, next_token], dim=1)
        _synchronize(device)
        decode_latencies.append(perf_counter() - decode_start)

    total_seconds = perf_counter() - total_start
    average_decode_latency = (
        sum(decode_latencies) / len(decode_latencies) if decode_latencies else 0.0
    )
    return GenerationBenchmarkResult(
        generated_tokens=max_new_tokens,
        total_seconds=total_seconds,
        time_to_first_token_seconds=time_to_first_token,
        average_decode_latency_seconds=average_decode_latency,
        tokens_per_second=max_new_tokens / max(total_seconds, 1e-9),
        peak_memory_bytes=_peak_memory(device),
    )


def _decode_one_token(
    model: torch.nn.Module,
    generated: torch.Tensor,
    *,
    context_length: int | None,
    greedy: bool,
) -> torch.Tensor:
    model_input = generated
    if context_length is not None:
        model_input = model_input[:, -context_length:]

    logits = model(model_input)
    next_token = sample_next_token(logits[:, -1, :], greedy=greedy)
    return torch.cat([generated, next_token], dim=1)


def _crop_caches(caches, *, max_length: int):
    if max_length <= 0:
        return [(k[:, :, :0, :], v[:, :, :0, :]) for k, v in caches]
    return [(k[:, :, -max_length:, :], v[:, :, -max_length:, :]) for k, v in caches]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark naive Transformer generation.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=32)
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
    naive_result = benchmark_naive_generation(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=model_config.get("context_length"),
    )
    cached_result = benchmark_cached_generation(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=model_config.get("context_length"),
    )
    _print_result("naive", naive_result)
    _print_result("cached", cached_result)
    print(
        "speedup: "
        f"{cached_result.tokens_per_second / max(naive_result.tokens_per_second, 1e-9):.2f}x"
    )


def _print_result(name: str, result: GenerationBenchmarkResult) -> None:
    print(f"{name}.generated_tokens: {result.generated_tokens}")
    print(f"{name}.time_to_first_token_seconds: {result.time_to_first_token_seconds:.6f}")
    print(f"{name}.average_decode_latency_seconds: {result.average_decode_latency_seconds:.6f}")
    print(f"{name}.tokens_per_second: {result.tokens_per_second:.2f}")
    print(f"{name}.total_seconds: {result.total_seconds:.6f}")
    if result.peak_memory_bytes is not None:
        print(f"{name}.peak_memory_mb: {result.peak_memory_bytes / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
