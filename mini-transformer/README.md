# Mini Transformer

Decoder-only Transformer language model implemented from scratch in PyTorch.

The goal is not to train a large model. The goal is to understand and implement the core mechanics behind modern LLMs: tokenization, embeddings, RoPE, causal attention, Transformer blocks, training, autoregressive generation, and KV-cache inference.

## Architecture

```text
input_ids: [B, T]
-> BPE token embeddings: [B, T, D]
-> TransformerBlock x n_layers
   -> RMSNorm
   -> multi-head causal self-attention with RoPE
   -> residual add
   -> RMSNorm
   -> MLP: D -> 4D -> GELU -> D
   -> residual add
-> final RMSNorm
-> LM head
-> logits: [B, T, V]
```

Where:

```text
B  = batch size
T  = sequence length
D  = model dimension
V  = vocabulary size
H  = number of attention heads
Dh = D / H
```

## Implemented

- BPE-style tokenizer trained from the local CSV corpus
- optional Rust-backed BPE training path for full-dataset runs
- next-token language-modeling dataset
- token embeddings
- RoPE applied to Q/K
- single-head attention for learning/debugging
- multi-head causal self-attention
- RMSNorm
- MLP/feed-forward block
- pre-norm Transformer decoder block
- full decoder-only language model
- AdamW training loop
- validation loss
- checkpoint saving
- gradient accumulation
- optional BF16 autocast hook
- greedy, temperature, and top-k generation
- KV-cache generation
- naive vs cached generation benchmark
- 68 unit tests

## What Is Not Used

- `nn.Transformer`
- Hugging Face Trainer
- FlashAttention
- distributed training
- pretrained tokenizers
- custom CUDA kernels

## Run

Use the PyTorch-enabled Python from the repository root:

```bash
/Users/eda/.pyenv/versions/3.11.7/bin/python3 mini-transformer/train.py --config mini-transformer/configs/smoke.yaml
```

For a more meaningful local run:

```bash
/Users/eda/.pyenv/versions/3.11.7/bin/python3 mini-transformer/train.py --config mini-transformer/configs/local_experiment.yaml
```

The larger default run is:

```bash
/Users/eda/.pyenv/versions/3.11.7/bin/python3 mini-transformer/train.py --config mini-transformer/configs/base.yaml
```

## Cloud GPU Run

For a rented GPU machine, clone the repo, place the dataset at `archive/train.csv` and
`archive/validation.csv`, then run:

```bash
bash scripts/setup_cloud_gpu.sh
```

That script creates a local virtual environment, installs the project dependencies,
checks that CUDA is visible, and launches:

```bash
python mini-transformer/train.py --config mini-transformer/configs/cloud_gpu.yaml
```

The cloud config uses:

```text
GPU device: cuda
Precision: BF16 autocast
Tokenizer: BPE, vocab size 1024
Tokenizer cache: tokenizers/tinystories_bpe_1024.json
Context length: 256
Batch size: 64
Gradient accumulation: 2
Effective batch size: 128 sequences
Model: 8 layers, 8 heads, D=384
Training: 20,000 optimizer steps
```

Recommended rental target: start with one RTX 4090 on Vast.ai or RunPod. H100/H200 is
much faster, but this model is small enough that the expensive cards are mostly useful
if you want results quickly rather than cheaply.

## Generation

With random or trained weights:

```bash
/Users/eda/.pyenv/versions/3.11.7/bin/python3 mini-transformer/generate.py \
  --config mini-transformer/configs/smoke.yaml \
  --prompt "Once upon a time" \
  --max-new-tokens 40 \
  --greedy
```

With a checkpoint:

```bash
/Users/eda/.pyenv/versions/3.11.7/bin/python3 mini-transformer/generate.py \
  --config mini-transformer/configs/base.yaml \
  --checkpoint checkpoints/step_003000.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 40 \
  --use-cache
```

New checkpoints include tokenizer state, so generation can load the tokenizer directly instead of retraining BPE from CSV.

## Benchmark

```bash
/Users/eda/.pyenv/versions/3.11.7/bin/python3 mini-transformer/eval/benchmark.py \
  --config mini-transformer/configs/smoke.yaml \
  --prompt "Once upon" \
  --max-new-tokens 4
```

Smoke benchmark on Mac:

| Mode | Tokens/s | TTFT | Avg Decode Latency |
| --- | ---: | ---: | ---: |
| Naive | 1463.44 | 0.001140s | 0.000530s |
| KV cache | 1356.74 | 0.000540s | 0.000793s |

The speedup is noisy on this tiny smoke model. KV cache becomes more important with longer contexts, larger models, and more generated tokens.

## Training Snapshot

A local `base.yaml` run produced these checkpoint losses:

| Step | Train Loss | Validation Loss |
| ---: | ---: | ---: |
| 1000 | 2.0093 | 2.2382 |
| 2000 | 1.5105 | 2.0616 |
| 3000 | 1.3373 | 1.6833 |
| 4000 | 1.0278 | 1.9417 |
| 5000 | 0.8082 | 2.3036 |
| 6000 | 0.6160 | 2.7518 |

The validation loss was best around step 3000, then worsened while training loss kept falling. That is useful evidence of overfitting on the capped training subset.

## Findings

- Causal masking is required so next-token training cannot look ahead.
- RoPE makes Q/K comparisons position-aware without changing tensor shapes.
- Multi-head attention learns several attention patterns in parallel.
- RMSNorm and residual connections keep the decoder block stable.
- The MLP transforms per-token features after attention has mixed token information.
- Naive generation recomputes the full prefix at every step.
- KV cache reuses old K/V tensors and computes only the newest token during decode.

## Limitations

- The BPE tokenizer is educational and simple, not production-grade.
- Full-dataset cloud training uses `fast_bpe`, which trains BPE from the local CSVs
  with the `tokenizers` package and caches the result.
- The local training run uses a capped dataset slice.
- Existing older checkpoints may not include tokenizer state; new checkpoints do.
- The current benchmark is a local smoke benchmark, not a full hardware study.
- BF16 support is wired through autocast, but the README does not yet include measured FP32 vs BF16 results.

## Next Work

- Run a controlled `local_experiment.yaml` training run and record generation examples.
- Run FP32 vs BF16 experiments.
- Run gradient accumulation experiments.
- Benchmark context lengths 32, 128, 256, and 512.
- Expand the final README tables with GPU memory and throughput measurements.
