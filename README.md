# TinyLM: Decoder-Only Transformer From Scratch

TinyLM is a small decoder-only language model implemented in PyTorch for learning
and demonstrating the mechanics behind modern LLMs.

The project intentionally implements the Transformer stack directly instead of using
`nn.Transformer`, Hugging Face `Trainer`, FlashAttention, or pretrained model code.
It covers tokenization, embeddings, RoPE, causal self-attention, multi-head attention,
RMSNorm, MLP blocks, residual connections, training, checkpointing, generation, and
KV-cache inference.

## Results

Cloud training was run on one NVIDIA RTX 4090 using BF16 autocast.

| Metric | Value |
| --- | ---: |
| Dataset | TinyStories CSV |
| Tokenizer | BPE, vocab size 1024 |
| Model | 8 layers, 8 heads, D=384 |
| Context length | 256 |
| Effective batch size | 128 sequences |
| Training steps | 20,000 |
| Final train loss | 1.3851 |
| Final validation loss | 1.1627 |
| Training throughput | 217,471 tokens/s |

Generation benchmark on the trained cloud checkpoint:

| Mode | Generated Tokens | TTFT | Avg Decode Latency | Tokens/s | Peak Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive | 100 | 0.282679s | 0.010086s | 78.05 | 65.93 MB |
| KV cache | 100 | 0.009137s | 0.009386s | 105.63 | 69.04 MB |

KV-cache generation was `1.35x` faster for this benchmark.

Example generated text from the trained model:

```text
Once upon a time, there was a little girl named Lily. She loved to play outside
with her friends. One day, she accidentally pinched her friend. Her friend got mad
and yelled at her to go home. Lily felt sad...
```

## Architecture

```text
input_ids: [B, T]
-> BPE token embeddings: [B, T, D]
-> TransformerBlock x n_layers
   -> RMSNorm
   -> multi-head causal self-attention
      -> Q/K/V projections
      -> RoPE on Q and K
      -> causal mask
      -> attention-weighted value mix
   -> residual add
   -> RMSNorm
   -> MLP: D -> 4D -> GELU -> D
   -> residual add
-> final RMSNorm
-> LM head
-> logits: [B, T, V]
```

Shape notation:

```text
B  = batch size
T  = sequence length
D  = model dimension
V  = vocabulary size
H  = number of attention heads
Dh = D / H
```

## Implemented

- CSV text data pipeline
- character tokenizer for first experiments
- educational Python BPE tokenizer
- optimized Rust-backed BPE path via `tokenizers`
- tokenizer caching for full-dataset cloud runs
- next-token language-modeling dataset
- token embedding table
- RoPE positional encoding
- single-head attention for learning/debugging
- multi-head causal self-attention
- KV cache for faster autoregressive decoding
- RMSNorm
- MLP/feed-forward block
- pre-norm Transformer decoder block
- full decoder-only language model
- AdamW training loop
- validation evaluation
- checkpoint saving with tokenizer/model metadata
- gradient accumulation
- BF16 autocast support
- greedy, temperature, and top-k generation
- local web UI for generation
- naive vs cached generation benchmark
- unit tests

## What Is Not Used

- `nn.Transformer`
- Hugging Face `Trainer`
- pretrained model weights
- FlashAttention
- distributed training
- custom CUDA kernels

The fast BPE path uses the `tokenizers` package because full-corpus BPE training is
not the main learning objective of this project. The model architecture and training
loop are still implemented directly.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dataset is expected at:

```text
archive/train.csv
archive/validation.csv
```

Each CSV must contain a `text` column.

## Training

Smoke test:

```bash
python mini-transformer/train.py --config mini-transformer/configs/smoke.yaml
```

Local experiment:

```bash
python mini-transformer/train.py --config mini-transformer/configs/local_experiment.yaml
```

Cloud GPU run:

```bash
bash setup_cloud_gpu.sh
```

The cloud config uses:

```text
GPU device: cuda
Precision: BF16 autocast
Tokenizer: fast_bpe, vocab size 1024
Tokenizer cache: tokenizers/tinystories_bpe_1024.json
Context length: 256
Batch size: 64
Gradient accumulation: 2
Effective batch size: 128 sequences
Model: 8 layers, 8 heads, D=384
Training: 20,000 optimizer steps
```

## Generation CLI

With a checkpoint:

```bash
python mini-transformer/generate.py \
  --config mini-transformer/configs/cloud_gpu.yaml \
  --checkpoint checkpoints/cloud_gpu/step_003000.pt \
  --prompt "Once upon a time, there was a boy named Tom." \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 40 \
  --use-cache \
  --device cpu
```

New checkpoints include tokenizer state, so generation can load the tokenizer directly
instead of retraining BPE from CSV.

## Local Generation UI

Launch the small local interface:

```bash
python mini-transformer/serve_generation.py \
  --config mini-transformer/configs/cloud_gpu.yaml \
  --checkpoint checkpoints/cloud_gpu/step_003000.pt \
  --device cpu
```

Then open:

```text
http://127.0.0.1:7860
```

The UI exposes:

- prompt
- max generated tokens
- temperature
- top-k
- greedy decoding
- KV-cache toggle
- loaded checkpoint label

Note: the local demo currently uses the recovered `step_003000` cloud checkpoint. The
full `step_020000` run completed successfully on the GPU, but the local download of
the final checkpoint archive was truncated. The final metrics above are from the
completed cloud run.

## Benchmark

Run:

```bash
python mini-transformer/eval/benchmark.py \
  --config mini-transformer/configs/cloud_gpu.yaml \
  --checkpoint checkpoints/cloud_gpu/step_020000.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 100
```

The benchmark compares naive autoregressive decoding against KV-cache decoding.

Naive generation recomputes the full prefix at every step. KV-cache generation keeps
old K/V tensors and only computes attention for the newest token.

## Engineering Notes

- Causal masking prevents tokens from attending to future targets.
- RoPE rotates Q and K vectors so attention scores become position-aware.
- Multi-head attention learns several attention patterns in parallel.
- RMSNorm stabilizes activations before attention and MLP sublayers.
- Residual connections preserve the running token representation across sublayers.
- The MLP performs per-token feature transformation after attention mixes context.
- Checkpoints store tokenizer state so generation can be reproduced without rebuilding
  the tokenizer.
- The generation UI is dependency-free and uses Python's built-in HTTP server.

## Limitations

- This is a small educational model, not an instruction-following assistant.
- Outputs are TinyStories-style continuations, not general-purpose answers.
- Prompt following is limited, especially with early checkpoints.
- The recovered local checkpoint is weaker than the final trained checkpoint.
- The BPE tokenizer is suitable for this project, but not a production tokenizer.
- No distributed training, FlashAttention, or large-scale hyperparameter sweep is used.
