# Mini Transformer

Small decoder-only Transformer language model built from scratch in PyTorch.

## Smoke Test

Run a short end-to-end training check from the repository root:

```bash
/Users/eda/.pyenv/versions/3.11.7/bin/python3 mini-transformer/train.py --config mini-transformer/configs/smoke.yaml
```

This config uses a tiny model and a small slice of the dataset so it should finish quickly.

## Full Training Config

The larger default config is:

```bash
/Users/eda/.pyenv/versions/3.11.7/bin/python3 mini-transformer/train.py --config mini-transformer/configs/base.yaml
```

Start with the smoke config before running the full config.
