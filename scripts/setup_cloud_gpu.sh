#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f "archive/train.csv" ] || [ ! -f "archive/validation.csv" ]; then
  echo "Missing dataset files."
  echo "Expected archive/train.csv and archive/validation.csv in the repository root."
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import torch

print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"bf16_supported: {torch.cuda.is_bf16_supported()}")
PY

python mini-transformer/train.py --config mini-transformer/configs/cloud_gpu.yaml
