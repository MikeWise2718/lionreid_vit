#!/bin/bash
set -e

# Lion Re-ID: Full training pipeline
# Run on a CUDA-capable machine (e.g. 4090)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${1:-../Lion_6_ID}"
EPOCHS="${2:-150}"

echo "============================================"
echo "  Lion Re-ID - Full Training Pipeline"
echo "  Data:   $DATA_DIR"
echo "  Epochs: $EPOCHS"
echo "============================================"

# Install dependencies (CUDA torch configured in pyproject.toml)
source ~/.local/bin/env 2>/dev/null || true
uv sync

# Verify CUDA
uv run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Step 1/4: Training CNN ==="
uv run python -m src.train -m cnn -e "$EPOCHS" -d "$DATA_DIR" -v

echo ""
echo "=== Step 2/4: Training ViT ==="
uv run python -m src.train -m vit -e "$EPOCHS" -d "$DATA_DIR" -v

echo ""
echo "=== Step 3/4: Ablation Studies ==="
uv run python -m src.ablation -t all -e "$EPOCHS" -d "$DATA_DIR"

echo ""
echo "=== Step 4/4: Generating Report ==="
uv run python -m src.report -o output

echo ""
echo "============================================"
echo "  Done! Results in output/"
echo "  PDF report: output/lion_reid_report.pdf"
echo "============================================"
