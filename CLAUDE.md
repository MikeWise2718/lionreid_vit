# CLAUDE.md

## Project Overview

Reimplementation of "A Novel Approach To Lion Re-Identification Using Vision Transformers" (Matlala et al., SACAIR 2024). Compares a custom CNN against a fine-tuned ViT-B/16 for identifying individual lions from photographs.

## Quick Reference

- **Language:** Python 3.13+
- **Package manager:** `uv`
- **Dataset:** `../Lion_6_ID/` (6 lions, 398 JPG images, not tracked in git)
- **Spec:** `specs/lionid_reimplementation.md`

## Commands

```bash
# Install dependencies
uv sync

# Train a model
uv run python -m src.train -m cnn -e 150 -d ../Lion_6_ID -v
uv run python -m src.train -m vit -e 150 -d ../Lion_6_ID -v

# Run ablation studies
uv run python -m src.ablation -t all -e 150 -d ../Lion_6_ID

# Generate PDF report with comparison plots
uv run python -m src.report -o output

# Full pipeline (for GPU machines)
bash run_all.sh ../Lion_6_ID 150
```

## Architecture

```
src/
  dataset.py    # Data loading, transforms, stratified splits
  cnn.py        # Custom 3-layer CNN + ablation variants (A1-A5)
  vit.py        # ViT-B/16 via timm + ablation variants (B1-B4)
  train.py      # Training loop with early stopping, CLI entry point
  evaluate.py   # Metrics computation and per-model plots
  ablation.py   # Ablation study runner
  report.py     # Comparison PNGs + consolidated PDF report
```

## CUDA Notes

- `pyproject.toml` pins PyTorch to `cu124` index — required on machines with CUDA 12.4/12.5 system cublas libraries
- cu128 builds will crash with `cublasLtCreate` errors on such systems
- CPU-only works fine for testing; CUDA required for full training

## Code Conventions

- Uses `rich` for console output and `rich-argparse` for CLI help formatting
- CLI flags: one-letter for common options (`-m`, `-e`, `-d`, `-o`, `-v`), two-letter for uncommon (`-th`, `-ms`)
- All images converted to 384x384 grayscale (single channel) per the paper's methodology
