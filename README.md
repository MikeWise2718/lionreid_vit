# Lion Re-Identification with Vision Transformers

Reimplementation of ["A Novel Approach To Lion Re-Identification Using Vision Transformers"](https://doi.org/10.1007/978-3-031-78255-8_16) (Matlala, van der Haar, Vandapalli — SACAIR 2024).

This project compares a custom CNN against a fine-tuned Vision Transformer (ViT-B/16) for identifying individual lions from photographs, reproducing the paper's methodology and ablation studies on a different dataset.

## Results

| Metric | CNN | ViT |
|--------|-----|-----|
| F1 Score | 0.558 | **0.754** |
| Precision | 0.541 | **0.754** |
| Recall | 0.585 | **0.763** |
| Accuracy | 61.7% | **81.7%** |
| ROC AUC | 0.845 | **0.958** |

The ViT significantly outperforms the CNN, consistent with the original paper's findings. See `output/lion_reid_report.pdf` for the full report including ablation studies and comparison with the original paper.

## Dataset

The original paper used 12 lions / 285 images from Drakenstein Lion Park (South Africa). Our reimplementation uses 6 lions / 398 images from Kgalagadi field photography.

Expected directory structure:
```
Lion_6_ID/
  13M33-BANTAM/
    <sighting-session>/
      *.JPG
  13M37-AXIOM/
    ...
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

> **Note:** `pyproject.toml` configures PyTorch with CUDA 12.4 wheels. For CPU-only usage, remove the `[tool.uv.index]` and `[tool.uv.sources]` sections.

## Usage

### Train a single model

```bash
# CNN (custom 3-layer architecture from the paper)
uv run python -m src.train -m cnn -e 150 -d ../Lion_6_ID -v

# ViT (ViT-B/16 pretrained on ImageNet-21k, fine-tuned)
uv run python -m src.train -m vit -e 150 -d ../Lion_6_ID -v
```

### Run ablation studies

```bash
uv run python -m src.ablation -t all -e 150 -d ../Lion_6_ID
```

**CNN ablations (A1-A5):** remove 3rd conv layer, halve filters, average pooling, ReLU activation, no dropout.

**ViT ablations (B1-B4):** reduce layers (12→8), remove positional encoding, larger patches (32x32), fewer attention heads (12→6).

### Generate report

```bash
uv run python -m src.report -o output
```

Produces individual PNGs in `output/comparison/` and a consolidated PDF at `output/lion_reid_report.pdf`.

### Full pipeline (GPU)

```bash
bash run_all.sh ../Lion_6_ID 150
```

Runs CNN training, ViT training, all ablations, and report generation sequentially.

## Project Structure

```
src/
  dataset.py    # Data loading, augmentation, stratified 70/15/15 splits
  cnn.py        # Custom CNN model + ablation variants
  vit.py        # ViT-B/16 (timm) + ablation variants
  train.py      # Training loop with early stopping
  evaluate.py   # Metrics, confusion matrices, ROC/PR curves
  ablation.py   # Ablation study runner
  report.py     # Comparison plots and PDF report generation
specs/
  lionid_reimplementation.md  # Implementation plan and task tracker
output/                       # Generated results (gitignored)
```

## Key Differences from the Paper

| Aspect | Paper | Ours |
|--------|-------|------|
| Lions | 12 (Drakenstein Lion Park) | 6 (Kgalagadi field data) |
| Images | 285 | 398 |
| Source | Social media | Field photography |
| Classes | 12 | 6 |
| Class balance | Not discussed | Imbalanced (22–134 per class) |
| GPU | A100 (Colab) | RTX 4090 |

## Reference

```bibtex
@inproceedings{matlala2024lion,
  title={A Novel Approach To Lion Re-Identification Using Vision Transformers},
  author={Matlala, Boitumelo and van der Haar, Dustin and Vandapalli, Hima},
  booktitle={SACAIR 2024},
  series={CCIS},
  volume={2326},
  pages={270--281},
  year={2025},
  publisher={Springer},
  doi={10.1007/978-3-031-78255-8_16}
}
```
