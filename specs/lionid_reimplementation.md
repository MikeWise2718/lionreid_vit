# Lion Re-Identification with Vision Transformers - Reimplementation Plan

Reimplementation of "A Novel Approach To Lion Re-Identification Using Vision Transformers"
(Matlala, van der Haar, Vandapalli - University of Johannesburg, SACAIR 2024).

## Dataset

- **Source:** `D:/senckenberg/Lion_6_ID/` (parent directory)
- **6 lions** (paper used 12 from Drakenstein Lion Park, we have 6 from a different source)
- Images organized as `<LionID>/` with sighting-session subfolders containing `.JPG` files
- **430 total images** (paper had 285)

| Lion ID | Name | Images |
|---------|------|--------|
| 13M33 | BANTAM | 83 |
| 13M37 | AXIOM | 28 |
| 13M38 | BUNDU | 135 |
| 13M43 | POEPOOG | 107 |
| 13M52 | FUZZY | 48 |
| 13M53 | GUASS | 29 |

**Note:** Dataset is imbalanced (BUNDU has ~5x more images than AXIOM). Will need to address this via augmentation or sampling strategy.

## Task Tracker

| # | Task | Status |
|---|------|--------|
| 1 | Project setup (uv, dependencies) | pending |
| 2 | Data loading & exploration | pending |
| 3 | Data preparation pipeline | pending |
| 4 | CNN baseline model | pending |
| 5 | Vision Transformer model | pending |
| 6 | Training loop & evaluation | pending |
| 7 | Ablation studies | pending |
| 8 | Results comparison & visualization | pending |

---

## Phase 1: Project Setup

- Initialize `uv` project with `pyproject.toml`
- Key dependencies: `torch`, `torchvision`, `timm`, `rich`, `rich-argparse`, `matplotlib`, `scikit-learn`, `Pillow`
- Create CLI entry point `train.py` with `rich` console output and `rich-argparse` formatting
- CLI args: `--data-dir`, `--model` (cnn|vit), `--epochs`, `--batch-size`, `--lr`, `--output-dir`, `--seed`

## Phase 2: Data Loading & Exploration

- Write a dataset loader that:
  - Walks `Lion_6_ID/` directory tree
  - Maps each top-level folder (e.g. `13M33-BANTAM`) to a class label (0-5)
  - Collects all `.JPG` paths with their labels
- Print dataset statistics: images per class, total count
- Display sample images per lion for visual sanity check

## Phase 3: Data Preparation Pipeline

Following the paper's methodology:

1. **Resize** all images to **384 x 384** pixels
2. **Convert to grayscale** (single channel) - paper used grayscale to simulate infrared/low-quality camera trap conditions
3. **Augmentation** (training set only):
   - Random horizontal flip
   - Random rotation (up to 20 degrees)
   - Color jitter (brightness, contrast, saturation, hue) - applied before grayscale conversion
   - Gaussian blur
4. **Normalize** pixel values
5. **Split**: 70% train / 15% validation / 15% test
   - Stratified split to preserve class distribution
   - Fixed random seed for reproducibility

## Phase 4: CNN Baseline

Replicate the paper's custom CNN architecture:

```
Input (1 x 384 x 384)
  -> Conv2d(1, 32, 3x3) + GELU + MaxPool2d(2x2)
  -> Conv2d(32, 64, 3x3) + GELU + MaxPool2d(2x2)
  -> Conv2d(64, 128, 3x3) + GELU + MaxPool2d(2x2)
  -> Flatten
  -> Linear(*, num_classes) + GELU + Dropout(0.25)
  -> Output (6 classes)
```

Hyperparameters:
- Batch size: 32
- Epochs: 150
- Learning rate: 1e-4
- Optimizer: Adam (betas: 0.9, 0.9999)
- Dropout: 0.25
- Classes: 6 (paper had 12)

## Phase 5: Vision Transformer

Use pre-trained ViT from `timm` library:

- Model: `vit_base_patch16_384` (ViT-B/16, pre-trained on ImageNet-21k)
- Modify patch embedding to accept **1-channel grayscale** input:
  - Replace first conv layer: `Conv2d(1, embed_dim, 16x16, stride=16)`
- Replace classification head for 6 classes

Fine-tuning config (from paper):
- Hidden layers: 12
- Attention heads: 12
- Hidden activation: GELU
- Hidden dropout: 0.25
- Attention dropout: 0.1
- Batch size: 32 (paper used 4096 for pre-training, but we fine-tune with smaller batch)
- Learning rate: 1e-4
- Optimizer: AdamW

## Phase 6: Training Loop & Evaluation

Shared training infrastructure for both models:

- Training loop with validation after each epoch
- Early stopping based on validation loss (patience ~10 epochs)
- Track per-epoch: loss, accuracy, precision, recall, F1
- Save best model checkpoint
- Log training progress with `rich` progress bars and live tables

**Evaluation metrics** (matching the paper):
- F1 Score (macro-averaged)
- Precision (macro-averaged)
- Recall (macro-averaged)
- ROC AUC (one-vs-rest, macro-averaged)
- Log Loss
- Mean Absolute Error
- Confusion matrix

## Phase 7: Ablation Studies

Replicate the paper's ablation experiments:

### CNN Ablations
| Experiment | Modification |
|-----------|-------------|
| A1 | Remove 3rd convolutional layer (128 filters) |
| A2 | Halve filter counts (16, 32, 64) |
| A3 | Replace max pooling with average pooling |
| A4 | Replace GELU with ReLU |
| A5 | Remove dropout from FC layer |

### ViT Ablations
| Experiment | Modification |
|-----------|-------------|
| B1 | Reduce transformer layers from 12 to 8 |
| B2 | Remove positional encoding |
| B3 | Increase patch size from 16x16 to 32x32 |
| B4 | Reduce attention heads from 12 to 6 |

## Phase 8: Results & Visualization

- Side-by-side comparison table (CNN vs ViT) matching paper's Table 3
- ROC curves (per-class and macro-averaged)
- Precision-Recall curves
- Training loss/accuracy curves over epochs
- Confusion matrices for both models
- Ablation results tables
- Save all plots to `output/` directory

## Key Differences from Paper

| Aspect | Paper | Ours |
|--------|-------|------|
| Lions | 12 (Drakenstein Lion Park) | 6 (Kgalagadi field data) |
| Images | 285 | 430 |
| Source | Social media | Field photography |
| Class balance | Not discussed | Imbalanced (28-135 per class) |
| Classes | 12 | 6 |

## File Structure (planned)

```
lionreid_vit/
  specs/
    lionid_reimplementation.md    # this file
  src/
    __init__.py
    dataset.py                    # data loading, splits, augmentation
    cnn.py                        # CNN model definition
    vit.py                        # ViT model setup (timm)
    train.py                      # training loop, evaluation
    evaluate.py                   # metrics, plots, comparison
  train.py                        # CLI entry point
  pyproject.toml
  output/                         # checkpoints, plots, results
```
