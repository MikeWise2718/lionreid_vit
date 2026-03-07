import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from rich.console import Console

console = Console()


def gather_predictions(model, dataloader, device):
    """Run model on dataloader and return (all_labels, all_preds, all_probs)."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = all_probs.argmax(axis=1)
    return all_labels, all_preds, all_probs


def compute_metrics(labels, preds, probs, class_names):
    """Compute all metrics matching the paper."""
    num_classes = len(class_names)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_score": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "log_loss": log_loss(labels, probs, labels=list(range(num_classes))),
        "mean_absolute_error": mean_absolute_error(labels, preds),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro",
            labels=list(range(num_classes)),
        )
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def plot_confusion_matrix(labels, preds, class_names, out_path):
    """Save confusion matrix plot."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(labels, probs, class_names, out_path):
    """Save per-class ROC curves."""
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_classes):
        binary_labels = (labels == i).astype(int)
        if binary_labels.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
        auc_val = roc_auc_score(binary_labels, probs[:, i])
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precision_recall_curves(labels, probs, class_names, out_path):
    """Save per-class precision-recall curves."""
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_classes):
        binary_labels = (labels == i).astype(int)
        if binary_labels.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(binary_labels, probs[:, i])
        ax.plot(rec, prec, label=class_names[i])
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_history(history, out_path):
    """Save training loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_model(model, test_dl, class_names, device, out_dir: Path) -> dict:
    """Full evaluation: compute metrics and save all plots."""
    labels, preds, probs = gather_predictions(model, test_dl, device)
    metrics = compute_metrics(labels, preds, probs, class_names)

    # Shorten class names for plots (e.g. "13M33-BANTAM" -> "BANTAM")
    short_names = [n.split("-", 1)[-1] if "-" in n else n for n in class_names]

    plot_confusion_matrix(labels, preds, short_names, out_dir / "confusion_matrix.png")
    plot_roc_curves(labels, probs, short_names, out_dir / "roc_curves.png")
    plot_precision_recall_curves(labels, probs, short_names, out_dir / "precision_recall.png")

    console.print(f"  Plots saved to {out_dir}")
    return metrics


def save_metrics(metrics: dict, out_dir: Path):
    """Save metrics dict as JSON."""
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
