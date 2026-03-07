import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter

from src.cnn import LionCNN, build_cnn_ablation
from src.dataset import build_dataloaders
from src.evaluate import evaluate_model, save_metrics
from src.train import train_one_epoch, validate, build_optimizer
from src.vit import build_vit, build_vit_ablation

console = Console()

CNN_ABLATIONS = {
    "baseline": "Full CNN (baseline)",
    "A1": "Remove 3rd conv layer",
    "A2": "Halve filter counts (16,32,64)",
    "A3": "Average pooling instead of max",
    "A4": "ReLU instead of GELU",
    "A5": "No dropout in FC layer",
}

VIT_ABLATIONS = {
    "baseline": "Full ViT (baseline)",
    "B1": "Reduce layers 12 -> 8",
    "B2": "Remove positional encoding",
    "B3": "Patch size 32x32 instead of 16x16",
    "B4": "Reduce attention heads 12 -> 6",
}


def build_ablation_model(model_type: str, variant: str, num_classes: int) -> nn.Module:
    if model_type == "cnn":
        if variant == "baseline":
            return LionCNN(num_classes=num_classes)
        return build_cnn_ablation(variant, num_classes=num_classes)
    else:
        if variant == "baseline":
            return build_vit(num_classes=num_classes)
        return build_vit_ablation(variant, num_classes=num_classes)


def run_ablation(
    model_type: str,
    variant: str,
    train_dl,
    val_dl,
    test_dl,
    class_names: list[str],
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    num_classes = len(class_names)
    model = build_ablation_model(model_type, variant, num_classes).to(device)
    optimizer = build_optimizer(model, model_type, lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    variant_dir = out_dir / f"{model_type}_{variant}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_dl, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), variant_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            console.print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(variant_dir / "best_model.pt", weights_only=True))
    metrics = evaluate_model(model, test_dl, class_names, device, variant_dir)
    save_metrics(metrics, variant_dir)
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run ablation studies for lion re-ID models",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "-d", "--data-dir", type=str, default="../Lion_6_ID",
        help="Path to lion image dataset",
    )
    parser.add_argument(
        "-t", "--type", type=str, choices=["cnn", "vit", "all"], default="all",
        dest="model_type", help="Which model ablations to run",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=150,
        help="Max epochs per ablation",
    )
    parser.add_argument(
        "-l", "--lr", type=float, default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32,
        help="Batch size",
    )
    parser.add_argument(
        "-p", "--patience", type=int, default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="output/ablations",
        help="Output directory for ablation results",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Ablation Studies[/] | device={device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dl, val_dl, test_dl, class_names = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    run_types = []
    if args.model_type in ("cnn", "all"):
        run_types.append(("cnn", CNN_ABLATIONS))
    if args.model_type in ("vit", "all"):
        run_types.append(("vit", VIT_ABLATIONS))

    for model_type, ablations in run_types:
        console.print(f"\n[bold cyan]{'='*50}[/]")
        console.print(f"[bold cyan]{model_type.upper()} Ablations[/]")
        console.print(f"[bold cyan]{'='*50}[/]")

        for variant, description in ablations.items():
            console.print(f"\n[bold]{model_type.upper()} - {variant}:[/] {description}")
            metrics = run_ablation(
                model_type, variant, train_dl, val_dl, test_dl, class_names,
                args.epochs, args.lr, args.patience, device, out_dir,
            )
            all_results[f"{model_type}_{variant}"] = metrics
            console.print(
                f"  -> F1={metrics['f1_score']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"Rec={metrics['recall']:.4f}  "
                f"AUC={metrics['roc_auc']:.4f}"
            )

    # Save combined results
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary tables
    for model_type, ablations in run_types:
        table = Table(title=f"{model_type.upper()} Ablation Results")
        table.add_column("Variant", style="cyan")
        table.add_column("Description")
        table.add_column("F1", justify="right", style="green")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("ROC AUC", justify="right")

        for variant, description in ablations.items():
            key = f"{model_type}_{variant}"
            if key in all_results:
                m = all_results[key]
                table.add_row(
                    variant, description,
                    f"{m['f1_score']:.4f}", f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}", f"{m['roc_auc']:.4f}",
                )
        console.print(table)

    console.print(f"\n[bold green]All ablation results saved to {out_dir}[/]")


if __name__ == "__main__":
    main()
