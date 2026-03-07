import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich_argparse import RichHelpFormatter

from src.cnn import LionCNN
from src.dataset import build_dataloaders
from src.evaluate import evaluate_model, save_metrics
from src.vit import build_vit

console = Console()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Lion Re-Identification: ViT vs CNN",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=str,
        default="../Lion_6_ID",
        help="Path to lion image dataset",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["cnn", "vit"],
        default="vit",
        help="Model architecture to train",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=150,
        help="Number of training epochs",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "-l", "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output",
        help="Directory for checkpoints and plots",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-p", "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args(argv)


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    if model_name == "cnn":
        model = LionCNN(num_classes=num_classes)
    else:
        model = build_vit(num_classes=num_classes, pretrained=True)
    return model.to(device)


def build_optimizer(model: nn.Module, model_name: str, lr: float) -> torch.optim.Optimizer:
    if model_name == "cnn":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.9999))
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr)


def train_one_epoch(model, train_dl, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(model, val_dl, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Device:[/] {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dl, val_dl, test_dl, class_names = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed,
    )
    num_classes = len(class_names)

    model = build_model(args.model, num_classes, device)
    optimizer = build_optimizer(model, args.model, args.lr)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"[bold]Model:[/] {args.model.upper()}  params={param_count:,}")

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Training {args.model.upper()}", total=args.epochs)

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_dl, criterion, device)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if args.verbose:
                console.print(
                    f"  Epoch {epoch:3d}/{args.epochs} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), out_dir / "best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                console.print(f"[yellow]Early stopping at epoch {epoch}[/]")
                break

            progress.update(task, advance=1)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))
    console.print("\n[bold green]Evaluating on test set...[/]")
    metrics = evaluate_model(model, test_dl, class_names, device, out_dir)
    save_metrics(metrics, out_dir)

    # Save training history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Print results table
    table = Table(title=f"{args.model.upper()} Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
    console.print(table)

    return metrics, history


def main(argv=None):
    args = parse_args(argv)
    console.print(f"[bold green]Lion Re-ID[/] | model={args.model} epochs={args.epochs} lr={args.lr}")
    console.print(f"  data-dir: {args.data_dir}")
    console.print(f"  output:   {args.output_dir}")
    console.print()
    train(args)


if __name__ == "__main__":
    main()
