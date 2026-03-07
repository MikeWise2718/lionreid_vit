import argparse
import sys

from rich.console import Console
from rich_argparse import RichHelpFormatter

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


def main(argv=None):
    args = parse_args(argv)
    console.print(f"[bold green]Lion Re-ID[/] | model={args.model} epochs={args.epochs} lr={args.lr}")
    console.print(f"  data-dir: {args.data_dir}")
    console.print(f"  output:   {args.output_dir}")
    # TODO: wire up dataset, model, training loop


if __name__ == "__main__":
    main()
