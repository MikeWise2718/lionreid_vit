import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter

from src.evaluate import plot_training_history

console = Console()


def load_results(output_dir: Path) -> dict:
    """Load metrics and history for CNN and ViT from output directory."""
    results = {}
    for model_name in ("cnn", "vit"):
        model_dir = output_dir / model_name
        metrics_file = model_dir / "metrics.json"
        history_file = model_dir / "history.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                results[model_name] = {"metrics": json.load(f)}
            if history_file.exists():
                with open(history_file) as f:
                    results[model_name]["history"] = json.load(f)
    return results


def load_ablation_results(output_dir: Path) -> dict | None:
    """Load ablation results if available."""
    ablation_file = output_dir / "ablations" / "ablation_results.json"
    if ablation_file.exists():
        with open(ablation_file) as f:
            return json.load(f)
    return None


def plot_comparison_bar(results: dict, out_path: Path):
    """Side-by-side bar chart comparing CNN vs ViT metrics."""
    metrics_to_plot = ["f1_score", "precision", "recall", "accuracy", "roc_auc"]
    labels = ["F1", "Precision", "Recall", "Accuracy", "ROC AUC"]

    models = list(results.keys())
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(models):
        vals = [results[model_name]["metrics"].get(m, 0) for m in metrics_to_plot]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], vals, width, label=model_name.upper())
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("CNN vs ViT: Test Metrics Comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_comparison_plots(results: dict, output_dir: Path):
    """Generate all comparison PNG plots."""
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart comparison
    plot_comparison_bar(results, comp_dir / "metrics_comparison.png")

    # Training curves for each model
    for model_name, data in results.items():
        if "history" in data:
            plot_training_history(data["history"], comp_dir / f"{model_name}_training.png")

    console.print(f"  Comparison plots saved to {comp_dir}")
    return comp_dir


def build_pdf_report(results: dict, ablation_results: dict | None, output_dir: Path):
    """Build a consolidated PDF report with all figures and tables."""
    comp_dir = output_dir / "comparison"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 40, "Lion Re-Identification", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "ViT vs CNN - Results Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(20)

    # Metrics comparison table
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Test Metrics Comparison", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    col_w = [50, 35, 35]
    pdf.cell(col_w[0], 8, "Metric", border=1)
    for model_name in results:
        pdf.cell(col_w[1], 8, model_name.upper(), border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    metric_labels = {
        "f1_score": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "roc_auc": "ROC AUC",
        "log_loss": "Log Loss",
        "mean_absolute_error": "Mean Abs. Error",
    }
    for key, label in metric_labels.items():
        pdf.cell(col_w[0], 7, label, border=1)
        for model_name in results:
            val = results[model_name]["metrics"].get(key, float("nan"))
            pdf.cell(col_w[1], 7, f"{val:.4f}", border=1, align="C")
        pdf.ln()

    # Comparison bar chart
    bar_path = comp_dir / "metrics_comparison.png"
    if bar_path.exists():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Metrics Comparison", new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(bar_path), x=10, w=190)

    # Per-model pages
    for model_name in results:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"{model_name.upper()} Results", new_x="LMARGIN", new_y="NEXT")

        # Training curves
        training_path = comp_dir / f"{model_name}_training.png"
        if training_path.exists():
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Training Curves", new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(training_path), x=10, w=190)

        model_dir = output_dir / model_name

        # Confusion matrix
        cm_path = model_dir / "confusion_matrix.png"
        if cm_path.exists():
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, f"{model_name.upper()} - Confusion Matrix", new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(cm_path), x=10, w=180)

        # ROC curves
        roc_path = model_dir / "roc_curves.png"
        if roc_path.exists():
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, f"{model_name.upper()} - ROC Curves", new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(roc_path), x=10, w=180)

        # Precision-Recall curves
        pr_path = model_dir / "precision_recall.png"
        if pr_path.exists():
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, f"{model_name.upper()} - Precision-Recall Curves", new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(pr_path), x=10, w=180)

    # Ablation results
    if ablation_results:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Ablation Studies", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        for prefix, title in [("cnn", "CNN Ablations"), ("vit", "ViT Ablations")]:
            variants = {k: v for k, v in ablation_results.items() if k.startswith(prefix)}
            if not variants:
                continue

            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", "B", 9)
            ab_cols = [40, 30, 30, 30, 30]
            pdf.cell(ab_cols[0], 7, "Variant", border=1)
            pdf.cell(ab_cols[1], 7, "F1", border=1, align="C")
            pdf.cell(ab_cols[2], 7, "Precision", border=1, align="C")
            pdf.cell(ab_cols[3], 7, "Recall", border=1, align="C")
            pdf.cell(ab_cols[4], 7, "ROC AUC", border=1, align="C")
            pdf.ln()

            pdf.set_font("Helvetica", "", 9)
            for variant_key, m in variants.items():
                label = variant_key.split("_", 1)[-1]
                pdf.cell(ab_cols[0], 7, label, border=1)
                pdf.cell(ab_cols[1], 7, f"{m['f1_score']:.4f}", border=1, align="C")
                pdf.cell(ab_cols[2], 7, f"{m['precision']:.4f}", border=1, align="C")
                pdf.cell(ab_cols[3], 7, f"{m['recall']:.4f}", border=1, align="C")
                pdf.cell(ab_cols[4], 7, f"{m.get('roc_auc', 0):.4f}", border=1, align="C")
                pdf.ln()
            pdf.ln(5)

    report_path = output_dir / "lion_reid_report.pdf"
    pdf.output(str(report_path))
    console.print(f"[bold green]PDF report saved to {report_path}[/]")
    return report_path


def print_comparison_table(results: dict):
    """Print rich comparison table to console."""
    table = Table(title="CNN vs ViT - Test Metrics")
    table.add_column("Metric", style="cyan")
    for model_name in results:
        table.add_column(model_name.upper(), justify="right", style="green")

    metric_labels = {
        "f1_score": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "roc_auc": "ROC AUC",
        "log_loss": "Log Loss",
        "mean_absolute_error": "Mean Abs. Error",
    }
    for key, label in metric_labels.items():
        row = [label]
        for model_name in results:
            val = results[model_name]["metrics"].get(key, float("nan"))
            row.append(f"{val:.4f}")
        table.add_row(*row)
    console.print(table)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate comparison report (PNGs + PDF)",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="output",
        help="Output directory containing model results",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    console.print("[bold green]Generating comparison report...[/]")

    results = load_results(output_dir)
    if not results:
        console.print("[red]No results found. Run training first.[/]")
        return

    print_comparison_table(results)
    generate_comparison_plots(results, output_dir)

    ablation_results = load_ablation_results(output_dir)
    build_pdf_report(results, ablation_results, output_dir)


if __name__ == "__main__":
    main()
