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

# Paper results from Matlala et al. (SACAIR 2024), Table 3
# 12 lions, 285 images from Drakenstein Lion Park
PAPER_RESULTS = {
    "cnn": {
        "f1_score": 0.8558,
        "precision": 0.8820,
        "recall": 0.8552,
        "roc_auc": 0.9973,
        "log_loss": 0.6075,
        "mean_absolute_error": 3.3872,
    },
    "vit": {
        "f1_score": 0.9744,
        "precision": 0.9787,
        "recall": 0.9747,
        "roc_auc": 1.0000,
        "log_loss": 0.0762,
        "mean_absolute_error": 0.9327,
    },
}

# Paper ablation impacts (reported as % change, not absolute values)
PAPER_CNN_ABLATIONS = {
    "A1: Remove 3rd conv layer": "-15% across all metrics",
    "A2: Halve filter counts": "-5% across all metrics",
    "A3: Avg pooling vs max": "-8% performance",
    "A4: ReLU vs GELU": "Required 250 epochs for 70% precision/accuracy",
    "A5: No dropout": "Overfitting, validation performance dropped",
}

PAPER_VIT_ABLATIONS = {
    "B1: Reduce layers 12->8": "-22% across all metrics",
    "B2: Remove positional encoding": "-17% across all metrics",
    "B3: Patch 32x32 vs 16x16": "Less detail captured, faster compute",
    "B4: Reduce heads 12->6": "Decreased global context, little improvement with more",
}


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


def plot_paper_comparison(results: dict, out_path: Path):
    """Bar chart comparing our results to the original paper."""
    metrics_to_plot = ["f1_score", "precision", "recall", "roc_auc"]
    labels = ["F1", "Precision", "Recall", "ROC AUC"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, model_name, title in zip(axes, ["cnn", "vit"], ["CNN", "ViT"]):
        if model_name not in results:
            continue
        ours = [results[model_name]["metrics"].get(m, 0) for m in metrics_to_plot]
        paper = [PAPER_RESULTS[model_name].get(m, 0) for m in metrics_to_plot]

        x = range(len(labels))
        width = 0.35
        bars1 = ax.bar([xi - width / 2 for xi in x], paper, width, label="Paper (12 lions)", color="#4C72B0")
        bars2 = ax.bar([xi + width / 2 for xi in x], ours, width, label="Ours (6 lions)", color="#DD8452")

        for bar, val in zip(bars1, paper):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars2, ours):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{title} Comparison")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)

    fig.suptitle("Our Results vs. Matlala et al. (2024)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_comparison_plots(results: dict, output_dir: Path):
    """Generate all comparison PNG plots."""
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart comparison
    plot_comparison_bar(results, comp_dir / "metrics_comparison.png")

    # Paper comparison
    plot_paper_comparison(results, comp_dir / "paper_comparison.png")

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

    # Paper comparison section
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Comparison with Original Study", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "Matlala et al. (SACAIR 2024) used 12 lions / 285 images from Drakenstein Lion Park social media. "
        "Our reimplementation uses 6 lions / 398 images from Kgalagadi field photography. "
        "Lower absolute scores are expected due to fewer classes and a different, potentially harder dataset."
    )
    pdf.ln(5)

    # Side-by-side table: paper vs ours
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Main Results", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 9)
    pc_cols = [35, 25, 25, 25, 25]
    pdf.cell(pc_cols[0], 7, "Metric", border=1)
    pdf.cell(pc_cols[1], 7, "Paper CNN", border=1, align="C")
    pdf.cell(pc_cols[2], 7, "Our CNN", border=1, align="C")
    pdf.cell(pc_cols[3], 7, "Paper ViT", border=1, align="C")
    pdf.cell(pc_cols[4], 7, "Our ViT", border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    metric_keys = [
        ("f1_score", "F1 Score"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("roc_auc", "ROC AUC"),
        ("log_loss", "Log Loss"),
        ("mean_absolute_error", "MAE"),
    ]
    for key, label in metric_keys:
        pdf.cell(pc_cols[0], 7, label, border=1)
        pdf.cell(pc_cols[1], 7, f"{PAPER_RESULTS['cnn'].get(key, 0):.4f}", border=1, align="C")
        our_cnn = results.get("cnn", {}).get("metrics", {}).get(key, float("nan"))
        pdf.cell(pc_cols[2], 7, f"{our_cnn:.4f}", border=1, align="C")
        pdf.cell(pc_cols[3], 7, f"{PAPER_RESULTS['vit'].get(key, 0):.4f}", border=1, align="C")
        our_vit = results.get("vit", {}).get("metrics", {}).get(key, float("nan"))
        pdf.cell(pc_cols[4], 7, f"{our_vit:.4f}", border=1, align="C")
        pdf.ln()
    pdf.ln(5)

    # Dataset differences table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Dataset Differences", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 9)
    dd_cols = [40, 55, 55]
    pdf.cell(dd_cols[0], 7, "Aspect", border=1)
    pdf.cell(dd_cols[1], 7, "Paper", border=1, align="C")
    pdf.cell(dd_cols[2], 7, "Ours", border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    diffs = [
        ("Lions", "12 (Drakenstein Lion Park)", "6 (Kgalagadi field data)"),
        ("Images", "285", "398"),
        ("Source", "Social media", "Field photography"),
        ("Classes", "12", "6"),
        ("Balance", "Not discussed", "Imbalanced (22-134/class)"),
        ("GPU", "A100 (Google Colab)", "RTX 4090"),
    ]
    for aspect, paper_val, our_val in diffs:
        pdf.cell(dd_cols[0], 7, aspect, border=1)
        pdf.cell(dd_cols[1], 7, paper_val, border=1, align="C")
        pdf.cell(dd_cols[2], 7, our_val, border=1, align="C")
        pdf.ln()
    pdf.ln(5)

    # Key findings
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Key Findings", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    findings = [
        "1. ViT significantly outperforms CNN in both studies, confirming the paper's main conclusion.",
        "2. Our absolute scores are lower (ViT F1: 0.75 vs 0.97), likely due to harder field images "
        "and class imbalance (smallest class has only 22 images).",
        "3. The relative ViT advantage is consistent: +35% F1 over CNN (ours) vs +14% (paper).",
        "4. ROC AUC remains high in both studies (>0.84 CNN, >0.95 ViT), showing strong "
        "discriminative ability even with lower precision/recall.",
    ]
    for finding in findings:
        pdf.multi_cell(0, 5, finding)
        pdf.ln(2)
    pdf.ln(3)

    # Ablation comparison
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Ablation Comparison", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 9)
    ac_cols = [50, 45, 45]
    pdf.cell(ac_cols[0], 7, "Modification", border=1)
    pdf.cell(ac_cols[1], 7, "Paper Impact", border=1, align="C")
    pdf.cell(ac_cols[2], 7, "Our Impact", border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    if ablation_results:
        abl_comparison = [
            ("CNN: No 3rd conv layer", "-15% all metrics",
             ablation_results.get("cnn_A1", {}), ablation_results.get("cnn_baseline", {})),
            ("CNN: Halve filters", "-5% all metrics",
             ablation_results.get("cnn_A2", {}), ablation_results.get("cnn_baseline", {})),
            ("CNN: Avg pooling", "-8% performance",
             ablation_results.get("cnn_A3", {}), ablation_results.get("cnn_baseline", {})),
            ("CNN: ReLU vs GELU", "Drastic accuracy reduction",
             ablation_results.get("cnn_A4", {}), ablation_results.get("cnn_baseline", {})),
            ("ViT: 8 layers vs 12", "-22% all metrics",
             ablation_results.get("vit_B1", {}), ablation_results.get("vit_baseline", {})),
            ("ViT: No pos. encoding", "-17% all metrics",
             ablation_results.get("vit_B2", {}), ablation_results.get("vit_baseline", {})),
            ("ViT: 32x32 patches", "Less detail, faster",
             ablation_results.get("vit_B3", {}), ablation_results.get("vit_baseline", {})),
            ("ViT: 6 heads vs 12", "Decreased global context",
             ablation_results.get("vit_B4", {}), ablation_results.get("vit_baseline", {})),
        ]
        for mod, paper_impact, variant_m, baseline_m in abl_comparison:
            base_f1 = baseline_m.get("f1_score", 0)
            var_f1 = variant_m.get("f1_score", 0)
            if base_f1 > 0:
                pct = ((var_f1 - base_f1) / base_f1) * 100
                our_impact = f"F1: {var_f1:.3f} ({pct:+.1f}%)"
            else:
                our_impact = "N/A"
            pdf.cell(ac_cols[0], 7, mod, border=1)
            pdf.cell(ac_cols[1], 7, paper_impact, border=1, align="C")
            pdf.cell(ac_cols[2], 7, our_impact, border=1, align="C")
            pdf.ln()
    pdf.ln(5)

    # Paper comparison bar chart
    paper_comp_path = comp_dir / "paper_comparison.png"
    if paper_comp_path.exists():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Visual Comparison with Paper", new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(paper_comp_path), x=10, w=190)

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
