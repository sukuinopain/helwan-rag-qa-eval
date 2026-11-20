"""Generate 300-DPI evaluation charts from `eval/eval_results.json`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_CATEGORY_ORDER = (
    "direct_factual",
    "multi_hop",
    "tabular_numerical",
    "unanswerable",
)
_CATEGORY_LABELS = {
    "direct_factual": "Direct Factual",
    "multi_hop": "Multi-Hop",
    "tabular_numerical": "Tabular &\nNumerical",
    "unanswerable": "Unanswerable",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#94A3B8")
    ax.set_axisbelow(True)


def _add_bar_labels(ax: plt.Axes, bars: Any, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1E293B",
        )


def plot_retrieval_metrics(report: dict[str, Any], output_path: Path) -> Path:
    """Save grouped bar chart of Recall@K and MRR across answerable categories."""
    summary = report.get("summary", {})
    by_cat = summary.get("by_category", {})
    overall = summary.get("overall", {})
    metadata = report.get("metadata", {})
    top_k = int(metadata.get("top_k", 5))
    embed_model = str(metadata.get("embedding_model", "intfloat/multilingual-e5-base"))

    keys = ["direct_factual", "multi_hop", "tabular_numerical"]
    labels = [_CATEGORY_LABELS[k] for k in keys] + ["Overall\n(Answerable)"]

    recall_vals = [float(by_cat.get(k, {}).get("recall_at_k") or 0.0) for k in keys] + [
        float(overall.get("recall_at_k") or 0.0)
    ]
    mrr_vals = [float(by_cat.get(k, {}).get("mrr") or 0.0) for k in keys] + [
        float(overall.get("mrr") or 0.0)
    ]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=300)
    bars_recall = ax.bar(
        x - width / 2,
        recall_vals,
        width,
        label=f"Recall@{top_k}",
        color="#2563EB",
        edgecolor="#1D4ED8",
        linewidth=0.6,
    )
    bars_mrr = ax.bar(
        x + width / 2,
        mrr_vals,
        width,
        label="MRR",
        color="#0D9488",
        edgecolor="#0F766E",
        linewidth=0.6,
    )

    _add_bar_labels(ax, bars_recall)
    _add_bar_labels(ax, bars_mrr)
    _clean_axes(ax)

    ax.axvline(2.5, color="#94A3B8", linestyle=":", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Score (0.0 – 1.0)")
    ax.set_title(f"Retrieval Performance by Question Category (Top-{top_k})", pad=22)
    ans_count = overall.get("answerable_count", 0)
    fig.suptitle(
        f"Embedding Model: {embed_model} | Answerable Queries (n={ans_count})",
        fontsize=9.5,
        color="#475569",
        y=0.925,
    )
    ax.legend(frameon=False, loc="upper right", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_generation_quality(report: dict[str, Any], output_path: Path) -> Path:
    """Save grouped bar chart of Faithfulness, Answer Accuracy, and Abstention Accuracy."""
    summary = report.get("summary", {})
    by_cat = summary.get("by_category", {})
    metadata = report.get("metadata", {})
    provider = str(metadata.get("llm_provider") or "configured-llm")
    llm_model = str(metadata.get("llm_model") or "default")
    judge_mode = str(metadata.get("judge_mode") or "auto")

    labels = [_CATEGORY_LABELS[k] for k in _CATEGORY_ORDER]
    faith_vals = [float(by_cat.get(k, {}).get("faithfulness") or 0.0) for k in _CATEGORY_ORDER]
    acc_vals = [float(by_cat.get(k, {}).get("answer_accuracy") or 0.0) for k in _CATEGORY_ORDER]
    abst_val = float(by_cat.get("unanswerable", {}).get("abstention_accuracy") or 0.0)

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=300)

    # Shift first three categories to 2 centered bars, and unanswerable to 3 bars.
    faith_x = np.array([x[0] - width / 2, x[1] - width / 2, x[2] - width / 2, x[3] - width])
    acc_x = np.array([x[0] + width / 2, x[1] + width / 2, x[2] + width / 2, x[3]])
    abst_x = np.array([x[3] + width])

    bars_faith = ax.bar(
        faith_x,
        faith_vals,
        width,
        label="Faithfulness",
        color="#2563EB",
        edgecolor="#1D4ED8",
        linewidth=0.6,
    )
    bars_acc = ax.bar(
        acc_x,
        acc_vals,
        width,
        label="Answer Accuracy",
        color="#F59E0B",
        edgecolor="#D97706",
        linewidth=0.6,
    )
    bars_abst = ax.bar(
        abst_x,
        [abst_val],
        width,
        label="Abstention Accuracy (Unanswerable only)",
        color="#10B981",
        edgecolor="#059669",
        linewidth=0.6,
    )

    _add_bar_labels(ax, bars_faith)
    _add_bar_labels(ax, bars_acc)
    _add_bar_labels(ax, bars_abst)
    _clean_axes(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.18)
    ax.set_ylabel("Score (0.0 – 1.0)")
    ax.set_title("Generation Quality Metrics Across Question Categories", pad=22)
    fig.suptitle(
        f"LLM: {provider} ({llm_model}) | Judge Mode: {judge_mode}",
        fontsize=9.5,
        color="#475569",
        y=0.925,
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_similarity_distribution(report: dict[str, Any], output_path: Path) -> Path:
    """Save boxplot + jittered scatter of Top-1 cosine similarity: Answerable vs Unanswerable."""
    results = report.get("results", [])
    ans_items = [
        r for r in results if r.get("answerable", True) and r.get("top_similarity") is not None
    ]
    unans_items = [
        r for r in results if not r.get("answerable", True) and r.get("top_similarity") is not None
    ]

    ans_scores = [float(r["top_similarity"]) for r in ans_items]
    unans_scores = [float(r["top_similarity"]) for r in unans_items]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=300)
    data = [ans_scores, unans_scores]
    positions = [1, 2]
    colors = ["#2563EB", "#EF4444"]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0F172A", "linewidth": 1.8},
        whiskerprops={"color": "#64748B", "linewidth": 1.2},
        capprops={"color": "#64748B", "linewidth": 1.2},
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.3)

    rng = np.random.default_rng(42)
    ans_x = rng.normal(loc=1.0, scale=0.055, size=len(ans_scores))
    unans_x = rng.normal(loc=2.0, scale=0.055, size=len(unans_scores))

    ax.scatter(
        ans_x,
        ans_scores,
        color="#1D4ED8",
        alpha=0.85,
        s=48,
        zorder=3,
        label=f"Answerable (n={len(ans_scores)})",
    )
    ax.scatter(
        unans_x,
        unans_scores,
        color="#DC2626",
        alpha=0.85,
        s=52,
        zorder=3,
        label=f"Unanswerable (n={len(unans_scores)})",
    )

    if unans_items:
        top_unans_idx = int(np.argmax(unans_scores))
        top_unans_item = unans_items[top_unans_idx]
        ax.annotate(
            f"{top_unans_item['id']} ({unans_scores[top_unans_idx]:.4f})",
            xy=(unans_x[top_unans_idx], unans_scores[top_unans_idx]),
            xytext=(14, 6),
            textcoords="offset points",
            fontsize=8.5,
            color="#991B1B",
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#991B1B", "lw": 0.9},
        )

    if ans_items:
        low_ans_idx = int(np.argmin(ans_scores))
        low_ans_item = ans_items[low_ans_idx]
        ax.annotate(
            f"{low_ans_item['id']} ({ans_scores[low_ans_idx]:.4f})",
            xy=(ans_x[low_ans_idx], ans_scores[low_ans_idx]),
            xytext=(-78, -14),
            textcoords="offset points",
            fontsize=8.5,
            color="#1E3A8A",
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#1E3A8A", "lw": 0.9},
        )

    for pos, scores, color in zip(positions, data, ["#1E3A8A", "#991B1B"], strict=True):
        if scores:
            mean_val = float(np.mean(scores))
            med_val = float(np.median(scores))
            ax.text(
                pos,
                min(scores) - 0.018 if min(scores) > 0.65 else max(scores) + 0.015,
                f"mean={mean_val:.3f} | med={med_val:.3f}",
                ha="center",
                va="top",
                fontsize=8.8,
                color=color,
                fontweight="bold",
            )

    _clean_axes(ax)
    all_scores = ans_scores + unans_scores
    y_min = max(0.0, (min(all_scores) - 0.04) if all_scores else 0.5)
    y_max = min(1.02, (max(all_scores) + 0.03) if all_scores else 1.0)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"Answerable\n(n={len(ans_scores)})", f"Unanswerable\n(n={len(unans_scores)})"]
    )
    ax.set_ylabel("Top-1 Cosine Similarity Score")
    ax.set_title("Top-1 Retrieval Similarity: Answerable vs. Unanswerable Queries", pad=16)
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_plots(
    results_path: Path | str = Path("eval/eval_results.json"),
    output_dir: Path | str = Path("eval/plots"),
) -> list[Path]:
    """Generate all evaluation PNG charts from `results_path` into `output_dir`."""
    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(
            f"Evaluation results file '{results_file}' not found. Run `python -m eval.run_eval`."
        )

    report: dict[str, Any] = json.loads(results_file.read_text(encoding="utf-8"))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _configure_style()
    outputs = [
        plot_retrieval_metrics(report, out_dir / "retrieval_metrics_by_category.png"),
        plot_generation_quality(report, out_dir / "generation_quality_by_category.png"),
        plot_similarity_distribution(report, out_dir / "similarity_distribution.png"),
    ]
    return outputs


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for chart generation."""
    parser = argparse.ArgumentParser(description="Generate evaluation charts from JSON results.")
    parser.add_argument(
        "--results",
        default="eval/eval_results.json",
        help="Path to eval_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/plots",
        help="Directory to write PNG plots",
    )
    args = parser.parse_args(argv)

    created = generate_all_plots(results_path=args.results, output_dir=args.output_dir)
    for path in created:
        print(f"Saved plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
