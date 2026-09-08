"""Load all method results, print a comparison table, and plot the metrics."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = {
    "TF-IDF": "results/baseline_tfidf.json",
    "Semantic": "results/semantic_nomic.json",
    "RAG + LLM": "results/rag_llm.json",
    "Agentic": "results/agentic.json",
}

METRICS = ["precision@k", "recall@k", "f1@k", "map"]


def load_all() -> dict[str, dict]:
    """Load every method's metrics JSON into a dict keyed by method name."""
    data: dict[str, dict] = {}
    for name, path in RESULTS.items():
        with open(path, encoding="utf-8") as f:
            data[name] = json.load(f)
    return data


def print_table(data: dict[str, dict]) -> None:
    """Print a plain-text comparison table across methods and metrics."""
    header = f"{'Method':<12}" + "".join(f"{m:>14}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for name in RESULTS:
        row = f"{name:<12}"
        for m in METRICS:
            row += f"{data[name][m]:>14.4f}"
        print(row)

def plot_metrics(data: dict[str, dict], out_path: str = "results/comparison.png") -> None:
    """Grouped bar chart: four methods across the four metrics."""
    import numpy as np
    import matplotlib.pyplot as plt

    methods = list(RESULTS)
    x = np.arange(len(METRICS))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(methods):
        values = [data[name][m] for m in METRICS]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision@10", "Recall@10", "F1@10", "MAP"])
    ax.set_ylabel("Score")
    ax.set_title("Requirements-to-code trace recovery on iTrust")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


def plot_slope(data: dict[str, dict], out_path: str = "results/slope.png") -> None:
    """Slope chart: each metric as a line across the four methods in pipeline order."""
    import matplotlib.pyplot as plt

    methods = list(RESULTS)  # TF-IDF, Semantic, RAG + LLM, Agentic (pipeline order)
    x = list(range(len(methods)))

    metric_labels = {
        "precision@k": "Precision@10",
        "recall@k": "Recall@10",
        "f1@k": "F1@10",
        "map": "MAP",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for metric in METRICS:
        y = [data[name][metric] for name in methods]
        line, = ax.plot(x, y, marker="o", markersize=7, linewidth=2)
        # Label each line at its right-hand end.
        ax.annotate(
            f"{metric_labels[metric]}  {y[-1]:.3f}",
            xy=(x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=line.get_color(),
            fontweight="bold",
        )
        # Value at the starting point too, for reference.
        ax.annotate(
            f"{y[0]:.3f}",
            xy=(x[0], y[0]),
            xytext=(-8, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            fontsize=9,
            color=line.get_color(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("Trace-recovery performance across the method pipeline (iTrust)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.margins(x=0.15)  # room for the right-hand labels

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved slope chart to {out_path}")


if __name__ == "__main__":
    all_data = load_all()
    print_table(all_data)
    print()
    plot_metrics(all_data)
    plot_slope(all_data)