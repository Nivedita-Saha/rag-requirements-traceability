"""Sweep beta for the IDF-weighted confirm variant against the flat baseline."""

from __future__ import annotations

from src.baseline import evaluate
from src.dataset import load_itrust
from src.graph import (
    build_base_graph, add_gold_edges, add_traces_to_edges,
    expand_and_rank, confirm_idf_and_rank,
)

BETAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

if __name__ == "__main__":
    dataset = load_itrust()
    g = build_base_graph(dataset)
    add_gold_edges(g, dataset)
    add_traces_to_edges(g, top_k=1000)

    flat = evaluate(expand_and_rank(g, expand=False), dataset.gold_links, k=10)

    print(f"{'beta':>6}{'P@10':>10}{'R@10':>10}{'F1@10':>10}{'MAP':>10}")
    print("-" * 46)
    print(f"{'flat':>6}{flat['precision@k']:>10.4f}{flat['recall@k']:>10.4f}"
          f"{flat['f1@k']:>10.4f}{flat['map']:>10.4f}")
    print("-" * 46)
    for b in BETAS:
        m = evaluate(confirm_idf_and_rank(g, beta=b), dataset.gold_links, k=10)
        print(f"{b:>6.1f}{m['precision@k']:>10.4f}{m['recall@k']:>10.4f}"
              f"{m['f1@k']:>10.4f}{m['map']:>10.4f}")