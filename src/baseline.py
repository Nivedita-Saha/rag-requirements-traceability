"""TF-IDF vector-space baseline for requirements-to-code trace recovery."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.dataset import Dataset, load_itrust


def rank_tfidf(dataset: Dataset) -> dict[str, list[tuple[str, float]]]:
    """Rank all code artefacts against each requirement by TF-IDF cosine similarity.

    Returns a dict: req_id -> list of (code_id, score) sorted high to low.
    """
    req_ids = list(dataset.requirements)
    code_ids = list(dataset.code)

    req_texts = [dataset.requirements[r] for r in req_ids]
    code_texts = [dataset.code[c] for c in code_ids]

    # Fit one shared vocabulary over requirements and code together, so both
    # sides live in the same vector space and are comparable.
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(req_texts + code_texts)

    req_vectors = matrix[: len(req_ids)]
    code_vectors = matrix[len(req_ids) :]

    # cosine_similarity returns a (n_reqs x n_code) matrix of scores.
    sims = cosine_similarity(req_vectors, code_vectors)

    rankings: dict[str, list[tuple[str, float]]] = {}
    for i, req_id in enumerate(req_ids):
        scored = list(zip(code_ids, sims[i]))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        rankings[req_id] = scored

    return rankings

def evaluate(
    rankings: dict[str, list[tuple[str, float]]],
    gold_links: set[tuple[str, str]],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate rankings against gold links.

    Precision, recall and F1 are computed at cut-off k (top-k retrieved per
    requirement). MAP (mean average precision) uses the full ranking and is
    cut-off independent. Requirements with no gold links are skipped.
    """
    # Group gold links by requirement: req_id -> set of true code_ids.
    gold_by_req: dict[str, set[str]] = {}
    for req_id, code_id in gold_links:
        gold_by_req.setdefault(req_id, set()).add(code_id)

    precisions, recalls, f1s, average_precisions = [], [], [], []

    for req_id, relevant in gold_by_req.items():
        if not relevant or req_id not in rankings:
            continue

        ranked_ids = [code_id for code_id, _ in rankings[req_id]]

        # --- Precision / recall / F1 at k ---
        top_k = ranked_ids[:k]
        hits = sum(1 for code_id in top_k if code_id in relevant)
        precision = hits / k
        recall = hits / len(relevant)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # --- Average precision over the full ranking (for MAP) ---
        num_hits = 0
        sum_prec = 0.0
        for rank, code_id in enumerate(ranked_ids, start=1):
            if code_id in relevant:
                num_hits += 1
                sum_prec += num_hits / rank
        average_precisions.append(sum_prec / len(relevant))

    n = len(precisions)
    return {
        "precision@k": sum(precisions) / n,
        "recall@k": sum(recalls) / n,
        "f1@k": sum(f1s) / n,
        "map": sum(average_precisions) / n,
        "num_requirements": n,
    }

def _save_results(metrics: dict[str, float], path: str) -> None:
    """Write metrics to a JSON file."""
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    dataset = load_itrust()
    rankings = rank_tfidf(dataset)
    metrics = evaluate(rankings, dataset.gold_links, k=10)

    print("TF-IDF baseline on iTrust")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    _save_results(metrics, "results/baseline_tfidf.json")
    print("\nSaved to results/baseline_tfidf.json")