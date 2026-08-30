"""Semantic retrieval baseline using nomic-embed-text (via Ollama) and ChromaDB."""

from __future__ import annotations

import ollama

from src.dataset import Dataset, load_itrust

EMBED_MODEL = "nomic-embed-text"

# nomic-embed-text handles ~8k tokens; truncate very long files so a single
# embedding stays meaningful and within the model's context window.
MAX_CHARS = 4000


def embed_text(text: str) -> list[float]:
    """Return the embedding vector for a single piece of text."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text[:MAX_CHARS])
    return response["embedding"]

def build_code_index(dataset: Dataset):
    """Embed all code artefacts and store them in an in-memory ChromaDB collection.

    Returns the collection, queryable by embedding for top-k retrieval.
    """
    import chromadb

    client = chromadb.Client()

    # Fresh collection each run: delete if a previous run left one behind.
    try:
        client.delete_collection("code")
    except Exception:
        pass
    collection = client.create_collection(
        name="code", metadata={"hnsw:space": "cosine"}
    )

    code_ids = list(dataset.code)
    embeddings = []
    for code_id in code_ids:
        embeddings.append(embed_text(dataset.code[code_id]))

    collection.add(ids=code_ids, embeddings=embeddings)
    return collection

def rank_semantic(dataset: Dataset, collection, top_k: int = 10) -> dict[str, list[tuple[str, float]]]:
    """Rank code artefacts against each requirement by embedding similarity.

    Returns req_id -> list of (code_id, score) sorted high to low, where score
    is similarity (1 - cosine distance). Only the top_k are returned per
    requirement, which is all the metrics at k need.
    """
    rankings: dict[str, list[tuple[str, float]]] = {}

    for req_id, req_text in dataset.requirements.items():
        query_vec = embed_text(req_text)
        result = collection.query(query_embeddings=[query_vec], n_results=top_k)

        ids = result["ids"][0]
        distances = result["distances"][0]

        # ChromaDB returns cosine distance; convert to similarity for ranking.
        scored = [(code_id, 1.0 - dist) for code_id, dist in zip(ids, distances)]
        rankings[req_id] = scored

    return rankings

if __name__ == "__main__":
    from src.baseline import evaluate, _save_results

    dataset = load_itrust()
    collection = build_code_index(dataset)

    # Retrieve a deeper list so MAP is computed over a meaningful ranking,
    # while precision/recall/F1 still report at k=10.
    rankings = rank_semantic(dataset, collection, top_k=50)
    metrics = evaluate(rankings, dataset.gold_links, k=10)

    print("Semantic retrieval (nomic-embed-text) on iTrust")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    _save_results(metrics, "results/semantic_nomic.json")
    print("\nSaved to results/semantic_nomic.json")