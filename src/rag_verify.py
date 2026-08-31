"""RAG + LLM verification: retrieve candidates, then verify each with a local LLM."""

from __future__ import annotations

import ollama

from src.dataset import Dataset, load_itrust

LLM_MODEL = "llama3.2:3b"

# Truncate code shown to the LLM so the prompt stays within context.
MAX_CODE_CHARS = 2500


def build_candidate_pool(
    tfidf_rankings: dict[str, list[tuple[str, float]]],
    semantic_rankings: dict[str, list[tuple[str, float]]],
    top_n: int = 10,
) -> dict[str, list[str]]:
    """Union the top-N candidates from both retrievers, per requirement.

    Returns req_id -> list of candidate code_ids (order not important here;
    the LLM verifier will re-rank them).
    """
    pool: dict[str, list[str]] = {}
    all_reqs = set(tfidf_rankings) | set(semantic_rankings)

    for req_id in all_reqs:
        tfidf_top = [cid for cid, _ in tfidf_rankings.get(req_id, [])[:top_n]]
        semantic_top = [cid for cid, _ in semantic_rankings.get(req_id, [])[:top_n]]
        # dict.fromkeys preserves order and de-duplicates.
        pool[req_id] = list(dict.fromkeys(tfidf_top + semantic_top))

    return pool

def verify_link(requirement: str, code: str) -> tuple[bool, float]:
    """Ask the LLM whether the code implements the requirement.

    Uses chain-of-thought: the model reasons briefly, then gives a verdict.
    Returns (is_link, confidence in 0..1). Falls back to (False, 0.0) if the
    response can't be parsed.
    """
    prompt = f"""You are analysing whether a piece of code implements a software requirement.

REQUIREMENT:
{requirement}

CODE:
{code[:MAX_CODE_CHARS]}

Think step by step in one or two sentences about whether this code implements,
or directly contributes to implementing, the requirement. Then give your final
answer on a new line in exactly this format:
VERDICT: YES or NO
CONFIDENCE: a number from 0 to 100"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    text = response["message"]["content"]

    # Parse the verdict and confidence out of the response.
    verdict = False
    confidence = 0.0
    for line in text.splitlines():
        upper = line.upper()
        if "VERDICT:" in upper:
            verdict = "YES" in upper.split("VERDICT:", 1)[1]
        elif "CONFIDENCE:" in upper:
            digits = "".join(ch for ch in upper.split("CONFIDENCE:", 1)[1] if ch.isdigit())
            if digits:
                confidence = min(int(digits), 100) / 100.0

    return verdict, confidence

def rank_rag(
    dataset: Dataset,
    candidate_pool: dict[str, list[str]],
) -> dict[str, list[tuple[str, float]]]:
    """Verify every candidate with the LLM and rank by a combined score.

    Score = verdict_boost + confidence, so a YES always outranks a NO, and
    within each group higher confidence ranks higher. This keeps NO candidates
    in the ranking (protecting recall) rather than discarding them.
    """
    rankings: dict[str, list[tuple[str, float]]] = {}

    for req_id, candidates in candidate_pool.items():
        req_text = dataset.requirements[req_id]
        scored: list[tuple[str, float]] = []

        for code_id in candidates:
            verdict, confidence = verify_link(req_text, dataset.code[code_id])
            score = (1.0 if verdict else 0.0) + confidence
            scored.append((code_id, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        rankings[req_id] = scored

    return rankings

if __name__ == "__main__":
    from src.baseline import rank_tfidf, evaluate, _save_results
    from src.semantic import build_code_index, rank_semantic

    dataset = load_itrust()

    print("Building TF-IDF rankings...")
    tfidf_rankings = rank_tfidf(dataset)

    print("Building semantic index and rankings...")
    collection = build_code_index(dataset)
    semantic_rankings = rank_semantic(dataset, collection, top_k=10)

    print("Building candidate pool (union of both)...")
    pool = build_candidate_pool(tfidf_rankings, semantic_rankings, top_n=10)

    total_candidates = sum(len(v) for v in pool.values())
    print(f"Verifying {total_candidates} candidate links with {LLM_MODEL} — this is slow.\n")

    # Verify with a simple progress counter.
    rankings: dict[str, list[tuple[str, float]]] = {}
    for i, (req_id, candidates) in enumerate(pool.items(), start=1):
        req_text = dataset.requirements[req_id]
        scored = []
        for code_id in candidates:
            verdict, confidence = verify_link(req_text, dataset.code[code_id])
            scored.append((code_id, (1.0 if verdict else 0.0) + confidence))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        rankings[req_id] = scored
        print(f"  [{i}/{len(pool)}] {req_id} done")

    metrics = evaluate(rankings, dataset.gold_links, k=10)

    print("\nRAG + LLM verification on iTrust")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    _save_results(metrics, "results/rag_llm.json")
    print("\nSaved to results/rag_llm.json")