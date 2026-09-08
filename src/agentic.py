"""Agentic traceability: a Verifier agent that chooses actions to gather context."""

from __future__ import annotations

import re

import ollama

from src.dataset import Dataset, load_itrust

LLM_MODEL = "llama3.2:3b"
MAX_CODE_CHARS = 2000
STEP_BUDGET = 3


def find_referenced_code(candidate_id: str, dataset: Dataset) -> list[str]:
    """Find other project classes whose ids appear as tokens in the candidate's source.

    This is a lightweight structural expansion: if AddPatientAction's source
    mentions AuthDAO, we treat AuthDAO as related context. Returns a list of
    code_ids (excluding the candidate itself).
    """
    source = dataset.code.get(candidate_id, "")
    if not source:
        return []

    # Tokenise the source into identifier-like words for fast membership checks.
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source))

    referenced = [
        other_id
        for other_id in dataset.code
        if other_id != candidate_id and other_id in tokens
    ]
    return referenced

def _llm_choose_action(requirement: str, candidate_id: str, context: str, steps_left: int) -> str:
    """Ask the agent which action to take. Returns one of EXPAND, RETRIEVE, DECIDE."""
    prompt = f"""You are verifying whether a candidate code class implements a requirement.
You may gather more context before deciding.

REQUIREMENT:
{requirement}

CANDIDATE CLASS: {candidate_id}

CONTEXT GATHERED SO FAR:
{context if context else "(none yet)"}

You have {steps_left} context-gathering step(s) left. Choose ONE action:
- EXPAND: see the classes this candidate references (useful if the link may be indirect)
- RETRIEVE: see more candidate classes retrieved for this requirement
- DECIDE: you have enough information to give a final verdict

Reply with exactly one word: EXPAND, RETRIEVE, or DECIDE."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    text = response["message"]["content"].upper()
    for action in ("EXPAND", "RETRIEVE", "DECIDE"):
        if action in text:
            return action
    return "DECIDE"  # default if unparseable


def _llm_decide(requirement: str, candidate_id: str, context: str) -> tuple[bool, float]:
    """Force a final YES/NO + confidence from the agent."""
    prompt = f"""Decide whether the candidate class implements OR is directly used to carry out
the action described in the requirement, using all context gathered.

Guidance: say YES if the class performs the action, or is a direct part of
performing it (for example a data-access, validation, or action class the
implementation relies on). Say NO if the class is only loosely related — same
general feature area, shared vocabulary, but not actually involved in this
specific action.

REQUIREMENT:
{requirement}

CANDIDATE CLASS: {candidate_id}

CONTEXT:
{context}

Use high confidence only when clear; lower confidence when unsure.
Answer in exactly this format:
VERDICT: YES or NO
CONFIDENCE: a number from 0 to 100"""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    text = response["message"]["content"]
    verdict, confidence = False, 0.0
    for line in text.splitlines():
        upper = line.upper()
        if "VERDICT:" in upper:
            verdict = "YES" in upper.split("VERDICT:", 1)[1]
        elif "CONFIDENCE:" in upper:
            digits = "".join(ch for ch in upper.split("CONFIDENCE:", 1)[1] if ch.isdigit())
            if digits:
                confidence = min(int(digits), 100) / 100.0
    return verdict, confidence

def verify_agentic(
    requirement: str,
    candidate_id: str,
    dataset: Dataset,
    extra_candidates: list[str],
) -> tuple[dict[str, tuple[bool, float]], list[str]]:
    """Run the agent loop for one candidate.

    Returns:
      - a dict of {code_id: (verdict, confidence)} for every class the agent
        actually decided on (the candidate, plus any referenced classes it
        confirmed while expanding)
      - the list of newly proposed code_ids discovered via EXPAND, so the
        caller can fold them into this requirement's candidate set.
    """
    context_parts: list[str] = [
        f"{candidate_id}:\n{dataset.code.get(candidate_id, '')[:MAX_CODE_CHARS]}"
    ]
    discovered: list[str] = []
    retrieve_cursor = 0

    for step in range(STEP_BUDGET):
        steps_left = STEP_BUDGET - step
        context = "\n\n".join(context_parts)
        action = _llm_choose_action(requirement, candidate_id, context, steps_left)

        if action == "DECIDE":
            break
        elif action == "EXPAND":
            refs = find_referenced_code(candidate_id, dataset)
            discovered.extend(refs)
            # Add a compact listing of referenced classes to context.
            listing = ", ".join(refs[:15]) if refs else "(none)"
            context_parts.append(f"Classes referenced by {candidate_id}: {listing}")
        elif action == "RETRIEVE":
            nxt = extra_candidates[retrieve_cursor : retrieve_cursor + 3]
            retrieve_cursor += 3
            discovered.extend(nxt)
            listing = ", ".join(nxt) if nxt else "(no more)"
            context_parts.append(f"Additional retrieved candidates: {listing}")

    # Final decision on the primary candidate.
    context = "\n\n".join(context_parts)
    verdict, confidence = _llm_decide(requirement, candidate_id, context)
    decisions = {candidate_id: (verdict, confidence)}

    return decisions, list(dict.fromkeys(discovered))

def rank_agentic(
    dataset: Dataset,
    candidate_pool: dict[str, list[str]],
) -> dict[str, list[tuple[str, float]]]:
    """Run the agent over the candidate pool, folding in discovered links.

    For each requirement: run the agent on each pooled candidate; collect its
    decision; and for classes discovered via EXPAND/RETRIEVE, score them too
    (single-shot decision, no further expansion) so real indirect links can be
    recovered. Rank all decided classes by verdict-boost + confidence.
    """
    rankings: dict[str, list[tuple[str, float]]] = {}

    for req_id, candidates in candidate_pool.items():
        req_text = dataset.requirements[req_id]
        scores: dict[str, float] = {}
        newly_discovered: set[str] = set()

        # Pass 1: run the agent on each primary candidate.
        for code_id in candidates:
            decisions, discovered = verify_agentic(
                req_text, code_id, dataset, extra_candidates=candidates
            )
            for cid, (verdict, conf) in decisions.items():
                scores[cid] = (1.0 if verdict else 0.0) + conf
            for d in discovered:
                if d not in candidates:
                    newly_discovered.add(d)

        # Pass 2: single-shot decide on discovered classes not already scored.
        for code_id in newly_discovered:
            verdict, conf = _llm_decide(
                req_text, code_id,
                f"{code_id}:\n{dataset.code.get(code_id, '')[:MAX_CODE_CHARS]}",
            )
            score = (1.0 if verdict else 0.0) + conf
            # Keep the better score if it was somehow already present.
            scores[code_id] = max(scores.get(code_id, 0.0), score)

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        rankings[req_id] = ranked

    return rankings

if __name__ == "__main__":
    from src.baseline import rank_tfidf, evaluate, _save_results
    from src.semantic import build_code_index, rank_semantic
    from src.rag_verify import build_candidate_pool

    dataset = load_itrust()

    print("Building TF-IDF rankings...")
    tfidf_rankings = rank_tfidf(dataset)

    print("Building semantic index and rankings...")
    collection = build_code_index(dataset)
    semantic_rankings = rank_semantic(dataset, collection, top_k=10)

    print("Building candidate pool (union of both)...")
    pool = build_candidate_pool(tfidf_rankings, semantic_rankings, top_n=10)

    print(f"Running agentic verification over {len(pool)} requirements — this is the slowest run.\n")

    rankings: dict[str, list[tuple[str, float]]] = {}
    for i, (req_id, candidates) in enumerate(pool.items(), start=1):
        req_text = dataset.requirements[req_id]
        scores: dict[str, float] = {}
        newly_discovered: set[str] = set()

        for code_id in candidates:
            decisions, discovered = verify_agentic(
                req_text, code_id, dataset, extra_candidates=candidates
            )
            for cid, (verdict, conf) in decisions.items():
                scores[cid] = (1.0 if verdict else 0.0) + conf
            for d in discovered:
                if d not in candidates:
                    newly_discovered.add(d)

        for code_id in newly_discovered:
            verdict, conf = _llm_decide(
                req_text, code_id,
                f"{code_id}:\n{dataset.code.get(code_id, '')[:MAX_CODE_CHARS]}",
            )
            score = (1.0 if verdict else 0.0) + conf
            scores[code_id] = max(scores.get(code_id, 0.0), score)

        rankings[req_id] = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        print(f"  [{i}/{len(pool)}] {req_id} done")

    metrics = evaluate(rankings, dataset.gold_links, k=10)

    print("\nAgentic traceability on iTrust")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    _save_results(metrics, "results/agentic.json")
    print("\nSaved to results/agentic.json")