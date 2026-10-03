"""Typed knowledge graph over iTrust artefacts, built on NetworkX."""

from __future__ import annotations

import networkx as nx
import json
import math
from pathlib import Path

from src.dataset import Dataset, load_itrust
from src.agentic import find_referenced_code

# Node types (stored as the "type" attribute on each node).
REQUIREMENT = "Requirement"
CODE = "Code"

# Edge types (stored as the edge key in the MultiDiGraph).
DEPENDS_ON = "depends_on"  # Code -> Code        (A's source references class B)
TRACES_TO = "traces_to"    # Requirement -> Code (recovered in Project 2)
GOLD = "gold"              # Requirement -> Code (ground-truth link)


def build_base_graph(dataset: Dataset) -> nx.MultiDiGraph:
    """Graph with Requirement and Code nodes plus structural depends_on edges.

    traces_to and gold edges are added in a later step: they come from
    Project 2's recovered rankings and the dataset's answer matrix.
    """
    g = nx.MultiDiGraph()

    for req_id, text in dataset.requirements.items():
        g.add_node(req_id, type=REQUIREMENT, text=text)

    # In iTrust a code artefact is one Java class, so Code node == class.
    for code_id, source in dataset.code.items():
        g.add_node(code_id, type=CODE, source=source)

    # Structural edges: reuse the reference extractor from the agentic method.
    for code_id in dataset.code:
        for referenced in find_referenced_code(code_id, dataset):
            g.add_edge(code_id, referenced, key=DEPENDS_ON)

    return g

def add_gold_edges(g: nx.MultiDiGraph, dataset: Dataset) -> int:
    """Add ground-truth Requirement -> Code edges from the answer matrix.

    Returns the number of gold edges added. These are the links every
    evaluation is scored against.
    """
    added = 0
    for req_id, code_id in dataset.gold_links:
        # Both endpoints already exist as nodes; guard anyway.
        if g.has_node(req_id) and g.has_node(code_id):
            g.add_edge(req_id, code_id, key=GOLD)
            added += 1
    return added


def add_traces_to_edges(
    g: nx.MultiDiGraph,
    rankings_path: str = "results/agentic_rankings.json",
    top_k: int = 10,
) -> int:
    """Add recovered Requirement -> Code edges from Project 2's agentic rankings.

    Loads the cached rankings and adds the top_k recovered code artefacts per
    requirement as traces_to edges, carrying the recovered score as an edge
    attribute. Returns the number of traces_to edges added.
    """
    with open(rankings_path, encoding="utf-8") as f:
        rankings = json.load(f)  # {req_id: [[code_id, score], ...]}

    added = 0
    for req_id, ranked in rankings.items():
        for rank, (code_id, score) in enumerate(ranked[:top_k]):
            if g.has_node(req_id) and g.has_node(code_id):
                g.add_edge(req_id, code_id, key=TRACES_TO, score=score, rank=rank)
                added += 1
    return added

def expand_and_rank(
    g: nx.MultiDiGraph,
    decay: float = 0.5,
    top_k: int = 10,
    expand: bool = True,
) -> dict[str, list[tuple[str, float]]]:
    """Graph-aware retrieval: expand each recovered link along depends_on.

    Seeds are Project 2's traces_to links; when expand is True, the classes
    those seeds depend on are added at the parent's score times `decay`.
    Ranking is ordered by (score desc, agentic rank asc, id) — a tie-break
    carried on the traces_to edges themselves, never graph insertion order,
    so gold edges cannot leak into the ranking.
    """
    BIG_RANK = 10**9  # pushes expansion-only classes below any tied seed
    rankings: dict[str, list[tuple[str, float]]] = {}

    req_ids = [n for n, d in g.nodes(data=True) if d["type"] == REQUIREMENT]
    for req_id in req_ids:
        scores: dict[str, float] = {}
        seed_rank: dict[str, int] = {}

        # Seed: flat recovered links, carrying score and agentic rank.
        for _, code_id, key, data in g.out_edges(req_id, keys=True, data=True):
            if key == TRACES_TO:
                if code_id not in scores or data["score"] > scores[code_id]:
                    scores[code_id] = data["score"]
                    seed_rank[code_id] = data["rank"]

        # Expand: classes each seed depends on, at a discounted score.
        if expand:
            for code_id, parent_score in list(scores.items()):
                for _, neighbour, key in g.out_edges(code_id, keys=True):
                    if key == DEPENDS_ON:
                        boosted = parent_score * decay
                        if boosted > scores.get(neighbour, 0.0):
                            scores[neighbour] = boosted
                        seed_rank.setdefault(neighbour, BIG_RANK)

        # Gold-independent, reproducible ordering.
        ranked = sorted(
            scores.items(),
            key=lambda pair: (-pair[1], seed_rank.get(pair[0], BIG_RANK), pair[0]),
        )
        rankings[req_id] = ranked[: max(top_k, len(ranked))]

    return rankings

def confirm_and_rank(
    g: nx.MultiDiGraph,
    beta: float = 0.3,
    top_k: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """Graph-aware re-ranking: confirm EXISTING candidates by neighbourhood support.

    No new classes are added. A candidate is boosted when high-scoring seeds
    depend on it (e.g. several strong Action classes all referencing one DAO),
    so the graph signal competes at the top of the ranking rather than being
    buried beneath the direct links. Same leak-free (score, rank, id) tie-break.
    """
    BIG_RANK = 10**9
    rankings: dict[str, list[tuple[str, float]]] = {}

    req_ids = [n for n, d in g.nodes(data=True) if d["type"] == REQUIREMENT]
    for req_id in req_ids:
        base: dict[str, float] = {}
        seed_rank: dict[str, int] = {}
        for _, code_id, key, data in g.out_edges(req_id, keys=True, data=True):
            if key == TRACES_TO:
                if code_id not in base or data["score"] > base[code_id]:
                    base[code_id] = data["score"]
                    seed_rank[code_id] = data["rank"]

        # Support: sum the scores of seeds that depend on each candidate.
        support: dict[str, float] = {c: 0.0 for c in base}
        for seed, seed_score in base.items():
            for _, dep, key in g.out_edges(seed, keys=True):
                if key == DEPENDS_ON and dep in base:  # confirm existing only
                    support[dep] += seed_score

        scores = {c: base[c] + beta * support[c] for c in base}
        ranked = sorted(
            scores.items(),
            key=lambda pair: (-pair[1], seed_rank.get(pair[0], BIG_RANK), pair[0]),
        )
        rankings[req_id] = ranked[: max(top_k, len(ranked))]

    return rankings

def confirm_idf_and_rank(
    g: nx.MultiDiGraph,
    beta: float = 0.3,
    top_k: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """Confirm variant that down-weights hub classes by inverse reference frequency.

    Same neighbourhood-support idea as confirm_and_rank, but each candidate's
    support is scaled by idf = log((N+1)/(indeg+1)), where indeg is how many
    classes depend on it project-wide. Generic hubs (high in-degree) are damped;
    rare, specific classes keep their support. beta=0 reduces to the flat
    baseline. Leak-free (score, rank, id) tie-break, as before.
    """
    BIG_RANK = 10**9
    code_nodes = [n for n, d in g.nodes(data=True) if d["type"] == CODE]
    N = len(code_nodes)

    indeg: dict[str, int] = {c: 0 for c in code_nodes}
    for _, v, key in g.edges(keys=True):
        if key == DEPENDS_ON:
            indeg[v] = indeg.get(v, 0) + 1
    idf = {c: math.log((N + 1) / (indeg.get(c, 0) + 1)) for c in code_nodes}

    rankings: dict[str, list[tuple[str, float]]] = {}
    req_ids = [n for n, d in g.nodes(data=True) if d["type"] == REQUIREMENT]
    for req_id in req_ids:
        base: dict[str, float] = {}
        seed_rank: dict[str, int] = {}
        for _, code_id, key, data in g.out_edges(req_id, keys=True, data=True):
            if key == TRACES_TO:
                if code_id not in base or data["score"] > base[code_id]:
                    base[code_id] = data["score"]
                    seed_rank[code_id] = data["rank"]

        support: dict[str, float] = {c: 0.0 for c in base}
        for seed, seed_score in base.items():
            for _, dep, key in g.out_edges(seed, keys=True):
                if key == DEPENDS_ON and dep in base:
                    support[dep] += seed_score * idf.get(dep, 0.0)

        scores = {c: base[c] + beta * support[c] for c in base}
        ranked = sorted(
            scores.items(),
            key=lambda pair: (-pair[1], seed_rank.get(pair[0], BIG_RANK), pair[0]),
        )
        rankings[req_id] = ranked[: max(top_k, len(ranked))]

    return rankings


if __name__ == "__main__":
    import json
    from pathlib import Path

    from src.baseline import evaluate

    DECAY = 0.3
    BETA = 0.3
    SEED_DEPTH = 1000

    dataset = load_itrust()
    g = build_base_graph(dataset)
    add_gold_edges(g, dataset)
    add_traces_to_edges(g, top_k=SEED_DEPTH)

    results = {
        "flat": evaluate(expand_and_rank(g, expand=False), dataset.gold_links, k=10),
        "expand": evaluate(expand_and_rank(g, decay=DECAY, expand=True), dataset.gold_links, k=10),
        "confirm": evaluate(confirm_and_rank(g, beta=BETA), dataset.gold_links, k=10),
    }

    print(f"{'Metric':<16}{'Flat (P2)':>12}{'Expand':>10}{'Confirm':>10}")
    print("-" * 48)
    for m in ("precision@k", "recall@k", "f1@k", "map"):
        print(f"{m:<16}{results['flat'][m]:>12.4f}{results['expand'][m]:>10.4f}{results['confirm'][m]:>10.4f}")
    print(f"{'num_requirements':<16}{results['flat']['num_requirements']:>12}"
          f"{results['expand']['num_requirements']:>10}{results['confirm']['num_requirements']:>10}")

    out = {"params": {"decay": DECAY, "beta": BETA, "seed_depth": SEED_DEPTH, "k": 10}, **results}
    Path("results/graph_comparison.json").parent.mkdir(parents=True, exist_ok=True)
    with open("results/graph_comparison.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results/graph_comparison.json")