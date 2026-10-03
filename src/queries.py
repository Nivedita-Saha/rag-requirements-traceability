"""Traceability queries over the knowledge graph (NetworkX stand-ins for Cypher)."""

from __future__ import annotations

import networkx as nx

from src.graph import REQUIREMENT, CODE, GOLD, TRACES_TO


def implements(g: nx.MultiDiGraph, req_id: str, source: str = GOLD) -> list[str]:
    """Code classes linked to a requirement.

    source=GOLD -> ground-truth implementers; source=TRACES_TO -> classes
    recovered by Project 2. Returns [] if there are none.
    """
    if not g.has_node(req_id):
        raise KeyError(f"no such requirement: {req_id}")
    return sorted(
        code_id
        for _, code_id, key in g.out_edges(req_id, keys=True)
        if key == source
    )


def orphan_requirements(g: nx.MultiDiGraph, source: str = GOLD) -> list[str]:
    """Requirements with no linked code under the given edge type."""
    out = []
    for n, d in g.nodes(data=True):
        if d["type"] != REQUIREMENT:
            continue
        if not any(key == source for _, _, key in g.out_edges(n, keys=True)):
            out.append(n)
    return sorted(out)


def orphan_code(g: nx.MultiDiGraph, source: str = GOLD) -> list[str]:
    """Code classes no requirement links to under the given edge type."""
    linked = {v for _, v, key in g.edges(keys=True) if key == source}
    return sorted(
        n for n, d in g.nodes(data=True)
        if d["type"] == CODE and n not in linked
    )


if __name__ == "__main__":
    from src.dataset import load_itrust
    from src.graph import build_base_graph, add_gold_edges, add_traces_to_edges

    dataset = load_itrust()
    g = build_base_graph(dataset)
    add_gold_edges(g, dataset)
    add_traces_to_edges(g, top_k=1000)

    code_total = sum(1 for _, d in g.nodes(data=True) if d["type"] == CODE)

    orphans = orphan_requirements(g)
    print(f"Orphan requirements (no gold-linked code): {len(orphans)}")
    print("  " + (", ".join(orphans) if orphans else "(none)"))

    # Example query on a requirement that DOES have gold links.
    example = next(
        n for n, d in g.nodes(data=True)
        if d["type"] == REQUIREMENT
        and any(k == GOLD for _, _, k in g.out_edges(n, keys=True))
    )
    gold_impl = implements(g, example, source=GOLD)
    recovered = implements(g, example, source=TRACES_TO)
    print(f"\nRequirement {example}:")
    print(f"  gold implementers ({len(gold_impl)}): {', '.join(gold_impl)}")
    print(f"  recovered links ({len(recovered)}): {', '.join(recovered[:10])}"
          f"{' ...' if len(recovered) > 10 else ''}")

    print(f"\nOrphan code (no gold-linked requirement): {len(orphan_code(g))} / {code_total}")