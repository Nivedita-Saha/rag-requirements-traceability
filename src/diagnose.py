"""Pin down HOW graph-flat diverges from the JSON ranking: set, length, or order."""

from __future__ import annotations

import json

from src.dataset import load_itrust
from src.graph import build_base_graph, add_gold_edges, add_traces_to_edges, expand_and_rank

dataset = load_itrust()

with open("results/agentic_rankings.json", encoding="utf-8") as f:
    raw = json.load(f)
json_rank = {r: [cid for cid, _ in lst] for r, lst in raw.items()}

g = build_base_graph(dataset)
add_gold_edges(g, dataset)
add_traces_to_edges(g, top_k=1000)
graph_full = expand_and_rank(g, expand=False)
graph_rank = {r: [cid for cid, _ in lst] for r, lst in graph_full.items()}

set_diff = length_diff = order_diff = 0
for r in json_rank:
    j, gr = json_rank[r], graph_rank.get(r, [])
    if set(j) != set(gr):
        set_diff += 1
    if len(j) != len(gr):
        length_diff += 1
    if j != gr:
        order_diff += 1

print(f"requirements:      {len(json_rank)}")
print(f"differ in SET:     {set_diff}")
print(f"differ in LENGTH:  {length_diff}")
print(f"differ in ORDER:   {order_diff}")

# One concrete example of a requirement that diverges.
for r in json_rank:
    if json_rank[r] != graph_rank.get(r, []):
        print(f"\nexample req: {r}")
        print(f"  json len {len(raw[r])}, graph len {len(graph_full[r])}")
        print("  json  first 8:", [(c, round(s, 4)) for c, s in raw[r][:8]])
        print("  graph first 8:", [(c, round(s, 4)) for c, s in graph_full[r][:8]])
        break