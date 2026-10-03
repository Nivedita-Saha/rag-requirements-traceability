"""Visualise one requirement's trace neighbourhood with pyvis (for the README)."""

from __future__ import annotations

from pyvis.network import Network

from src.dataset import load_itrust
from src.graph import (
    build_base_graph, add_gold_edges, add_traces_to_edges,
    GOLD, TRACES_TO, DEPENDS_ON,
)

REQ = "UC10E1"        # requirement to centre the picture on
TOP_RECOVERED = 8     # how many recovered links to show (by agentic rank)
OUT = "results/graph_sample.html"

COL_REQ = "#b31b1b"    # requirement      (red)
COL_GOLD = "#2e7d32"   # gold implementer (green)
COL_RECOV = "#1565c0"  # recovered only   (blue)


def main() -> None:
    dataset = load_itrust()
    g = build_base_graph(dataset)
    add_gold_edges(g, dataset)
    add_traces_to_edges(g, top_k=1000)

    if not g.has_node(REQ):
        raise SystemExit(f"{REQ} not in graph")

    gold = {c for _, c, k in g.out_edges(REQ, keys=True) if k == GOLD}
    recov = sorted(
        ((c, d["rank"]) for _, c, k, d in g.out_edges(REQ, keys=True, data=True)
         if k == TRACES_TO),
        key=lambda pair: pair[1],
    )
    recov_ids = [c for c, _ in recov[:TOP_RECOVERED]]
    shown = list(dict.fromkeys(list(gold) + recov_ids))  # gold always shown

    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff")
    net.set_options("""
    {
      "nodes": {
        "font": {"size": 15, "face": "arial", "background": "rgba(255,255,255,0.85)"}
      },
      "edges": {"smooth": {"type": "continuous"}},
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -32000,
          "centralGravity": 0.15,
          "springLength": 260,
          "springConstant": 0.02,
          "damping": 0.35
        },
        "minVelocity": 0.5,
        "stabilization": {"iterations": 350}
      }
    }
    """)
    net.add_node(REQ, label=REQ, color=COL_REQ, shape="box", size=28)
    for c in shown:
        net.add_node(c, label=c, color=(COL_GOLD if c in gold else COL_RECOV), size=18)
        if c in gold:
            net.add_edge(REQ, c, color=COL_GOLD, width=3, title="gold link")
        else:
            net.add_edge(REQ, c, color=COL_RECOV, dashes=True, title="recovered link")

    shown_set = set(shown)
    for u in shown:
        for _, v, k in g.out_edges(u, keys=True):
            if k == DEPENDS_ON and v in shown_set:
                net.add_edge(u, v, color="#9e9e9e", width=1, title="depends_on")

    net.write_html(OUT, notebook=False)
    print(f"Wrote {OUT}")
    print(f"  requirement {REQ}: {len(gold)} gold (green), {len(recov_ids)} recovered (blue)")
    print("Open it in a browser and screenshot for the README.")


if __name__ == "__main__":
    main()