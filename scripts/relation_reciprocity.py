"""Survey per-relation reciprocity and forward/inverse-CS symmetry across the corpus.

Bidirectionality (an undirected pair connected both ways) is what makes a synthetic
graph's simple edge count match the original (see
`docs/notes/motif_reachability_and_edge_multiplicity.md`). This script tests the
hypothesis that bidirectionality is carried by a **per-relation reciprocity** — that
relations split cleanly into *symmetric* (whenever a→b via r, also b→a via r; its
subject and object entity sets coincide, so its forward CS = inverse CS) and
*asymmetric* (disjoint sides, no reciprocation) — rather than being an entity-level
correlation. If so, reproducing per-relation reciprocity (a Block-B-style quantile
feature) + a shared entity pool for symmetric relations reproduces bidirectionality.

Per graph, over entity–entity content edges (exclude rdf:type, literal endpoints,
self-loops):

    same_rel_reciprocity[r] = fraction of r's directed edges (a→b) with (b→a) also via r
    cs_symmetry[r]          = |S_r ∩ O_r| / |S_r ∪ O_r|   (entities both emit and receive r)
    overall reciprocity     = edge-weighted mean of same_rel_reciprocity
    any_rel_bidir_frac      = fraction of directed pairs whose reverse exists via ANY relation
    symmetric_edge_frac     = fraction of edges in relations with reciprocity > 0.5 (bimodality)
    cs_invcs_jaccard        = per-entity Jaccard(forward CS, inverse CS), mean/median

Usage
-----
    python scripts/relation_reciprocity.py fb237_v4 wn18rr_v4
    python scripts/relation_reciprocity.py            # all corpus graphs
    python scripts/relation_reciprocity.py --top 15   # show 15 relations per graph
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
from edge_multiplicity import _find_source_file, _corpus_graphs, _graph_dir  # noqa: E402
from kgsynth.generator._constants import _RDF_TYPE  # noqa: E402
from kgsynth.kg_io import load_kg  # noqa: E402

_OUT_DIR = _REPO / "experiments" / "relation_reciprocity"


def _measure(name: str, top: int):
    graph_dir = _graph_dir(name)
    if graph_dir is None:
        print(f"  {name}: not in corpus — skipping")
        return None, None
    src = _find_source_file(graph_dir)
    print(f"  {name}: loading {src.name} …", flush=True)
    g = load_kg(src)
    is_lit = (g.vs["is_literal"] if "is_literal" in g.vertex_attributes()
              else [False] * g.vcount())

    cs: dict[int, set] = defaultdict(set)          # entity → relations emitted
    inv_cs: dict[int, set] = defaultdict(set)      # entity → relations received
    Sr: dict[str, set] = defaultdict(set)
    Or: dict[str, set] = defaultdict(set)
    edges: dict[str, set] = defaultdict(set)       # relation → {(s,o)}
    dir_pairs: set = set()
    for e in g.es:
        if e["predicate"] == _RDF_TYPE:
            continue
        s, o = e.source, e.target
        if s == o or is_lit[s] or is_lit[o]:
            continue
        r = e["predicate"]
        cs[s].add(r); inv_cs[o].add(r)
        Sr[r].add(s); Or[r].add(o); edges[r].add((s, o))
        dir_pairs.add((s, o))

    n_edges = sum(len(v) for v in edges.values())
    if n_edges == 0:
        print(f"  {name}: no content edges — skipping")
        return None, None

    # per-relation reciprocity + CS symmetry
    per_rel = []
    recip_edges = 0
    for r, es in edges.items():
        recip = sum(1 for (s, o) in es if (o, s) in es)
        recip_edges += recip
        sym = len(Sr[r] & Or[r]) / max(1, len(Sr[r] | Or[r]))
        per_rel.append({"relation": r.rsplit("/", 1)[-1], "edges": len(es),
                        "reciprocity": round(recip / len(es), 4), "cs_symmetry": round(sym, 4)})
    per_rel.sort(key=lambda d: -d["edges"])

    # any-relation bidirectional pairs (matches bidirectional_ratio's numerator)
    und = {(min(s, o), max(s, o)) for (s, o) in dir_pairs}
    any_bidir = (len(dir_pairs) - len(und)) / max(1, len(dir_pairs))

    ents = set(cs) | set(inv_cs)
    jac = np.array([len(cs[v] & inv_cs[v]) / max(1, len(cs[v] | inv_cs[v])) for v in ents])
    sym_edge_frac = sum(d["edges"] for d in per_rel if d["reciprocity"] > 0.5) / n_edges

    summary = {
        "graph": name,
        "relations": len(edges),
        "content_edges": n_edges,
        "overall_reciprocity": round(recip_edges / n_edges, 4),
        "any_rel_bidir_pair_frac": round(any_bidir, 4),
        "symmetric_edge_frac": round(sym_edge_frac, 4),
        "n_symmetric_rel": sum(1 for d in per_rel if d["reciprocity"] > 0.5),
        "n_asymmetric_rel": sum(1 for d in per_rel if d["reciprocity"] < 0.1),
        "cs_invcs_jaccard_mean": round(float(jac.mean()), 4),
        "cs_invcs_jaccard_median": round(float(np.median(jac)), 4),
        # bimodality: how few relations sit in the ambiguous middle (0.1..0.5)
        "mid_band_edge_frac": round(
            sum(d["edges"] for d in per_rel if 0.1 <= d["reciprocity"] <= 0.5) / n_edges, 4),
    }
    for d in per_rel[:top]:
        print(f"      {d['relation'][:24]:<24} edges={d['edges']:>7}  "
              f"recip={d['reciprocity']:.2f}  cs_sym={d['cs_symmetry']:.2f}")
    return summary, per_rel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs", nargs="*", help="graph names (default: all corpus graphs)")
    ap.add_argument("--top", type=int, default=10, help="relations to print per graph")
    ap.add_argument("--out", type=Path, default=_OUT_DIR)
    args = ap.parse_args()

    graphs = args.graphs or _corpus_graphs()
    print(f"Relation-reciprocity survey for {len(graphs)} graph(s): {', '.join(graphs)}\n")

    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for name in graphs:
        summary, per_rel = _measure(name, args.top)
        if summary is None:
            continue
        rows.append(summary)
        with open(args.out / f"{name}_per_relation.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_rel[0]))
            w.writeheader(); w.writerows(per_rel)
    if not rows:
        raise SystemExit("no graphs measured")

    with open(args.out / "reciprocity_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'graph':>14}  {'overall':>8}  {'bidir_pair':>10}  {'symm_edge%':>10}  "
          f"{'midband%':>9}  {'CS∩invCS jac':>12}")
    print("-" * 78)
    for r in rows:
        print(f"{r['graph']:>14}  {r['overall_reciprocity']:>8.2f}  "
              f"{r['any_rel_bidir_pair_frac']:>10.2f}  {r['symmetric_edge_frac']:>10.2f}  "
              f"{r['mid_band_edge_frac']:>9.2f}  {r['cs_invcs_jaccard_mean']:>12.2f}")
    print("\noverall = edge-wtd same-rel reciprocity;  symm_edge% = edges in relations with "
          "reciprocity>0.5;\nmidband% = edges in relations with reciprocity 0.1–0.5 "
          "(low ⇒ bimodal / cleanly split).")
    print(f"→ {args.out}/reciprocity_summary.csv")


if __name__ == "__main__":
    main()
