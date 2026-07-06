"""Measure the directed→simple **edge-multiplicity (pair-overlap)** gap between a
graph's original and its Stage-2 synthetic output, across the corpus.

Motifs are counted on the *simple undirected* graph
(``g.as_undirected(...).simplify()``). Stage-2 wires each relation's edges
near-independently, so two relations rarely land on the same node pair and pairs
are rarely bidirectional — the synthetic simple graph is then inflated (more
distinct edges) relative to the original, which pushes edge-density-sensitive
motifs (paw, 5-cycles) above target and moves the simple-degree sequence off the
manifold Stage-3's degree-preserving swaps can reach. This script quantifies that
gap per graph so we can see whether it is universal or fb237-specific.

Metrics (entity–entity content edges; ``rdf:type`` edges, literal endpoints and
self-loops excluded — the structural comparison, and literal-free so the synthetic
side is comparable):

    n_edges    directed content edges  (a→b, a,b non-literal, a≠b)
    dir_pairs  distinct directed pairs
    und_pairs  distinct undirected pairs   (== the simple-graph edge count)
    parallel   n_edges / dir_pairs    (>1: same ordered pair carries ≥2 relations)
    bidir      dir_pairs / und_pairs   (>1: pair connected in both directions)
    rho        n_edges / und_pairs     (= parallel·bidir; directed→simple collapse)

``rho ≈ 1`` means "essentially a simple graph" (no overlap); the original's ``rho``
is the target. ``inflation = synth und_pairs / orig und_pairs`` is how many extra
simple edges Stage-2 invents.

The Stage-2 graph is built exactly as ``signature_roundtrip`` would with zero
refinement (Stage 1 + Stage 2, no Stage-3 rewiring), from the cached target blocks.

Usage
-----
    python scripts/edge_multiplicity.py                    # all corpus graphs
    python scripts/edge_multiplicity.py fb237_v4 wn18rr_v4
    python scripts/edge_multiplicity.py --seed 7 --out experiments/edge_multiplicity/
"""

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from profile_stage3_deltas import _build_stage2_graph  # noqa: E402
from generator._constants import _RDF_TYPE  # noqa: E402
from kg_io import load_kg  # noqa: E402

_SEARCH_DIRS = [_REPO / "data" / "graphs", _REPO / "data" / "test_graphs"]
_OUT_DIR = _REPO / "experiments" / "edge_multiplicity"


def _find_source_file(graph_dir: Path) -> "Path | None":
    """First non-synthetic .nt/.ttl(.gz) graph file in ``graph_dir``."""
    for pat in ("*.nt", "*.ttl", "*.nt.gz", "*.ttl.gz"):
        hits = sorted(p for p in graph_dir.glob(pat) if not p.stem.endswith("_synth"))
        if hits:
            return hits[0]
    return None


def _corpus_graphs() -> list[str]:
    """All graph dirs holding both a signature/ and a source file (dedup by name)."""
    names: list[str] = []
    for root in _SEARCH_DIRS:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if (d / "signature").is_dir() and _find_source_file(d) and d.name not in names:
                names.append(d.name)
    return names


def _graph_dir(name: str) -> "Path | None":
    for root in _SEARCH_DIRS:
        if (root / name / "signature").is_dir():
            return root / name
    return None


def _overlap_stats(g) -> dict:
    """Entity–entity content-edge pair-overlap stats for an igraph graph.

    Excludes rdf:type edges, self-loops and edges with a literal endpoint (the
    ``is_literal`` vertex attribute; absent ⇒ all False, e.g. synthetic graphs).
    """
    is_lit = g.vs["is_literal"] if "is_literal" in g.vertex_attributes() else [False] * g.vcount()
    n_edges = 0
    dir_pairs: set[tuple[int, int]] = set()
    und_pairs: set[tuple[int, int]] = set()
    for e in g.es:
        if e["predicate"] == _RDF_TYPE:
            continue
        a, b = e.source, e.target
        if a == b or is_lit[a] or is_lit[b]:
            continue
        n_edges += 1
        dir_pairs.add((a, b))
        und_pairs.add((a, b) if a < b else (b, a))
    nd, ndp, nup = n_edges, len(dir_pairs), len(und_pairs)
    return {
        "n_edges": nd,
        "dir_pairs": ndp,
        "und_pairs": nup,
        "parallel": round(nd / ndp, 4) if ndp else float("nan"),
        "bidir": round(ndp / nup, 4) if nup else float("nan"),
        "rho": round(nd / nup, 4) if nup else float("nan"),
    }


def _measure(name: str, seed: int, orig_only: bool = False) -> "dict | None":
    """Measure a graph's original (and, unless ``orig_only``, Stage-2 synthetic)
    pair-overlap stats.

    ``orig_only`` skips the Stage-2 build — a fast survey of how much overlap the
    *targets* demand (the build is the slow part on large corpus graphs, and
    Stage-2's ρ≈1 is a structural property already confirmed on small graphs)."""
    graph_dir = _graph_dir(name)
    if graph_dir is None:
        print(f"  {name}: not found in corpus — skipping")
        return None
    src = _find_source_file(graph_dir)
    print(f"  {name}: loading original {src.name} …", flush=True)
    orig = _overlap_stats(load_kg(src))
    row = {"graph": name}
    for k, v in orig.items():
        row[f"orig_{k}"] = v
    if orig_only:
        for k in orig:
            row[f"synth_{k}"] = float("nan")
        row["inflation"] = float("nan")
        row["rho_gap"] = float("nan")
        return row
    print(f"  {name}: building Stage-2 synthetic (seed {seed}) …", flush=True)
    try:
        synth = _overlap_stats(_build_stage2_graph(name, seed))
    except Exception as exc:  # noqa: BLE001 — report and continue over the corpus
        print(f"  {name}: Stage-2 build failed ({exc}) — skipping")
        return None
    for k, v in synth.items():
        row[f"synth_{k}"] = v
    row["inflation"] = (round(synth["und_pairs"] / orig["und_pairs"], 4)
                        if orig["und_pairs"] else float("nan"))
    row["rho_gap"] = round(orig["rho"] - synth["rho"], 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs", nargs="*", help="graph names (default: all corpus graphs)")
    ap.add_argument("--seed", type=int, default=42, help="master seed for the Stage-2 build")
    ap.add_argument("--out", type=Path, default=_OUT_DIR, help="output directory for the CSV")
    ap.add_argument("--orig-only", action="store_true",
                    help="only measure originals (skip the slow Stage-2 build) — a fast "
                         "survey of how much pair overlap the targets demand")
    args = ap.parse_args()

    graphs = args.graphs or _corpus_graphs()
    mode = "originals only" if args.orig_only else "original vs Stage-2"
    print(f"Measuring edge-multiplicity ({mode}) for {len(graphs)} graph(s): {', '.join(graphs)}\n")

    rows = []
    for name in graphs:
        r = _measure(name, args.seed, orig_only=args.orig_only)
        if r is not None:
            rows.append(r)
    if not rows:
        raise SystemExit("no graphs measured")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / ("edge_multiplicity_orig.csv" if args.orig_only
                           else "edge_multiplicity.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Console summary: the collapse ratio rho (original vs synthetic) is the headline.
    print("\n"
          f"{'graph':>14}  {'orig rho':>9}  {'synth rho':>9}  {'orig par':>9}  "
          f"{'synth par':>9}  {'orig bidir':>10}  {'synth bidir':>11}  {'edge infl':>9}")
    print("-" * 100)
    for r in rows:
        print(f"{r['graph']:>14}  {r['orig_rho']:>9.3f}  {r['synth_rho']:>9.3f}  "
              f"{r['orig_parallel']:>9.3f}  {r['synth_parallel']:>9.3f}  "
              f"{r['orig_bidir']:>10.3f}  {r['synth_bidir']:>11.3f}  {r['inflation']:>9.3f}")
    print(f"\nrho ≈ 1 ⇒ no pair overlap.  edge infl = synthetic simple edges / original.")
    print(f"→ {csv_path}")


if __name__ == "__main__":
    main()
