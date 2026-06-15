"""Standalone igraph ground-truth oracle for the Block E cross-check test.

Computes the *library* motif/star ground truth (igraph ``motifs_randesu`` plus a
degree-based star count) for one KG file, on Block E's undirected simplification.

It lives in its own module with a ``__main__`` entry point so the test can run it
in a **child process** under a wall-clock timeout: ``motifs_randesu`` is a
GIL-holding igraph C call, so a same-process thread could neither interrupt it
nor free the CPU. Running it as a subprocess lets the test kill a runaway
enumeration cleanly and report a normal test failure instead of hanging.

Run as::

    python _block_e_library_oracle.py <graph-path> <format>

and it prints the ground truth as JSON on stdout.
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import igraph
from kg_io import load_kg


def load_graph(path: str, fmt: str) -> igraph.Graph:
    """Load a KG file via load_kg.

    The raw dataset files often have no extension (e.g. ``59622641``); load_kg
    detects the serialization from content, so the path is passed straight
    through. ``fmt`` is retained for the CLI signature but no longer needed.
    """
    return load_kg(path)


def isoclass_index(size: int, degseq: tuple[int, ...]) -> int:
    """Return the igraph isomorphism-class index of the undirected graph on
    ``size`` vertices with the given sorted degree sequence.

    Resolving the index by structure (rather than hard-coding a magic number)
    keeps the mapping self-documenting and robust to igraph reordering classes
    between versions. Each degree sequence we look up is unique among the
    connected isoclasses of its size, so the match is unambiguous.
    """
    target = tuple(sorted(degseq))
    idx = 0
    while True:
        try:
            g = igraph.Graph.Isoclass(size, idx, directed=False)
        except Exception as exc:  # walked past the last class for this size
            raise AssertionError(
                f"no size-{size} isoclass with degree sequence {target}"
            ) from exc
        if tuple(sorted(g.degree())) == target:
            return idx
        idx += 1


def compute_ground_truth(path: str, fmt: str) -> dict:
    """Exact library motif/star counts on Block E's undirected simplification.

    ``lib_seconds`` measures only the igraph motif-counting calls (the work
    Block E's custom counters replace), excluding graph loading.
    """
    g_und = load_graph(path, fmt).as_undirected(combine_edges="first").simplify()
    _t0 = time.perf_counter()
    m3 = g_und.motifs_randesu(size=3)
    m4 = g_und.motifs_randesu(size=4)
    lib_seconds = time.perf_counter() - _t0
    degrees = g_und.degree()
    return {
        "n": g_und.vcount(),
        "lib_seconds": lib_seconds,
        "triangle": int(m3[isoclass_index(3, (2, 2, 2))]),          # K3
        "four_cycle": int(m4[isoclass_index(4, (2, 2, 2, 2))]),     # C4
        "diamond": int(m4[isoclass_index(4, (2, 2, 3, 3))]),        # K4 minus an edge
        "k4": int(m4[isoclass_index(4, (3, 3, 3, 3))]),             # complete K4
        "tailed": int(m4[isoclass_index(4, (1, 2, 2, 3))]),         # paw
        # Stars are non-induced (motifs_randesu counts induced subgraphs), so use
        # the degree-based definition: Σ_v C(deg(v), k).
        "stars": {str(k): sum(math.comb(d, k) for d in degrees if d >= k) for k in range(2, 11)},
    }


if __name__ == "__main__":
    json.dump(compute_ground_truth(sys.argv[1], sys.argv[2]), sys.stdout)
