"""Shared brute-force motif oracles for the counter tests.

These enumerate subgraphs directly (no sampling, no incremental deltas) so they
serve as independent ground truth for the exact and colour-coding counters and
for the incremental swap deltas in ``generator.local_updates``.

Graphs are small in the tests that use these, so the exponential combination
enumeration is acceptable.
"""

from itertools import combinations

import igraph


def und(n: int, edges: list[tuple[int, int]]) -> igraph.Graph:
    """Build a simple undirected graph on ``n`` nodes with the given edges."""
    g = igraph.Graph(n=n, directed=False)
    g.add_edges(edges)
    return g


def adj(n: int, edges: list[tuple[int, int]]) -> list[dict]:
    """Adjacency-multiplicity dict list matching the format used in stage3."""
    a: list[dict] = [{} for _ in range(n)]
    for u, v in edges:
        a[u][v] = a[u].get(v, 0) + 1
        a[v][u] = a[v].get(u, 0) + 1
    return a


def brute_tri_counts(a: list[dict], n: int) -> tuple[int, dict[int, int]]:
    """Brute-force (total triangles, per-node triangle counts) on adj membership."""
    per_node: dict[int, int] = {v: 0 for v in range(n)}
    total = 0
    for u in range(n):
        nbrs = [w for w in a[u] if w > u]
        for ai in range(len(nbrs)):
            for bi in range(ai + 1, len(nbrs)):
                w, x = nbrs[ai], nbrs[bi]
                if x in a[w]:           # u<w<x triangle
                    total += 1
                    per_node[u] += 1
                    per_node[w] += 1
                    per_node[x] += 1
    return total, per_node


def _subset_connected(a: list[dict], verts: tuple[int, ...], vs: set[int]) -> bool:
    """True iff the induced subgraph on ``vs`` is connected (BFS from verts[0])."""
    start = verts[0]
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in a[cur]:
            if nb in vs and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(vs)


def brute_induced_cycles(a: list[dict], k: int) -> int:
    """Count induced (chordless) k-cycles by brute force over all k-subsets.

    A k-subset induces a k-cycle iff every vertex has exactly 2 neighbours
    inside the subset (2-regular) and the subset is connected (a single cycle,
    not e.g. two disjoint triangles for k=6).
    """
    n = len(a)
    count = 0
    for verts in combinations(range(n), k):
        vs = set(verts)
        degs = {v: sum(1 for u in vs if u != v and u in a[v]) for v in vs}
        if any(d != 2 for d in degs.values()):
            continue
        if _subset_connected(a, verts, vs):
            count += 1
    return count


def brute_motifsk(a: list[dict], k: int) -> dict[tuple, int]:
    """Count connected induced k-node graphlets, keyed by sorted degree sequence.

    Matches the ``{sorted_degree_sequence: count}`` convention of the exact and
    colour-coding counters. For k=2..6 the sorted degree sequence is a complete
    invariant for the graphlets the counters report (edge, wedge/triangle;
    P4/paw/C4/diamond/K4; C5; C6), so this is a faithful oracle for those types.
    """
    n = len(a)
    counts: dict[tuple, int] = {}
    for verts in combinations(range(n), k):
        vs = set(verts)
        if not _subset_connected(a, verts, vs):
            continue
        ds = tuple(sorted(sum(1 for u in vs if u != v and u in a[v]) for v in vs))
        counts[ds] = counts.get(ds, 0) + 1
    return counts


def brute_stars(a: list[dict], max_k: int = 10) -> dict[int, int]:
    """Count induced k-stars for k=2..``max_k`` by brute force.

    A k-star is one centre connected to k leaves with NO edges among the leaves
    (induced-subgraph condition).
    """
    n = len(a)
    totals: dict[int, int] = {k: 0 for k in range(2, max_k + 1)}
    for v in range(n):
        nb = list(a[v])
        d = len(nb)
        for k in range(2, min(d, max_k) + 1):
            for subset in combinations(nb, k):
                # induced star iff no leaf-leaf edge exists in the subset
                if all(
                    b not in a[x]
                    for i, x in enumerate(subset)
                    for b in subset[i + 1:]
                ):
                    totals[k] += 1
    return totals
