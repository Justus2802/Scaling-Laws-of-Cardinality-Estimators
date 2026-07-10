"""Shared exact-enumeration helpers for motif counting.

Module-level helpers used by the exact counter implementations: the 4-node
edge-pair classifier and the ESCAPE k-node enumerator.  Graphlet-type constants
live on MotifCounter (the base class).  The colour-coding sampler functions live
in ``cc_motif_counter`` alongside the estimator that drives them.
"""

from collections import defaultdict

import igraph

from ._base import MotifCounter


def _count_motifs4_through_edge(adj: list, u: int, v: int) -> dict[tuple, int]:
    """Count 4-node motif instances containing undirected edge {u, v}.

    Iterates unordered pairs from (N(u)∪N(v))\\{u,v}, classifies each
    4-node subgraph by sorted degree sequence.  ``adj`` is a list of dicts
    (neighbour → count); only key presence matters here.
    Cost: O((deg_u + deg_v)²).
    """
    counts: dict[tuple, int] = {}
    candidates = list((set(adj[u].keys()) | set(adj[v].keys())) - {u, v})
    for i in range(len(candidates)):
        w = candidates[i]
        for j in range(i + 1, len(candidates)):
            x = candidates[j]
            uw = w in adj[u]
            ux = x in adj[u]
            vw = w in adj[v]
            vx = x in adj[v]
            wx = x in adj[w]
            dw = uw + vw + wx
            dx = ux + vx + wx
            if dw == 0 or dx == 0:
                continue
            du = 1 + uw + ux
            dv = 1 + vw + vx
            ds = tuple(sorted((du, dv, dw, dx)))
            if ds in MotifCounter.MOTIF4_DS:
                counts[ds] = counts.get(ds, 0) + 1
    return counts


# Maximum degree for exact 5-node enumeration via ESCAPE.
_ESCAPE_MAX_DEGREE = 50


def count_motifsk_escape(
    g: igraph.Graph, k: int, max_degree: "int | None" = None
) -> dict[tuple, int]:
    """Exact k-node connected graphlet counts via BFS expansion (ESCAPE, WWW 2017).

    Anchors each k-set at its minimum-index node, expands connected partial
    sets by BFS, deduplicates via sorted-tuple key.  Cost: O(m·Δ^(k-2)).
    Raises ``RuntimeError`` if max degree exceeds the degree guard; callers
    should fall back to CC sampling in that case.

    Returns ``{within-subset_sorted_degree_sequence: count}`` over all connected
    induced k-node subgraphs; the cycle entry is keyed by ``C5_DS``/``C6_DS``.

    :param g: undirected simple graph.
    :param k: subgraph size to enumerate (k >= 1).
    :param max_degree: degree-guard override; defaults to ``_ESCAPE_MAX_DEGREE``.
        Raise it to admit graphs with an isolated hub at the cost of a slower
        (Δ^(k-2)) enumeration.
    """
    guard = _ESCAPE_MAX_DEGREE if max_degree is None else max_degree
    n = g.vcount()
    if n < k:
        return {}

    max_deg = max(g.degree()) if n > 0 else 0
    if max_deg > guard:
        raise RuntimeError(
            f"ESCAPE: max degree {max_deg} > {guard}; "
            "use CC sampling instead."
        )

    adj: list[set[int]] = [set() for _ in range(n)]
    for e in g.es:
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)

    counts: defaultdict[tuple, int] = defaultdict(int)

    def _degk(nodes: tuple) -> tuple:
        node_set = set(nodes)
        return tuple(sorted(
            sum(1 for nb in node_set if nb != nd and nb in adj[nd])
            for nd in node_set
        ))

    for u in range(n - (k - 1)):
        seen: set[tuple] = set()
        stack: list[tuple] = [(u,)]
        while stack:
            partial = stack.pop()
            partial_set = set(partial)
            reach: set[int] = set()
            for nd in partial:
                reach |= adj[nd]
            reach -= partial_set
            reach = {v for v in reach if v > u}
            for v in reach:
                new_partial = tuple(sorted(partial_set | {v}))
                if new_partial in seen:
                    continue
                seen.add(new_partial)
                if len(new_partial) == k:
                    counts[_degk(new_partial)] += 1
                else:
                    stack.append(new_partial)

    return dict(counts)


def count_motifs5_escape(g: igraph.Graph) -> dict[tuple, int]:
    """Exact 5-node graphlet counts via ESCAPE (thin wrapper for k=5).

    Retained for back-compat; delegates to ``count_motifsk_escape(g, 5)``.
    """
    return count_motifsk_escape(g, 5)
