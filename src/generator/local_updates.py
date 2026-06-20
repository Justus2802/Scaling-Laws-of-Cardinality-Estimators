"""Incremental motif-count update helpers for the SA rewiring loop.

All functions operate on the live adjacency dict ``adj`` (list of dicts)
maintained inside ``refine()``, not on an ``igraph.Graph``.  They compute
exact delta values per double-edge swap so the SA loop avoids full remeasures.

Public API
----------
_adj_inc                    — increment edge count in adj dict
_adj_dec                    — decrement edge count in adj dict
_triangle_node_delta        — Δ(triangles) and per-node Δt_v for one swap
_count_motifs4_through_edge — raw 4-node motif counts touching a given edge
_motif4_delta               — Δ(4-node motif counts) for one swap
"""

# The four connected 4-node graphlet degree sequences (C4, diamond, K4, paw).
# Mirrors MotifCounter.MOTIF4_DS; defined locally to keep this module dependency-free.
_MOTIF4_DS: frozenset[tuple] = frozenset({(2, 2, 2, 2), (2, 2, 3, 3), (3, 3, 3, 3), (1, 2, 2, 3)})


def _adj_inc(adj: list, u: int, v: int) -> None:
    adj[u][v] = adj[u].get(v, 0) + 1
    adj[v][u] = adj[v].get(u, 0) + 1


def _adj_dec(adj: list, u: int, v: int) -> None:
    adj[u][v] -= 1
    if adj[u][v] == 0:
        del adj[u][v]
    adj[v][u] -= 1
    if adj[v][u] == 0:
        del adj[v][u]


def _triangle_node_delta(
    adj: list, s1: int, o1: int, s2: int, o2: int
) -> tuple[int, dict[int, int]]:
    """Compute per-node and aggregate triangle change from swapping o1↔o2.

    Returns (ΔT, node_deltas) where ΔT = gained − lost triangles and
    node_deltas maps node → change in its per-node triangle count.
    Cost: O((deg_s1 + deg_o1 + deg_s2 + deg_o2) · Δ).
    """
    nd: dict[int, int] = {}

    def _sub(u: int, v: int) -> None:
        for w in set(adj[u]) & set(adj[v]):
            nd[u] = nd.get(u, 0) - 1
            nd[v] = nd.get(v, 0) - 1
            nd[w] = nd.get(w, 0) - 1

    def _add(u: int, v: int) -> None:
        for w in set(adj[u]) & set(adj[v]):
            nd[u] = nd.get(u, 0) + 1
            nd[v] = nd.get(v, 0) + 1
            nd[w] = nd.get(w, 0) + 1

    _sub(s1, o1)
    _sub(s2, o2)
    _adj_dec(adj, s1, o1)
    _adj_dec(adj, s2, o2)
    _add(s1, o2)
    _add(s2, o1)
    _adj_inc(adj, s1, o1)
    _adj_inc(adj, s2, o2)

    delta_T = sum(nd.values()) // 3
    return delta_T, nd


def _count_motifs4_through_edge(adj: list, u: int, v: int) -> dict[tuple, int]:
    """Count 4-node motif instances containing undirected edge {u, v}.

    Iterates unordered pairs from (N(u)∪N(v))\\{u,v}, classifies each
    4-node subgraph by sorted degree sequence.
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
            if ds in _MOTIF4_DS:
                counts[ds] = counts.get(ds, 0) + 1
    return counts


def _motif4_delta(
    adj: list, s1: int, o1: int, s2: int, o2: int
) -> dict[tuple, int]:
    """Compute change in 4-node motif counts from swapping (s1,o1)↔(s2,o2).

    Uses inclusion-exclusion to avoid double-counting 4-node subgraphs
    that span both swapped edges.
    Cost: O((deg_s1 + deg_o1 + deg_s2 + deg_o2)²).
    """
    def _overlap(a: int, b: int, c: int, d: int) -> dict[tuple, int]:
        if len({a, b, c, d}) < 4:
            return {}
        nodes = [a, b, c, d]
        degs = [sum(1 for nd2 in nodes if nd2 != nd and nd2 in adj[nd]) for nd in nodes]
        if min(degs) == 0:
            return {}
        ds = tuple(sorted(degs))
        return {ds: 1} if ds in _MOTIF4_DS else {}

    def _count_pair(ea: tuple, eb: tuple) -> dict[tuple, int]:
        cu = _count_motifs4_through_edge(adj, *ea)
        cv = _count_motifs4_through_edge(adj, *eb)
        ov = _overlap(ea[0], ea[1], eb[0], eb[1])
        result: dict[tuple, int] = {}
        for k in set(cu) | set(cv) | set(ov):
            result[k] = cu.get(k, 0) + cv.get(k, 0) - ov.get(k, 0)
        return result

    before = _count_pair((s1, o1), (s2, o2))

    _adj_dec(adj, s1, o1)
    _adj_dec(adj, s2, o2)
    _adj_inc(adj, s1, o2)
    _adj_inc(adj, s2, o1)

    after = _count_pair((s1, o2), (s2, o1))

    _adj_dec(adj, s1, o2)
    _adj_dec(adj, s2, o1)
    _adj_inc(adj, s1, o1)
    _adj_inc(adj, s2, o2)

    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in set(before) | set(after)
        if after.get(k, 0) != before.get(k, 0)
    }
