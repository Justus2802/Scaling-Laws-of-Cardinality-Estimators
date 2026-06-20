"""Incremental motif-count update helpers for the SA rewiring loop.

All functions operate on the live adjacency dict ``adj`` (list of dicts)
maintained inside ``refine()``, not on an ``igraph.Graph``.  They compute
exact delta values per double-edge swap so the SA loop avoids full remeasures.

Public API
----------
_adj_inc                    — increment edge count in adj dict
_adj_dec                    — decrement edge count in adj dict
_triangle_node_delta        — Δ(triangles) and per-node Δt_v for one swap
_classify4                  — motif degree-seq of an induced 4-node subgraph
_motifs4_through_pairs      — {4-set: degree-seq} for motifs containing swap pairs
_motif4_delta               — Δ(4-node motif counts) for one swap
_induced_cycles_through_nodes — set of induced k-cycles touching given anchors
_cycle_delta                — Δ(induced 5-/6-cycle counts) for one swap

The per-edge motif primitive ``_count_motifs4_through_edge`` lives in
``motif_counter._common`` (shared with ``ExactMotifCounter``).

Cycle semantics
---------------
The motif counters classify a k-node subset by its *induced* degree sequence
(neighbours counted within the subset), so ``five_cycle_count`` /
``six_cycle_count`` count only **chordless** (induced) cycles — degree
sequences (2,2,2,2,2) and (2,2,2,2,2,2).  The cycle delta below therefore
tracks induced cycles, not arbitrary closed walks: a swap changes the count
both by adding/removing a cycle edge and by adding a chord (destroys an
induced cycle) or removing a chord (can create one).
"""

from collections import Counter


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


_PAW_DS: tuple = (1, 2, 2, 3)
# All four connected 4-node motifs (C4, paw, diamond, K4).
_MOTIF4_ALL: frozenset[tuple] = frozenset({(2, 2, 2, 2), _PAW_DS, (2, 2, 3, 3), (3, 3, 3, 3)})


def _classify4(adj: list, a: int, b: int, w: int, x: int) -> "tuple | None":
    """Return the motif degree sequence of the induced subgraph on {a,b,w,x}.

    Returns one of (2,2,2,2)/(1,2,2,3)/(2,2,3,3)/(3,3,3,3), or ``None`` when the
    four nodes do not induce a connected 4-node motif.  Treats ``adj`` keys as a
    simple graph.
    """
    e_ab = b in adj[a]; e_aw = w in adj[a]; e_ax = x in adj[a]
    e_bw = w in adj[b]; e_bx = x in adj[b]; e_wx = x in adj[w]
    m = e_ab + e_aw + e_ax + e_bw + e_bx + e_wx
    if m == 6:
        return (3, 3, 3, 3)
    if m == 5:
        return (2, 2, 3, 3)
    if m == 4:
        da = e_ab + e_aw + e_ax; db = e_ab + e_bw + e_bx
        dw = e_aw + e_bw + e_wx; dx = e_ax + e_bx + e_wx
        return (2, 2, 2, 2) if min(da, db, dw, dx) == 2 else _PAW_DS
    return None  # m <= 3: tree/disconnected, not a tracked motif


def _motifs4_through_pairs(adj: list, pairs, types: frozenset) -> dict[frozenset, tuple]:
    """Map each motif 4-set (of a requested type) *containing a given pair* to its
    degree sequence — keyed by ``frozenset`` of its four vertices.

    For each pair {a,b}, the other two vertices are drawn from ``N(a)∪N(b)``.  This
    is exact for C4/diamond/K4: every vertex of those motifs lies within one hop of
    any vertex pair.  The paw is the sole exception — when {a,b} are the two
    triangle vertices whose common apex carries the pendant, that pendant sits two
    hops away and is missed.  Those paws are recovered by a separate scan (apex
    ``p ∈ N(a)∩N(b)``, pendant ``s ∈ N(p)`` with ``s ∉ N(a)∪N(b)``) when the paw is
    requested, keeping the whole routine O(Δ²) per pair.
    """
    result: dict[frozenset, tuple] = {}
    want_paw = _PAW_DS in types
    for a, b in pairs:
        na, nb = adj[a], adj[b]
        cand = (set(na) | set(nb))
        cand.discard(a); cand.discard(b)
        cand = list(cand)
        for i in range(len(cand)):
            w = cand[i]
            for j in range(i + 1, len(cand)):
                x = cand[j]
                key = frozenset((a, b, w, x))
                if key in result:
                    continue
                ds = _classify4(adj, a, b, w, x)
                if ds is not None and ds in types:
                    result[key] = ds
        # Recover paws whose pendant hangs off the apex of triangle {a,b,p}
        # (pendant two hops from the {a,b} pair, so missed by the scan above).
        if want_paw and b in na:
            for p in set(na) & set(nb):
                for s in adj[p]:
                    if s == a or s == b or s in na or s in nb:
                        continue
                    key = frozenset((a, b, p, s))
                    if key not in result:
                        result[key] = _PAW_DS
    return result


def _motif4_delta(
    adj: list, s1: int, o1: int, s2: int, o2: int, *, types: frozenset = _MOTIF4_ALL
) -> dict[tuple, int]:
    """Compute change in 4-node motif counts from swapping (s1,o1)↔(s2,o2).

    Only counts for the motif types in ``types`` are returned.  Only 4-sets
    containing a *changed* node pair can change type, so it suffices to enumerate
    motif 4-sets containing one of the four swap pairs before and after the swap
    and diff the per-type counts.  This correctly handles swaps whose edge acts as
    a *diagonal* of a motif (e.g. a C4 turning into a diamond when a chord is added)
    and the paw's two-hop pendant (see ``_motifs4_through_pairs``).
    Cost: O(Δ²), computed before and after the swap.
    """
    pairs = ((s1, o1), (s2, o2), (s1, o2), (s2, o1))
    before = Counter(_motifs4_through_pairs(adj, pairs, types).values())
    _adj_dec(adj, s1, o1); _adj_dec(adj, s2, o2)
    _adj_inc(adj, s1, o2); _adj_inc(adj, s2, o1)
    after = Counter(_motifs4_through_pairs(adj, pairs, types).values())
    _adj_dec(adj, s1, o2); _adj_dec(adj, s2, o1)
    _adj_inc(adj, s1, o1); _adj_inc(adj, s2, o2)

    return {
        ds: after[ds] - before[ds]
        for ds in set(before) | set(after)
        if ds in types and after[ds] != before[ds]
    }


def _induced_cycles_through_nodes(
    adj: list, anchors, k: int
) -> set[frozenset]:
    """Enumerate induced (chordless) k-cycles whose vertex set contains an anchor.

    Treats ``adj`` as a simple graph (key presence = edge, multiplicity ignored,
    matching the simple-graph projection the motif counters use).  Each cycle is
    returned once as a ``frozenset`` of its k vertices.

    A vertex set is an induced k-cycle iff its induced subgraph is exactly a
    cycle — i.e. the only edges among the k vertices are the k consecutive ones,
    no chords.  Anchored DFS builds chordless paths ``a = p0, p1, …`` and closes
    them back to ``a``; the chord-free condition is enforced incrementally so
    only genuine induced cycles are emitted.
    Cost: O(Δ^(k-1)) per anchor, Δ = max degree of explored nodes.
    """
    found: set[frozenset] = set()

    def _dfs(path: list, inpath: set, a: int) -> None:
        pos = len(path) - 1          # index of the current last vertex
        last = path[pos]
        target = pos + 1             # index we are about to fill
        for x in adj[last]:
            if x in inpath:
                continue
            if target < k - 1:
                # Interior vertex: no chord to any earlier vertex (incl. anchor a),
                # which also rules out premature short cycles back to a.
                if any(path[i] in adj[x] for i in range(pos)):
                    continue
                path.append(x)
                inpath.add(x)
                _dfs(path, inpath, a)
                inpath.discard(x)
                path.pop()
            else:
                # Closing vertex: must link back to a, with no chord to p1..p_{pos-1}.
                if a not in adj[x]:
                    continue
                if any(path[i] in adj[x] for i in range(1, pos)):
                    continue
                found.add(frozenset(path + [x]))

    if k >= 3:
        for a in set(anchors):
            _dfs([a], {a}, a)
    return found


def _cycle_delta(
    adj: list, s1: int, o1: int, s2: int, o2: int, *, k5: bool = True, k6: bool = True
) -> tuple[int, int]:
    """Compute Δ(induced 5-cycle, 6-cycle) counts for swapping (s1,o1)↔(s2,o2).

    Returns (Δc5, Δc6); a disabled size contributes 0.

    Only induced cycles whose vertex set contains a *changed* node pair can flip
    status (every other induced subgraph is identical before and after).  All
    such cycles touch one of the four swap endpoints, so it suffices to count
    induced cycles through ``{s1,o1,s2,o2}`` before and after the swap and diff
    them.  A cycle that is induced in *both* graphs would have to contain an
    unchanged induced subgraph and so cannot contain a changed pair, hence
    cancels — making the simple set-size difference exact.
    Cost: O(Δ^(k-1)) per endpoint, computed before and after the swap.
    """
    anchors = {s1, o1, s2, o2}
    ks = ([5] if k5 else []) + ([6] if k6 else [])

    before = {k: len(_induced_cycles_through_nodes(adj, anchors, k)) for k in ks}

    _adj_dec(adj, s1, o1)
    _adj_dec(adj, s2, o2)
    _adj_inc(adj, s1, o2)
    _adj_inc(adj, s2, o1)

    after = {k: len(_induced_cycles_through_nodes(adj, anchors, k)) for k in ks}

    _adj_dec(adj, s1, o2)
    _adj_dec(adj, s2, o1)
    _adj_inc(adj, s1, o1)
    _adj_inc(adj, s2, o2)

    return after.get(5, 0) - before.get(5, 0), after.get(6, 0) - before.get(6, 0)
