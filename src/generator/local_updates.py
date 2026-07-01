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
_induced_paths              — induced (chordless) a→b paths up to a length
_induced_cycles_through_pair — set of induced k-cycles containing a given vertex pair
_cycle_delta                — Δ(induced 5-/6-cycle counts) for one swap
_star_count_delta           — Δ(induced k-star counts) for one swap (exact, O(Δ²))
_tree_entropy_delta         — Δ(depth-2 tree template entropy) and updated freq dict
_path_entropy_delta         — Δ(k-hop path template entropy) for k=2..K and updated freq dicts

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

import math
from collections import Counter, defaultdict


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


def _induced_paths(adj: list, a: int, b: int, max_edges: int) -> list:
    """All induced (chordless) paths from ``a`` to ``b`` of edge-length 1..max_edges.

    Returns a list of vertex tuples ``(a, …, b)``.  "Induced" means the only edges
    among the path's vertices are the consecutive ones — the arcs of an induced
    cycle are exactly such paths.

    Distance-bounded pruning: an interior vertex is dropped once it can no longer
    reach ``b`` within the edges that remain.  The two cheapest, most valuable
    cases use ``b``'s 1-hop set ``tb`` — when one edge remains the vertex must be a
    neighbour of ``b``; when two remain it must be within two hops — which cuts the
    two deepest (and most explosive) DFS levels.
    Cost: O(Δ^(max_edges-1)) worst case, far less in sparse neighbourhoods.
    """
    out: list = []
    tb = set(adj[b])                        # b's 1-hop neighbourhood
    path = [a]
    inp = {a}

    def _dfs() -> None:
        last = path[-1]
        L = len(path)                       # vertices so far; edges so far = L-1
        for x in adj[last]:
            if x in inp:
                continue
            if x == b:
                # Close the arc.  ``b`` may be adjacent to ``a`` (that is just the
                # other arc / cycle-closing edge, not a chord), so the chord check
                # skips the start vertex; a genuine a–b chord is caught later by
                # the degree-2 test in _induced_cycles_through_pair.
                if any(path[i] in adj[b] for i in range(1, L - 1)):
                    continue
                out.append(tuple(path) + (b,))   # closed an induced path of length L
                continue
            ra = max_edges - L              # edges left after adding x as interior
            if ra < 1:                      # no room to still reach b
                continue
            if ra == 1:                     # must finish x→b
                if x not in tb:
                    continue
            elif ra == 2:                   # must reach b within two hops of x
                if x not in tb and tb.isdisjoint(adj[x]):
                    continue
            # Interior vertex: reject a chord to any earlier vertex except `last`.
            if any(path[i] in adj[x] for i in range(L - 1)):
                continue
            path.append(x)
            inp.add(x)
            _dfs()
            path.pop()
            inp.discard(x)

    if a != b:
        _dfs()
    return out


def _induced_cycles_through_pair(adj: list, a: int, b: int, k: int) -> set[frozenset]:
    """Induced (chordless) k-cycles whose vertex set contains both ``a`` and ``b``.

    Every such cycle splits at {a,b} into two internally-disjoint induced paths
    (arcs) of edge-lengths l1 + l2 = k.  Enumerate induced a→b paths and pair
    complementary lengths; a candidate vertex set is an induced k-cycle iff every
    vertex has exactly two neighbours inside it (which rules out any chord,
    including the a–b chord when both arcs are length ≥2).
    Cost: O(Δ^(k-2)).
    """
    by_len: dict[int, list] = defaultdict(list)
    for p in _induced_paths(adj, a, b, k - 1):
        by_len[len(p) - 1].append(set(p))

    found: set[frozenset] = set()
    for l1 in range(1, k // 2 + 1):
        for s1set in by_len.get(l1, ()):
            for s2set in by_len.get(k - l1, ()):
                v = s1set | s2set
                if len(v) != k:              # interiors must be disjoint (share only a,b)
                    continue
                ok = True
                for u in v:
                    au = adj[u]
                    c = 0
                    for w in v:
                        if w != u and w in au:
                            c += 1
                            if c > 2:
                                break
                    if c != 2:
                        ok = False
                        break
                if ok:
                    found.add(frozenset(v))
    return found


def _inner_edge_count(adj: list, v: int) -> int:
    """Count edges among the neighbors of v (inner edges for induced-star computation)."""
    nbrs = set(adj[v])
    count = 0
    for u in nbrs:
        for w in adj[u]:
            if w in nbrs and w > u:
                count += 1
    return count


def _star_contributions(
    adj: list, v: int, max_k: int, max_center_degree: "float | None" = None
) -> list[int]:
    """Induced k-star counts contributed by node v as center, for k=2..max_k.

    Uses inclusion-exclusion over inner edges (edges between neighbors of v).
    Triangle-free centers contribute C(d, k) exactly; others are corrected.
    Returns list of length max_k+1 indexed by k (indices 0,1 unused).

    The inclusion-exclusion is O(2^(inner edges)), which explodes on clustered
    hubs.  When ``max_center_degree`` is set, a center whose simple degree exceeds
    it contributes zeros (is skipped).  Degree is invariant under degree-preserving
    swaps, so a node's skip status is identical before and after any swap — the
    same centers are excluded from the baseline and from every incremental delta,
    keeping the tracked total self-consistent.
    """
    nbrs_v = set(adj[v])
    d = len(nbrs_v)
    totals = [0] * (max_k + 1)
    if d < 2 or (max_center_degree is not None and d > max_center_degree):
        return totals

    inner_edges = [
        (u, w)
        for u in nbrs_v
        for w in adj[u]
        if w in nbrs_v and w > u
    ]

    if not inner_edges:
        for k in range(2, min(d, max_k) + 1):
            totals[k] = math.comb(d, k)
        return totals

    # Inclusion-exclusion: subtract subsets containing ≥1 inner edge
    ie = len(inner_edges)
    correction = [0] * (max_k + 1)
    for mask in range(1, 1 << ie):
        verts: set[int] = set()
        bits = mask
        sign_exp = 0
        while bits:
            idx = (bits & -bits).bit_length() - 1
            u, w = inner_edges[idx]
            verts.add(u)
            verts.add(w)
            sign_exp += 1
            bits &= bits - 1
        s = len(verts)
        sign = (-1) ** (sign_exp + 1)
        for k in range(s, min(d, max_k) + 1):
            correction[k] += sign * math.comb(d - s, k - s)

    for k in range(2, min(d, max_k) + 1):
        totals[k] = math.comb(d, k) - correction[k]
    return totals


def _star_count_delta(
    adj: list,
    s1: int, o1: int, s2: int, o2: int,
    max_k: int = 10,
    max_center_degree: "float | None" = None,
) -> dict[int, int]:
    """Compute Δ(induced k-star counts) for k=2..max_k for swapping (s1,o1)↔(s2,o2).

    Only nodes whose neighborhood structure changes can change their star
    contributions: the four endpoint nodes s1, o1, s2, o2, and any node that
    has both of a changed pair among its neighbors (because its inner-edge set
    changes).  This is O(Δ²) total, matching the exactness of the counter.

    ``max_center_degree`` is forwarded to ``_star_contributions``: centers above
    it are skipped (the O(2^(inner edges)) inclusion-exclusion is intractable on
    clustered hubs).  Degree is swap-invariant, so a skipped center is excluded
    from both the before and after sums and contributes 0 to the delta — matching
    the same threshold applied to the baseline count.

    Returns a dict {k: delta} with only nonzero entries.
    """
    # Nodes whose neighbor sets change: s1 loses o1/gains o2, s2 loses o2/gains o1.
    # Their star contributions change because their degree changes — but degree is
    # preserved (they keep the same total degree), so only inner-edge changes matter.
    # Nodes that *contain* a changed pair {s1,o1},{s2,o2},{s1,o2},{s2,o1} as neighbors
    # also change their inner-edge set.
    affected: set[int] = {s1, o1, s2, o2}
    for u in list(adj[s1]):
        if u in adj[o1] or u in adj[o2]:
            affected.add(u)
    for u in list(adj[s2]):
        if u in adj[o1] or u in adj[o2]:
            affected.add(u)

    before: dict[int, int] = {}
    for v in affected:
        for k, cnt in enumerate(_star_contributions(adj, v, max_k, max_center_degree)):
            if cnt:
                before[k] = before.get(k, 0) + cnt

    _adj_dec(adj, s1, o1)
    _adj_dec(adj, s2, o2)
    _adj_inc(adj, s1, o2)
    _adj_inc(adj, s2, o1)

    after: dict[int, int] = {}
    for v in affected:
        for k, cnt in enumerate(_star_contributions(adj, v, max_k, max_center_degree)):
            if cnt:
                after[k] = after.get(k, 0) + cnt

    _adj_dec(adj, s1, o2)
    _adj_dec(adj, s2, o1)
    _adj_inc(adj, s1, o1)
    _adj_inc(adj, s2, o2)

    return {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)
            if after.get(k, 0) != before.get(k, 0)}


def _tree_entropy_delta(
    rel_out: list[dict],
    pair_freq: dict,
    s1: int, o1: int, p: str,
    s2: int, o2: int,
) -> tuple[float, dict]:
    """Compute Δ(depth-2 tree template entropy) for swapping (s1 →p o1, s2 →p o2).

    A depth-2 tree template rooted at node v is the multiset of (p, r') pairs
    where p is the relation on the root→child edge and r' is any relation on a
    child→grandchild edge.  Swapping (s1 →p o1) to (s1 →p o2) replaces all
    (p, r') pairs contributed by o1's outgoing relations with those from o2.
    Symmetrically for the (s2 →p o2) → (s2 →p o1) side.

    Returns (new_entropy, new_pair_freq) — the caller applies them only on accept.
    Cost: O(out_degree(o1) + out_degree(o2)) per swap.
    """
    new_freq = dict(pair_freq)

    def _swap_child(old_obj: int, new_obj: int) -> None:
        for r2 in rel_out[old_obj]:
            key = (p, r2)
            c = new_freq.get(key, 0) - 1
            if c <= 0:
                new_freq.pop(key, None)
            else:
                new_freq[key] = c
        for r2 in rel_out[new_obj]:
            key = (p, r2)
            new_freq[key] = new_freq.get(key, 0) + 1

    _swap_child(o1, o2)  # s1: old child o1 → new child o2
    _swap_child(o2, o1)  # s2: old child o2 → new child o1

    total = sum(new_freq.values())
    if total == 0:
        return 0.0, new_freq
    inv = 1.0 / total
    h = 0.0
    for c in new_freq.values():
        if c > 0:
            p_i = c * inv
            h -= p_i * math.log(p_i)
    return h, new_freq


def _entropy_from_freq(freq: dict) -> float:
    """Shannon entropy of a frequency dict {key: count}."""
    total = sum(freq.values())
    if total == 0:
        return 0.0
    inv = 1.0 / total
    h = 0.0
    for c in freq.values():
        if c > 0:
            p_i = c * inv
            h -= p_i * math.log(p_i)
    return h


def _path_entropy_delta(
    out_edges: list[list],
    path_freqs: dict[int, dict],
    s1: int, o1: int, p: str,
    s2: int, o2: int,
) -> tuple[dict[int, float], dict[int, dict]]:
    """Compute Δ(path template entropy) for k=2 and k=3 for swapping (s1 →p o1, s2 →p o2).

    ``out_edges[v]`` is a list of ``(relation, target)`` pairs for directed
    outgoing edges from node v.  Tracks only walks where the swapped edge is
    the **first hop** (root→child), so templates have the form (p, r2[, r3]).

    - k=2: template (p, r2) for each r2 ∈ out_edges[child].
      Cost: O(deg(o1) + deg(o2)).
    - k=3: template (p, r2, r3) for each (r2, mid) ∈ out_edges[child],
      r3 ∈ out_edges[mid].  Cost: O(Δ²) worst case.

    Returns (new_entropies, new_path_freqs) keyed by k.
    The caller applies them only on accept.
    """
    new_freqs: dict[int, dict] = {k: dict(path_freqs[k]) for k in path_freqs}

    def _dec(freq: dict, key) -> None:
        c = freq.get(key, 0) - 1
        if c <= 0:
            freq.pop(key, None)
        else:
            freq[key] = c

    def _update_child(old_obj: int, new_obj: int) -> None:
        # k=2: (p, r2)
        if 2 in new_freqs:
            f2 = new_freqs[2]
            for r2, _ in out_edges[old_obj]:
                _dec(f2, (p, r2))
            for r2, _ in out_edges[new_obj]:
                key = (p, r2)
                f2[key] = f2.get(key, 0) + 1

        # k=3: (p, r2, r3)
        if 3 in new_freqs:
            f3 = new_freqs[3]
            for r2, mid in out_edges[old_obj]:
                for r3, _ in out_edges[mid]:
                    _dec(f3, (p, r2, r3))
            for r2, mid in out_edges[new_obj]:
                for r3, _ in out_edges[mid]:
                    key = (p, r2, r3)
                    f3[key] = f3.get(key, 0) + 1

    _update_child(o1, o2)  # s1: was going to o1, now goes to o2
    _update_child(o2, o1)  # s2: was going to o2, now goes to o1

    new_entropies = {k: _entropy_from_freq(new_freqs[k]) for k in new_freqs}
    return new_entropies, new_freqs


def _cycle_delta(
    adj: list, s1: int, o1: int, s2: int, o2: int, *, k5: bool = True, k6: bool = True
) -> tuple[int, int]:
    """Compute Δ(induced 5-cycle, 6-cycle) counts for swapping (s1,o1)↔(s2,o2).

    Returns (Δc5, Δc6); a disabled size contributes 0.

    Only induced cycles whose vertex set contains a *changed* node pair can flip
    status (every other induced subgraph is identical before and after).  The four
    changed pairs are exactly the toggled edges {s1,o1},{s2,o2},{s1,o2},{s2,o1}, so
    counting induced cycles through those pairs before and after the swap and
    diffing is exact: a cycle that is induced in *both* graphs cannot contain a
    changed pair (its induced subgraph would differ), so it cancels.
    Cost: O(Δ^(k-2)) per pair, computed before and after the swap.
    """
    pairs = ((s1, o1), (s2, o2), (s1, o2), (s2, o1))
    ks = ([5] if k5 else []) + ([6] if k6 else [])

    def _count(k: int) -> int:
        seen: set[frozenset] = set()
        for a, b in pairs:
            seen |= _induced_cycles_through_pair(adj, a, b, k)
        return len(seen)

    before = {k: _count(k) for k in ks}

    _adj_dec(adj, s1, o1)
    _adj_dec(adj, s2, o2)
    _adj_inc(adj, s1, o2)
    _adj_inc(adj, s2, o1)

    after = {k: _count(k) for k in ks}

    _adj_dec(adj, s1, o2)
    _adj_dec(adj, s2, o1)
    _adj_inc(adj, s1, o1)
    _adj_inc(adj, s2, o2)

    return after.get(5, 0) - before.get(5, 0), after.get(6, 0) - before.get(6, 0)
