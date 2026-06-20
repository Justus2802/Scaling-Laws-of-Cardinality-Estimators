"""Pluggable motif counting strategies.

``MotifCounter`` is the shared counting interface used by both Stage 3 rewiring
(``generator.stage3``) and the Block E signature measurement (``signature.block_e``).
Implementations can be swapped in via module-level constants in each consumer.

Public API
----------
cc_run              — colour-coding graphlet estimator (Bressan et al. 2021)
cc_run_stars        — colour-coding star-treelet estimator
count_stars_exact   — exact induced k-star counts for k=2..10
MotifCounter        — abstract base class
CCMotifCounter      — colour-coding implementation (sampling-based)
ExactMotifCounter   — exact enumeration for k ≤ 4
ESCAPEFiveNodeCounter — exact 5-node graphlet counting via algebraic identities
                        (Pinar, Seshadhri, Vishal — WWW 2017)

Private helpers (used by stage3 incremental delta)
--------------------------------------------------
MOTIF4_DS
_count_motifs4_through_edge
_motif4_delta
"""

import math
from abc import ABC, abstractmethod
from collections import defaultdict

import igraph
import numpy as np
import scipy.sparse

# 4-node connected motif types tracked during rewiring (sorted degree sequences).
MOTIF4_DS: frozenset = frozenset({(2, 2, 2, 2), (2, 2, 3, 3), (3, 3, 3, 3), (1, 2, 2, 3)})

# σ_H: number of directed spanning P_k paths for each graphlet type.
# Used by cc_run to convert raw sample proportions to estimated counts.
_SIGMA: dict[tuple[int, ...], int] = {
    # k=3
    (2, 2, 2): 6,       # triangle (C3): 3 spanning P3 paths × 2 directions
    # k=4
    (1, 1, 2, 2): 2,   # P4 path
    (2, 2, 2, 2): 8,   # C4
    (1, 2, 2, 3): 4,   # tailed triangle
    (2, 2, 3, 3): 8,   # diamond
    (3, 3, 3, 3): 24,  # K4
    # k=5
    (2, 2, 2, 2, 2): 10,  # C5
    # k=6
    (2, 2, 2, 2, 2, 2): 12,  # C6
}


def cc_run(
    g_und: igraph.Graph,
    k: int,
    n_samples: int,
    rng: np.random.Generator,
    *,
    _A: "scipy.sparse.csr_matrix | None" = None,
    _adj: "list[np.ndarray] | None" = None,
) -> dict[tuple[int, ...], int]:
    """Colour-coding estimator for k-node graphlet counts (Bressan et al. 2021).

    Randomly assigns k colors, builds a directed path-treelet DP via sparse
    matrix products, samples n_samples colorful k-paths by backtracking, and
    returns {degree_sequence_tuple: estimated_count}.

    Pass pre-built _A (csr_matrix) and _adj (neighbour lists) to avoid
    rebuilding them for every k — they are identical across all CC calls.

    The σ_H correction (number of directed P_k paths spanning motif H) and
    the p_k = k!/k^k colorfulness probability are applied so the returned
    counts estimate the true graphlet frequencies.
    """
    p_k = math.factorial(k) / (k ** k)

    n = g_und.vcount()
    if n < k:
        return {}

    colors = rng.integers(0, k, size=n, dtype=np.int32)

    n_sets   = 1 << k
    full_set = n_sets - 1

    A = _A if _A is not None else scipy.sparse.csr_matrix(
        g_und.get_adjacency_sparse()
    ).astype(np.float32)

    dp = np.zeros((n, n_sets), dtype=np.float32)
    dp[np.arange(n), 1 << colors] = 1.0
    dp_levels = [dp]

    for step in range(1, k):
        dp_next = np.zeros((n, n_sets), dtype=np.float32)
        for c in range(k):
            mc = 1 << c
            S_src = np.array(
                [S for S in range(n_sets)
                 if not (S & mc) and bin(S).count('1') == step],
                dtype=np.int32,
            )
            if len(S_src) == 0:
                continue
            S_dst = S_src | mc
            node_mask = (colors == c).astype(np.float32)[:, None]
            dp_next[:, S_dst] += (A @ dp_levels[-1][:, S_src]) * node_mask
        dp_levels.append(dp_next)

    t = float(dp_levels[-1][:, full_set].sum())
    if t == 0:
        return {}

    adj = _adj if _adj is not None else [
        np.array(g_und.neighbors(v), dtype=np.int32) for v in range(n)
    ]

    wfinal = dp_levels[-1][:, full_set].astype(np.float64)
    wfinal /= wfinal.sum()

    v_starts = rng.choice(n, size=n_samples, p=wfinal)

    paths_nodes: list[list[int]] = [[int(v)] for v in v_starts]
    S_arr   = [full_set] * n_samples
    valid   = [True]     * n_samples

    for bk_level in range(k - 1, 0, -1):
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in range(n_samples):
            if not valid[i]:
                continue
            v  = paths_nodes[i][-1]
            sp = S_arr[i] ^ (1 << int(colors[v]))
            groups[(v, sp)].append(i)

        for (v, sp), idxs in groups.items():
            nbrs = adj[v]
            dv   = len(nbrs)
            if dv == 0:
                for i in idxs: valid[i] = False
                continue
            nw  = dp_levels[bk_level - 1][nbrs, sp].astype(np.float64)
            tot = nw.sum()
            if tot == 0:
                for i in idxs: valid[i] = False
                continue
            chosen = nbrs[rng.choice(dv, size=len(idxs), p=nw / tot)]
            for j, i in enumerate(idxs):
                paths_nodes[i].append(int(chosen[j]))
                S_arr[i] = sp

    _HUB_ADJ_THRESH = 200
    hub_adj_sets: dict[int, set[int]] = {
        v: set(adj[v].tolist())
        for v in range(n)
        if len(adj[v]) > _HUB_ADJ_THRESH
    }

    raw_counts: defaultdict[tuple[int, ...], int] = defaultdict(int)
    n_valid = 0

    for i in range(n_samples):
        if not valid[i] or len(paths_nodes[i]) != k:
            continue
        node_list = paths_nodes[i]
        node_set  = set(node_list)
        if len(node_set) != k:
            continue
        n_valid += 1

        local_adj: dict[int, set[int]] = {
            v: hub_adj_sets[v] if v in hub_adj_sets else set(adj[v].tolist())
            for v in node_set
        }
        deg_in = tuple(sorted(
            sum(1 for u in node_set if u != v and u in local_adj[v])
            for v in node_list
        ))
        raw_counts[deg_in] += 1

    if n_valid == 0:
        return {}

    result: dict[tuple[int, ...], int] = {}
    for deg_seq, cnt in raw_counts.items():
        sigma    = _SIGMA.get(deg_seq, 1)
        estimated = (cnt / n_valid) * t / sigma / p_k
        result[deg_seq] = max(0, int(round(estimated)))
    return result


def cc_run_stars(
    g_und: igraph.Graph,
    n_samples: int,
    rng: np.random.Generator,
    *,
    _A: "scipy.sparse.csr_matrix | None" = None,
    _adj: "list[np.ndarray] | None" = None,
) -> dict[int, int]:
    """Colour-coding star-treelet estimator for induced k-star counts, k=2..10.

    A k-star = one centre connected to k leaves with NO edges between leaves.

    Star treelet DP (one run per k):
      1. Assign k+1 random colors.
      2. color_hist[v,c] = # neighbours of v with color c (sparse mat-mul).
      3. dp_star[v] = Π_{c ≠ color[v]} color_hist[v,c].
      4. Sample n_samples centres ∝ dp_star; pick one neighbour per non-own colour.
      5. Accept as induced k-star only if leaves have no mutual edges.
      6. Estimate: ĝ = (raw_star/n_valid) × t / p_{k+1}  (σ=1, centre roots the tree).
    """
    n = g_und.vcount()
    if n == 0:
        return {k: 0 for k in range(2, 11)}

    A_csr = _A   if _A   is not None else scipy.sparse.csr_matrix(
        g_und.get_adjacency_sparse()
    )
    adj   = _adj if _adj is not None else [
        np.array(g_und.neighbors(v), dtype=np.int32) for v in range(n)
    ]

    results: dict[int, int] = {}

    for k in range(2, 11):
        K   = k + 1
        p_K = math.factorial(K) / (K ** K)

        colors = rng.integers(0, K, size=n, dtype=np.int32)

        one_hot = np.zeros((n, K), dtype=np.float32)
        one_hot[np.arange(n), colors] = 1.0
        color_hist = (A_csr @ one_hot).astype(np.float64)

        dp_star = np.ones(n, dtype=np.float64)
        for c in range(K):
            mask = (colors != c)
            dp_star[mask] *= color_hist[mask, c]

        t = float(dp_star.sum())
        if t == 0:
            results[k] = 0
            continue

        w       = dp_star / t
        centres = rng.choice(n, size=n_samples, p=w)

        unique_centres = np.unique(centres)
        adj_by_color: dict[int, dict[int, np.ndarray]] = {}
        for v in unique_centres:
            v = int(v)
            nb = adj[v]
            adj_by_color[v] = (
                {c: nb[colors[nb] == c] for c in range(K)}
                if len(nb) > 0
                else {c: np.array([], dtype=np.int32) for c in range(K)}
            )

        raw_star = 0
        n_valid  = 0

        for centre in centres:
            v  = int(centre)
            c0 = int(colors[v])
            leaf_nodes = [v]
            ok = True

            for c in range(K):
                if c == c0:
                    continue
                cands = adj_by_color[v][c]
                if len(cands) == 0:
                    ok = False
                    break
                leaf_nodes.append(int(cands[rng.integers(len(cands))]))

            if not ok or len(set(leaf_nodes)) != K:
                continue
            n_valid += 1

            leaf_set = set(leaf_nodes)
            deg_in   = sorted(
                int(np.sum(np.isin(adj[u], list(leaf_set))))
                for u in leaf_nodes
            )
            if deg_in == [1] * k + [k]:
                raw_star += 1

        if n_valid == 0:
            results[k] = 0
        else:
            results[k] = max(0, int(round((raw_star / n_valid) * t / p_K)))

    return results


# ── Incremental delta helpers ────────────────────────────────────────────────
# Used by stage3's SA loop to update cached motif counts per swap without a
# full remeasure. Not part of the MotifCounter interface — they operate on the
# live adj dict maintained inside refine(), not on an igraph.Graph.

def _count_motifs4_through_edge(adj: list, u: int, v: int) -> dict[tuple, int]:
    """Count 4-node motif instances containing undirected edge {u, v}.

    Iterates unordered pairs from (N(u)∪N(v))\\{u,v}, looks up the 5 remaining
    possible edges in O(1), classifies by sorted degree sequence.
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
            du = 1 + uw + ux   # edge {u,v} is always present
            dv = 1 + vw + vx
            ds = tuple(sorted((du, dv, dw, dx)))
            if ds in MOTIF4_DS:
                counts[ds] = counts.get(ds, 0) + 1
    return counts


def _motif4_delta(
    adj: list, s1: int, o1: int, s2: int, o2: int
) -> dict[tuple, int]:
    """Compute change in 4-node motif counts from swapping (s1,o1)↔(s2,o2).

    Temporarily removes old edges and adds new ones so that motifs spanning
    both new edges are correctly detected.  Uses inclusion-exclusion to avoid
    double-counting 4-node subgraphs containing both swapped edges.
    Cost: O((deg_s1 + deg_o1 + deg_s2 + deg_o2)²).
    """
    def _adj_inc(u: int, v: int) -> None:
        adj[u][v] = adj[u].get(v, 0) + 1
        adj[v][u] = adj[v].get(u, 0) + 1

    def _adj_dec(u: int, v: int) -> None:
        adj[u][v] -= 1
        if adj[u][v] == 0:
            del adj[u][v]
        adj[v][u] -= 1
        if adj[v][u] == 0:
            del adj[v][u]

    def _overlap(a: int, b: int, c: int, d: int) -> dict[tuple, int]:
        if len({a, b, c, d}) < 4:
            return {}
        nodes = [a, b, c, d]
        degs = [sum(1 for nd2 in nodes if nd2 != nd and nd2 in adj[nd]) for nd in nodes]
        if min(degs) == 0:
            return {}
        ds = tuple(sorted(degs))
        return {ds: 1} if ds in MOTIF4_DS else {}

    def _count_pair(ea: tuple, eb: tuple) -> dict[tuple, int]:
        cu = _count_motifs4_through_edge(adj, *ea)
        cv = _count_motifs4_through_edge(adj, *eb)
        ov = _overlap(ea[0], ea[1], eb[0], eb[1])
        result: dict[tuple, int] = {}
        for k in set(cu) | set(cv) | set(ov):
            result[k] = cu.get(k, 0) + cv.get(k, 0) - ov.get(k, 0)
        return result

    before = _count_pair((s1, o1), (s2, o2))

    _adj_dec(s1, o1)
    _adj_dec(s2, o2)
    _adj_inc(s1, o2)
    _adj_inc(s2, o1)

    after = _count_pair((s1, o2), (s2, o1))

    _adj_dec(s1, o2)
    _adj_dec(s2, o1)
    _adj_inc(s1, o1)
    _adj_inc(s2, o2)

    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in set(before) | set(after)
        if after.get(k, 0) != before.get(k, 0)
    }


# ── Exact star counter ───────────────────────────────────────────────────────

def count_stars_exact(g: igraph.Graph) -> dict[int, int]:
    """Count induced k-stars exactly for k=2..10.

    A k-star is a centre node v connected to k leaves with NO edges between
    the leaves (induced subgraph condition).  For each node v with degree d,
    the count equals the number of size-k independent sets in N(v).

    Algorithm: for each node v, find the edges within N(v) (the
    neighbourhood-induced edge set E_v).  If E_v is empty, every k-subset
    is independent: contribute C(d, k).  Otherwise, count non-independent
    k-subsets via inclusion-exclusion over E_v:

        stars_k(v) = C(d,k) - Σ_{F⊆E_v, F≠∅} (-1)^{|F|+1} · C(d - |V(F)|, k - |V(F)|)

    Each term corresponds to subsets of edges F; V(F) is the set of
    endpoints of F.  We enumerate subsets of E_v up to size floor(k/2)
    (larger subsets can't cover a k-set with k nodes).  E_v is small for
    sparse KGs so this is fast in practice.

    Hub nodes (degree > HUB_THRESH) enumerate k-subsets directly and test
    independence rather than inclusion-exclusion, avoiding exponential blow-up
    on complete neighbourhoods.

    Cost: O(m · Δ + hub_count · Δ^MAX_K / MAX_K!) in the worst case; fast for
    the sparse neighbourhoods typical of KGs.
    """
    HUB_THRESH = 50   # above this degree, switch to direct subset enumeration
    MAX_K = 10

    n = g.vcount()
    if n == 0:
        return {k: 0 for k in range(2, MAX_K + 1)}

    # Build undirected neighbour sets (g should already be a simple graph)
    nbr: list[set[int]] = [set() for _ in range(n)]
    for e in g.es:
        nbr[e.source].add(e.target)
        nbr[e.target].add(e.source)

    totals = [0] * (MAX_K + 1)

    for v in range(n):
        nb = nbr[v]
        d = len(nb)
        if d < 2:
            continue

        nb_list = list(nb)

        # Find edges within N(v)
        inner_edges: list[tuple[int, int]] = []
        for u in nb_list:
            for w in nbr[u]:
                if w in nb and w > u:
                    inner_edges.append((u, w))

        if not inner_edges:
            # Pure star — every k-subset is independent
            for k in range(2, min(d, MAX_K) + 1):
                totals[k] += math.comb(d, k)
            continue

        if d > HUB_THRESH:
            # For high-degree nodes enumerate k-subsets and test independence directly.
            # Costly only for large k, but MAX_K=10 keeps it tractable when d≤500.
            nb_arr = nb_list
            inner_set: set[tuple[int, int]] = set(inner_edges)
            from itertools import combinations
            for k in range(2, min(d, MAX_K) + 1):
                cnt = 0
                for subset in combinations(nb_arr, k):
                    ok = True
                    for i in range(k):
                        for j in range(i + 1, k):
                            a, b = subset[i], subset[j]
                            if (min(a, b), max(a, b)) in inner_set:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        cnt += 1
                totals[k] += cnt
            continue

        # Inclusion-exclusion over subsets of inner_edges.
        # For each non-empty subset F of E_v: V(F) = endpoints of edges in F.
        # Contribution to stars_k(v): (-1)^{|F|+1} * C(d - |V(F)|, k - |V(F)|).
        # We only need subsets up to size floor(k/2) since |V(F)| ≥ 2|F|/… but
        # iterating all 2^|E_v| subsets is fine for small |E_v| (sparse KGs).
        ie = len(inner_edges)
        # delta[k] = correction to subtract from C(d,k)
        correction = [0] * (MAX_K + 1)
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
            sign = (-1) ** (sign_exp + 1)  # inclusion-exclusion sign
            for k in range(s, min(d, MAX_K) + 1):
                correction[k] += sign * math.comb(d - s, k - s)

        for k in range(2, min(d, MAX_K) + 1):
            totals[k] += math.comb(d, k) - correction[k]

    return {k: totals[k] for k in range(2, MAX_K + 1)}


# ── Abstract base ────────────────────────────────────────────────────────────

class MotifCounter(ABC):
    """Counts motif instances in an undirected simple graph, grouped by family.

    Implementations are selected via module-level constants in each consumer
    (``INITIAL_MOTIF_COUNTER``/``REMEASURE_MOTIF_COUNTER`` in stage3.py,
    ``MOTIF_COUNTER`` in block_e.py).
    """

    @abstractmethod
    def count_triangles(self, g: igraph.Graph) -> int:
        """Count triangles (3-node cycles) exactly."""

    @abstractmethod
    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        """Count k-node connected graphlets.

        Returns ``{sorted_degree_sequence_tuple: count}``.
        ``ExactMotifCounter`` only supports k ≤ 4; raises ``NotImplementedError``
        for larger k.
        """

    @abstractmethod
    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        """Count induced k-stars for k=2..10. Returns ``{k: count}``."""

    # Convenience wrappers — concrete, not abstract:
    def count_motifs3(self, g: igraph.Graph) -> dict[tuple, int]:
        """Count 3-node graphlets: (2,2,2)=triangle, (1,1,2)=open wedge."""
        return self.count_motifsk(g, 3)

    def count_motifs4(self, g: igraph.Graph) -> dict[tuple, int]:
        """Count 4-node connected motifs (C4, diamond, K4, paw)."""
        return self.count_motifsk(g, 4)


# ── Implementations ──────────────────────────────────────────────────────────

class CCMotifCounter(MotifCounter):
    """Colour-coding estimator (Bressan et al. 2021).

    Triangle count is exact (via igraph ``list_triangles``); all graphlet and
    star counts are estimated by the colour-coding sampler.
    """

    def __init__(self, n_samples: int = 10_000, seed: int = 1) -> None:
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    def count_triangles(self, g: igraph.Graph) -> int:
        return len(g.list_triangles()) if g.vcount() >= 3 else 0

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        return cc_run(g, k, self._n_samples, self._rng)

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        return count_stars_exact(g)


class ExactMotifCounter(MotifCounter):
    """Exact motif counter via full subgraph enumeration (k ≤ 4 only).

    Triangle count uses igraph's ``list_triangles``; k=3 and k=4 graphlets are
    counted by direct enumeration.  Cost is O(m·Δ²) for k=4 where Δ is the
    maximum degree.

    Raises ``NotImplementedError`` for k ≥ 5.
    """

    def count_triangles(self, g: igraph.Graph) -> int:
        return len(g.list_triangles()) if g.vcount() >= 3 else 0

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        if k == 2:
            m = g.ecount()
            return {(1, 1): m} if m > 0 else {}
        if k == 3:
            return self._count_motifs3(g)
        if k == 4:
            return self._count_motifs4_exact(g)
        raise NotImplementedError(
            f"ExactMotifCounter does not support k={k}; use CCMotifCounter for k ≥ 5"
        )

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        return count_stars_exact(g)

    def _count_motifs3(self, g: igraph.Graph) -> dict[tuple, int]:
        n = g.vcount()
        if n < 2:
            return {}
        tris = len(g.list_triangles()) if n >= 3 else 0
        degs = g.degree()
        wedges = sum(d * (d - 1) // 2 for d in degs) - 3 * tris
        result: dict[tuple, int] = {}
        if tris > 0:
            result[(2, 2, 2)] = tris
        if wedges > 0:
            result[(1, 1, 2)] = wedges
        return result

    def _count_motifs4_exact(self, g: igraph.Graph) -> dict[tuple, int]:
        n = g.vcount()
        adj: list[dict] = [{} for _ in range(n)]
        for e in g.es:
            u, v = e.source, e.target
            adj[u][v] = 1
            adj[v][u] = 1

        counts: dict[tuple, int] = {}
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for ds, cnt in _count_motifs4_through_edge(adj, u, v).items():
                    counts[ds] = counts.get(ds, 0) + cnt

        # Each 4-node subgraph is discovered once per edge it contributes to a
        # _count_motifs4_through_edge call.  Divisors = number of edges per motif type
        # that actually produce a valid (w,x) pair in the enumeration:
        #   C4 (2,2,2,2): 4 edges   → /4
        #   Diamond (2,2,3,3): 5    → /5
        #   K4 (3,3,3,3): 6         → /6
        #   Paw (1,2,2,3): 3        → /3
        _DIVISOR = {(2, 2, 2, 2): 4, (2, 2, 3, 3): 5, (3, 3, 3, 3): 6, (1, 2, 2, 3): 3}
        return {
            ds: cnt // _DIVISOR[ds]
            for ds, cnt in counts.items()
            if ds in _DIVISOR and cnt > 0
        }


class ESCAPEFiveNodeCounter:
    """Exact 5-node graphlet counting, inspired by ESCAPE (Pinar, Seshadhri, Vishal — WWW 2017).

    Enumerates all 5-node induced connected subgraphs exactly.
    Anchors each 5-set at its minimum-index node u, then BFS-expands to build
    all connected 5-node sets reachable from u, ensuring each set is counted once.

    Cost: O(m · Δ³) in the worst case; fast on sparse graphs (KGs).
    Exact and deterministic — no sampling variance.
    """

    # Maximum degree for exact enumeration. Above this, hubs make the BFS
    # expansion too expensive; callers should fall back to CC sampling.
    MAX_DEGREE_EXACT = 50

    def count_motifs5(self, g: igraph.Graph) -> dict[tuple, int]:
        """Count 5-node connected induced subgraphs, grouped by degree sequence.

        Returns {sorted_degree_seq_tuple: count}, or raises ``RuntimeError``
        if the graph's max degree exceeds ``MAX_DEGREE_EXACT`` (hub nodes make
        exact enumeration impractical — fall back to CC sampling).
        """
        n = g.vcount()
        if n < 5:
            return {}

        max_deg = max(g.degree()) if n > 0 else 0
        if max_deg > self.MAX_DEGREE_EXACT:
            raise RuntimeError(
                f"ESCAPEFiveNodeCounter: max degree {max_deg} > {self.MAX_DEGREE_EXACT}; "
                "graph too dense for exact 5-node enumeration — use CC sampling instead."
            )

        adj: list[set[int]] = [set() for _ in range(n)]
        for e in g.es:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)

        counts: dict[tuple, int] = defaultdict(int)

        def _deg5(five: tuple) -> tuple:
            five_set = set(five)
            return tuple(sorted(
                sum(1 for nb in five_set if nb != nd and nb in adj[nd])
                for nd in five_set
            ))

        # Enumerate all 5-node connected induced subgraphs anchored at u = min(5-set).
        # Grow connected partial sets by DFS; deduplicate via sorted-tuple key.
        for u in range(n - 4):
            seen_partial: set[tuple] = set()
            stack: list[tuple] = [(u,)]

            while stack:
                partial = stack.pop()
                partial_set = set(partial)
                reach = set()
                for nd in partial:
                    reach |= adj[nd]
                reach -= partial_set
                reach = {v for v in reach if v > u}

                for v in reach:
                    new_partial = tuple(sorted(partial_set | {v}))
                    if new_partial in seen_partial:
                        continue
                    seen_partial.add(new_partial)
                    if len(new_partial) == 5:
                        deg_seq = _deg5(new_partial)
                        counts[deg_seq] += 1
                    else:
                        stack.append(new_partial)

        return dict(counts)


class HybridMotifCounter(MotifCounter):
    """Exact counting for triangles, k=3, k=4; ESCAPE-exact for k=5; CC sampling for k≥6.

    Uses ESCAPEFiveNodeCounter for k=5 (deterministic, no sampling variance).
    Recommended for signature measurement — avoids CC variance on the most
    common motifs while staying tractable for large k.
    """

    def __init__(self, n_samples: int = 10_000, seed: int = 1) -> None:
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)
        self._exact = ExactMotifCounter()
        self._escape5 = ESCAPEFiveNodeCounter()

    def count_triangles(self, g: igraph.Graph) -> int:
        return self._exact.count_triangles(g)

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        if k <= 4:
            return self._exact.count_motifsk(g, k)
        if k == 5:
            try:
                return self._escape5.count_motifs5(g)
            except RuntimeError:
                # High-degree hub nodes make exact enumeration impractical; use CC.
                return cc_run(g, k, self._n_samples, self._rng)
        return cc_run(g, k, self._n_samples, self._rng)

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        return count_stars_exact(g)
