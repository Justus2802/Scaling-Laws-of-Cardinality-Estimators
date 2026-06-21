"""Shared helpers and CC-sampling functions for motif counting.

Module-level constants and functions used by all counter implementations.
Graphlet-type constants live on MotifCounter (the base class) and are accessed
here via MotifCounter.SIGMA.
"""

import math
from collections import defaultdict

import igraph
import numpy as np
import scipy.sparse

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


def cc_run(
    g_und: igraph.Graph,
    k: int,
    n_samples: int,
    rng: np.random.Generator,
    *,
    n_colorings: int = 1,
    _A: "scipy.sparse.csr_matrix | None" = None,
    _adj: "list[np.ndarray] | None" = None,
) -> dict[tuple[int, ...], int]:
    """Colour-coding estimator for k-node graphlet counts (Bressan et al. 2021).

    Each colouring randomly assigns k colors, builds a directed path-treelet DP
    via sparse matrix products, samples ``n_samples`` colorful k-paths by
    backtracking, and yields a per-type estimate.  A single colouring detects any
    given k-motif only when it is "colorful" (all k vertices distinct colors),
    probability ``p_k = k!/k^k`` — only ~1.5% at k=6 — so on graphs with few or
    clustered instances one colouring frequently finds nothing (DP total t = 0).

    Following color-coding (Alon–Yuster–Zwick 1995), this averages the unbiased
    per-colouring estimate over ``n_colorings`` independent colourings: it both
    escapes the all-zero failure mode and cuts variance ~1/n_colorings.  The
    averaging count is fixed in advance (not conditioned on observed counts), so
    the estimator stays unbiased.

    Returns ``{degree_sequence_tuple: estimated_count}``.
    """
    p_k = math.factorial(k) / (k ** k)

    n = g_und.vcount()
    if n < k:
        return {}

    n_sets   = 1 << k
    full_set = n_sets - 1

    # Colouring-independent structure — built once, reused across all colourings.
    A = _A if _A is not None else scipy.sparse.csr_matrix(
        g_und.get_adjacency_sparse()
    ).astype(np.float32)
    adj = _adj if _adj is not None else [
        np.array(g_und.neighbors(v), dtype=np.int32) for v in range(n)
    ]
    _HUB_ADJ_THRESH = 200
    hub_adj_sets: dict[int, set[int]] = {
        v: set(adj[v].tolist())
        for v in range(n)
        if len(adj[v]) > _HUB_ADJ_THRESH
    }

    def _one_coloring() -> dict[tuple[int, ...], float]:
        """Unbiased per-type estimate from a single random colouring ({} if t=0)."""
        colors = rng.integers(0, k, size=n, dtype=np.int32)

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

        return {
            deg_seq: (cnt / n_valid) * t / MotifCounter.SIGMA.get(deg_seq, 1) / p_k
            for deg_seq, cnt in raw_counts.items()
        }

    # Average the per-colouring estimates (missing types contribute 0).
    sums: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for _ in range(max(1, n_colorings)):
        for deg_seq, est in _one_coloring().items():
            sums[deg_seq] += est

    denom = max(1, n_colorings)
    result: dict[tuple[int, ...], int] = {}
    for deg_seq, total in sums.items():
        val = max(0, int(round(total / denom)))
        if val > 0:
            result[deg_seq] = val
    return result


def cc_run_stars(
    g_und: igraph.Graph,
    n_samples: int,
    rng: np.random.Generator,
    *,
    _A: "scipy.sparse.csr_matrix | None" = None,
    _adj: "list[np.ndarray] | None" = None,
) -> dict[int, int]:
    """Colour-coding star-treelet estimator for induced k-star counts, k=2..10."""
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


# Maximum degree for exact 5-node enumeration via ESCAPE.
_ESCAPE_MAX_DEGREE = 50


def count_motifs5_escape(g: igraph.Graph) -> dict[tuple, int]:
    """Exact 5-node graphlet counts via BFS expansion (ESCAPE, WWW 2017).

    Anchors each 5-set at its minimum-index node, expands connected partial
    sets by BFS, deduplicates via sorted-tuple key.  Cost: O(m·Δ³).
    Raises ``RuntimeError`` if max degree exceeds ``_ESCAPE_MAX_DEGREE``;
    callers should fall back to CC sampling in that case.
    """
    n = g.vcount()
    if n < 5:
        return {}

    max_deg = max(g.degree()) if n > 0 else 0
    if max_deg > _ESCAPE_MAX_DEGREE:
        raise RuntimeError(
            f"ESCAPE: max degree {max_deg} > {_ESCAPE_MAX_DEGREE}; "
            "use CC sampling instead."
        )

    adj: list[set[int]] = [set() for _ in range(n)]
    for e in g.es:
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)

    counts: defaultdict[tuple, int] = defaultdict(int)

    def _deg5(five: tuple) -> tuple:
        five_set = set(five)
        return tuple(sorted(
            sum(1 for nb in five_set if nb != nd and nb in adj[nd])
            for nd in five_set
        ))

    for u in range(n - 4):
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
                if len(new_partial) == 5:
                    counts[_deg5(new_partial)] += 1
                else:
                    stack.append(new_partial)

    return dict(counts)


