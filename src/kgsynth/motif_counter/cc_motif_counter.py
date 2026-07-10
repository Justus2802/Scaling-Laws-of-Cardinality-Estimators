"""Colour-coding motif counter and its sampling functions (Bressan et al. 2021).

Holds the colour-coding estimators ``cc_run`` (graphlets) and ``cc_run_stars`` /
``cc_run_stars_loop`` (induced stars) together with ``CCMotifCounter``, the
``MotifCounter`` implementation that drives them.
"""

import math
from collections import defaultdict

import igraph
import numpy as np
import scipy.sparse

from ._base import MotifCounter
from .._logging import get_logger

log = get_logger(__name__)

# Emit a progress log every this many colourings in the CC estimators.
_COLORING_LOG_EVERY = 4


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
    _n = max(1, n_colorings)
    for _i in range(_n):
        for deg_seq, est in _one_coloring().items():
            sums[deg_seq] += est
        if (_i + 1) % _COLORING_LOG_EVERY == 0:
            log.info("CC k=%d motifs: %d/%d colourings done", k, _i + 1, _n)

    denom = max(1, n_colorings)
    result: dict[tuple[int, ...], int] = {}
    for deg_seq, total in sums.items():
        val = max(0, int(round(total / denom)))
        if val > 0:
            result[deg_seq] = val
    return result


def cc_run_stars_loop(
    g_und: igraph.Graph,
    n_samples: int,
    rng: np.random.Generator,
    *,
    n_colorings: int = 1,
    _A: "scipy.sparse.csr_matrix | None" = None,
    _adj: "list[np.ndarray] | None" = None,
) -> dict[int, int]:
    """Reference (un-vectorised) colour-coding star estimator — see ``cc_run_stars``.

    Identical estimator to :func:`cc_run_stars` but with the per-centre sampling
    loop written in plain Python.  Retained only as a correctness/speed baseline
    for benchmarking against the vectorised implementation; production code calls
    :func:`cc_run_stars`.
    """
    n = g_und.vcount()
    # No edges → no stars (and igraph's sparse adjacency rejects edgeless graphs).
    if n == 0 or g_und.ecount() == 0:
        return {k: 0 for k in range(2, 11)}

    A_csr = _A   if _A   is not None else scipy.sparse.csr_matrix(
        g_und.get_adjacency_sparse()
    )
    adj   = _adj if _adj is not None else [
        np.array(g_und.neighbors(v), dtype=np.int32) for v in range(n)
    ]

    def _one_coloring(k: int, K: int, p_K: float) -> float:
        """Unbiased induced-k-star estimate from a single random colouring."""
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
            return 0.0

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
            return 0.0
        return (raw_star / n_valid) * t / p_K

    results: dict[int, int] = {}
    for k in range(2, 11):
        K   = k + 1
        p_K = math.factorial(K) / (K ** K)
        # Average the unbiased per-colouring estimate over n_colorings draws.
        total = 0.0
        for _i in range(n_colorings):
            total += _one_coloring(k, K, p_K)
            if (_i + 1) % _COLORING_LOG_EVERY == 0:
                log.info("CC k=%d stars: %d/%d colourings done", k, _i + 1, n_colorings)
        results[k] = max(0, int(round(total / n_colorings)))

    return results


def cc_run_stars(
    g_und: igraph.Graph,
    n_samples: int,
    rng: np.random.Generator,
    *,
    n_colorings: int = 1,
    _A: "scipy.sparse.csr_matrix | None" = None,
    _adj: "list[np.ndarray] | None" = None,
) -> dict[int, int]:
    """Colour-coding star-treelet estimator for induced k-star counts, k=2..10.

    A k-star is detected by one colouring only when its K=k+1 vertices are all
    coloured distinctly (prob ``p_K = K!/K^K`` — e.g. ~7e-3 at k=5, ~1e-4 at
    k=10), so a single colouring frequently samples nothing at large k and
    collapses the estimate to 0.  Following colour-coding (Alon–Yuster–Zwick
    1995; Bressan et al. 2021), the per-colouring unbiased estimate is averaged
    over ``n_colorings`` independent colourings: this escapes the all-zero
    failure mode and cuts variance ~``1/n_colorings``.  The averaging count is
    fixed in advance (not conditioned on observed counts), so the estimator
    stays unbiased.  Cost scales linearly with ``n_colorings``.

    This is the vectorised implementation: the ``n_samples`` centre samples
    are processed with batched NumPy array ops (one uniform leaf draw per colour
    over all samples, and a batched ``searchsorted`` adjacency test) rather than
    a per-centre Python loop.  It is the same unbiased estimator as
    :func:`cc_run_stars_loop` (identical sampling distribution, different RNG
    stream), just much faster.
    """
    n = g_und.vcount()
    # No edges → no stars (and igraph's sparse adjacency rejects edgeless graphs).
    if n == 0 or g_und.ecount() == 0:
        return {k: 0 for k in range(2, 11)}

    A_csr = _A if _A is not None else scipy.sparse.csr_matrix(
        g_und.get_adjacency_sparse()
    )

    # Colouring-independent edge structures, built once and reused across all
    # colourings/star sizes. ``edge_keys`` is a sorted array of undirected-edge
    # keys (lo*n + hi) for O(log m) vectorised adjacency queries; ``src_all`` /
    # ``dst_all`` hold both directions of every edge for per-colour grouping.
    ei = np.array(g_und.get_edgelist(), dtype=np.int64)
    lo_e = np.minimum(ei[:, 0], ei[:, 1])
    hi_e = np.maximum(ei[:, 0], ei[:, 1])
    edge_keys = np.unique(lo_e * n + hi_e)
    src_all = np.concatenate([ei[:, 0], ei[:, 1]])
    dst_all = np.concatenate([ei[:, 1], ei[:, 0]])

    def _is_edge(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Vectorised undirected-adjacency test for paired node-id arrays."""
        ekey = np.minimum(u, v) * n + np.maximum(u, v)
        pos = np.clip(np.searchsorted(edge_keys, ekey), 0, len(edge_keys) - 1)
        return edge_keys[pos] == ekey

    def _one_coloring(k: int, K: int, p_K: float) -> float:
        """Unbiased induced-k-star estimate from a single random colouring.

        Fully vectorised over the ``n_samples`` centre samples: leaves are
        drawn with one batched uniform draw per colour (via a per-(node,colour)
        CSR-style neighbour grouping), and the induced-star test is a batched
        edge-count over the K sampled vertices.
        """
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
            return 0.0

        w       = dp_star / t
        S       = n_samples
        centres = rng.choice(n, size=S, p=w)

        # Group neighbours by (node, colour): ``dst_sorted`` holds every directed
        # neighbour ordered by key = node*K + nbr_colour, so the colour-``c``
        # neighbours of node ``v`` are the contiguous slice
        # ``dst_sorted[offsets[v*K+c] : offsets[v*K+c+1]]``. ``counts_flat`` is
        # the block size = #colour-c neighbours of ``v`` ( = color_hist[v, c] ).
        counts_flat = np.rint(color_hist).astype(np.int64).reshape(-1)
        offsets = np.empty(n * K + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts_flat, out=offsets[1:])
        order = np.argsort(src_all * K + colors[dst_all])
        dst_sorted = dst_all[order]

        c0    = colors[centres].astype(np.int64)
        nodes = np.zeros((S, K), dtype=np.int64)   # column c holds the colour-c vertex
        valid = np.ones(S, dtype=bool)

        # For each colour c, the samples whose centre is NOT colour c need a leaf
        # of colour c. Draw one uniform neighbour per such sample in a batch.
        for c in range(K):
            idx = np.nonzero(c0 != c)[0]
            if len(idx) == 0:
                continue
            key  = centres[idx].astype(np.int64) * K + c
            cnt  = counts_flat[key]
            zero = cnt == 0
            valid[idx[zero]] = False              # no colour-c neighbour → invalid
            safe = np.where(zero, 1, cnt)
            r    = (rng.random(len(idx)) * safe).astype(np.int64)  # uniform in [0, cnt)
            pos  = np.clip(offsets[key] + np.where(zero, 0, r), 0, len(dst_sorted) - 1)
            nodes[idx, c] = dst_sorted[pos]
        nodes[np.arange(S), c0] = centres          # centre occupies its own colour column

        n_valid = int(valid.sum())
        if n_valid == 0:
            return 0.0

        # Induced-star test: the K vertices have distinct colours (hence are
        # distinct nodes), and each leaf is adjacent to the centre by
        # construction. The sample is a pure induced k-star iff the only edges
        # among the K vertices are the k centre→leaf edges — i.e. the count of
        # adjacent vertex pairs equals k (no leaf-leaf edge).
        adj_count = np.zeros(S, dtype=np.int64)
        for a in range(K):
            for b in range(a + 1, K):
                adj_count += _is_edge(nodes[:, a], nodes[:, b])

        raw_star = int(np.count_nonzero(valid & (adj_count == k)))
        return (raw_star / n_valid) * t / p_K

    results: dict[int, int] = {}
    for k in range(2, 11):
        K   = k + 1
        p_K = math.factorial(K) / (K ** K)
        # Average the unbiased per-colouring estimate over n_colorings draws.
        total = 0.0
        for _i in range(n_colorings):
            total += _one_coloring(k, K, p_K)
            if (_i + 1) % _COLORING_LOG_EVERY == 0:
                log.info("CC k=%d stars: %d/%d colourings done", k, _i + 1, n_colorings)
        results[k] = max(0, int(round(total / n_colorings)))

    return results


class CCMotifCounter(MotifCounter):
    """Colour-coding estimator (Bressan et al. 2021).

    Triangle count is exact (via igraph ``list_triangles``); all graphlet and
    star counts are estimated by the colour-coding sampler.

    ``n_colorings`` is the number of independent random colourings the estimate is
    averaged over.  A k-motif is detected by one colouring only when it is
    colourful (prob ``k!/k^k`` ≈ 1.5% at k=6), so a single colouring misses
    everything on graphs with few/clustered instances; averaging several
    colourings (Alon–Yuster–Zwick 1995; Motivo / Bressan et al. 2021) escapes that
    all-zero failure and reduces variance ~``1/n_colorings``.  Cost scales linearly
    with it (each colouring rebuilds the O(m·2^k) DP).

    When ``adaptive`` is True the per-call path-sample count scales with graph
    size instead of being fixed: ``max(500, min(n·20, n_samples·5))`` where ``n``
    is the node count and ``n_samples`` acts as the base budget.  Tiny graphs stay
    fast (floor 500) while large graphs draw up to ``n_samples·5`` samples.  The
    resolved sample count drives both the graphlet path sampler (``cc_run``) and
    the star centre samples (``cc_run_stars``).
    """

    def __init__(self, n_samples: int = 5_000, seed: int = 1, n_colorings: int = 16,
                 adaptive: bool = False) -> None:
        self._n_samples = n_samples
        self._n_colorings = n_colorings
        self._adaptive = adaptive
        self._rng = np.random.default_rng(seed)

    def _resolve_samples(self, g: igraph.Graph) -> int:
        """Path-sample count for *g* — size-scaled when ``adaptive``, else fixed.

        :param g: undirected simple graph being counted.
        """
        if not self._adaptive:
            return self._n_samples
        return max(500, min(g.vcount() * 20, self._n_samples * 5))

    def count_triangles(self, g: igraph.Graph) -> int:
        return len(g.list_triangles()) if g.vcount() >= 3 else 0

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        n_samples = self._resolve_samples(g)
        log.info("CC count_motifsk k=%d: %d samples × %d colourings (n=%d%s)",
                 k, n_samples, self._n_colorings, g.vcount(),
                 ", adaptive" if self._adaptive else "")
        return cc_run(g, k, n_samples, self._rng, n_colorings=self._n_colorings)

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        n_samples = self._resolve_samples(g)
        log.info("CC count_stars k=2..10: %d samples × %d colourings (n=%d%s)",
                 n_samples, self._n_colorings, g.vcount(),
                 ", adaptive" if self._adaptive else "")
        return cc_run_stars(g, n_samples, self._rng, n_colorings=self._n_colorings)
