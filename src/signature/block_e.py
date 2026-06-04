"""Block E — Motif shape distribution features."""

import math
from collections import defaultdict
from typing import Any

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.sparse

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED

log = get_logger(__name__)

_SAMPLE_BUDGET = 100_000  # default walk samples for path/tree templates
_MAX_K         = 10       # longest path template walk
_LARGE_N       = 50_000   # above this, sample an induced subgraph for structural counts
_SAMPLE_N      = 10_000   # seed nodes when n > _LARGE_N (expanded to full neighborhoods)


class BlockE(SignatureBlock):
    """Block E — Motif shape distribution of a KG.

    Exact counts for 3- and 4-node motifs on the undirected simplification.
    Path and tree templates are estimated by random walk sampling.

    Usage::

        b = BlockE().calculate(g)
        b.as_vector()                      # fixed-length comparison vector
        b.as_dict()                        # named key-value pairs
        b.visualize()                      # interactive matplotlib figure
        b.visualize(mode="text")           # CLI summary
        b.visualize(path="out.png")        # save plot to file
    """

    def __init__(self) -> None:
        self._triangle_count = _NOT_CALCULATED
        self._four_cycle_count = _NOT_CALCULATED
        self._five_cycle_count = _NOT_CALCULATED
        self._six_cycle_count = _NOT_CALCULATED
        self._diamond_count = _NOT_CALCULATED
        self._k4_count = _NOT_CALCULATED
        self._tailed_triangle_count = _NOT_CALCULATED
        self._star_counts = _NOT_CALCULATED
        self._path_template_zipf = _NOT_CALCULATED
        self._path_template_entropy = _NOT_CALCULATED
        self._tree_template_zipf = _NOT_CALCULATED
        self._tree_template_entropy = _NOT_CALCULATED

    @property
    def triangle_count(self) -> int:
        return self._require("triangle_count", self._triangle_count)

    @property
    def four_cycle_count(self) -> int:
        return self._require("four_cycle_count", self._four_cycle_count)

    @property
    def five_cycle_count(self) -> int:
        return self._require("five_cycle_count", self._five_cycle_count)

    @property
    def six_cycle_count(self) -> int:
        return self._require("six_cycle_count", self._six_cycle_count)

    @property
    def diamond_count(self) -> int:
        return self._require("diamond_count", self._diamond_count)

    @property
    def k4_count(self) -> int:
        return self._require("k4_count", self._k4_count)

    @property
    def tailed_triangle_count(self) -> int:
        return self._require("tailed_triangle_count", self._tailed_triangle_count)

    @property
    def star_counts(self) -> dict[int, int]:
        return self._require("star_counts", self._star_counts)

    @property
    def path_template_zipf(self) -> dict[int, float]:
        return self._require("path_template_zipf", self._path_template_zipf)

    @property
    def path_template_entropy(self) -> dict[int, float]:
        return self._require("path_template_entropy", self._path_template_entropy)

    @property
    def tree_template_zipf(self) -> float:
        return self._require("tree_template_zipf", self._tree_template_zipf)

    @property
    def tree_template_entropy(self) -> float:
        return self._require("tree_template_entropy", self._tree_template_entropy)

    def calculate(self, g: igraph.Graph, sample_budget: int = _SAMPLE_BUDGET) -> "BlockE":
        """Compute Block E (motif distribution) of the graph signature.

        Exact counts for 3- and 4-node motifs on the undirected simplification.
        Path and tree templates are estimated by random walk sampling.
        For large graphs (n >= _LARGE_N), 4-node and 5/6-cycle counts are
        estimated via color coding (Bressan et al. 2021) on the full graph.
        The CC DP scales with m (edges), not n^k, so no subgraph sampling needed.
        """
        g_und = g.as_undirected(combine_edges="first").simplify()
        n = g_und.vcount()
        m = g_und.ecount()
        log.info("Block E: graph has %d nodes, %d edges", n, m)

        # n_samples scales with graph size so tiny test graphs stay fast (500 samples)
        # while large graphs get up to budget×5 samples.  Sampling is O(n_samples×k)
        # but negligible vs the DP build O(m×2^k); one coloring is enough for large
        # graphs (paper §2.3: coloring noise averages out over many instances).
        _n_samples = max(500, min(n * 20, sample_budget * 5))
        _rng       = np.random.default_rng(1)

        if n == 0:
            self._triangle_count = 0
            self._four_cycle_count = self._diamond_count = 0
            self._k4_count = self._tailed_triangle_count = 0
            self._star_counts = {k: 0 for k in range(2, 11)}
            self._five_cycle_count = self._six_cycle_count = 0
            self._path_template_zipf = {}
            self._path_template_entropy = {}
            self._tree_template_zipf = float("nan")
            self._tree_template_entropy = float("nan")
            return self

        # Build adjacency structures once and reuse across all CC calls.
        # Building _adj (n numpy arrays) and _A (sparse matrix) takes O(n+m) and
        # is otherwise repeated for every k — 4 times in total.
        _A   = scipy.sparse.csr_matrix(g_und.get_adjacency_sparse()).astype(np.float32)
        _adj = [np.array(g_und.neighbors(v), dtype=np.int32) for v in range(n)]

        # Triangles: exact via igraph list_triangles() — O(m√m), reliable even on
        # large sparse graphs where CC misses triangles (too rare to sample).
        log.info("Block E: computing triangles (exact list_triangles)…")
        _tris = g_und.list_triangles() if n >= 3 else []
        self._triangle_count = len(_tris)
        log.info("Block E: computed triangle_count (%d)", self._triangle_count)

        # k=3 CC run for path_template_zipf[3] (triangle count is exact above).
        log.info("Block E: running CC k=3 for graphlet-type distribution (%d samples)…", _n_samples)
        motifs3 = BlockE._cc_run(g_und, 3, _n_samples, _rng, _A=_A, _adj=_adj)

        log.info("Block E: computing 4-node motifs (CC k=4, %d samples)…", _n_samples)
        motifs4 = BlockE._cc_run(g_und, 4, _n_samples, _rng, _A=_A, _adj=_adj)
        self._four_cycle_count      = motifs4.get((2, 2, 2, 2), 0)
        self._diamond_count         = motifs4.get((2, 2, 3, 3), 0)
        self._k4_count              = motifs4.get((3, 3, 3, 3), 0)
        self._tailed_triangle_count = motifs4.get((1, 2, 2, 3), 0)
        log.info("Block E: computed four_cycle_count (%d)", self._four_cycle_count)
        log.info("Block E: computed diamond_count (%d)", self._diamond_count)
        log.info("Block E: computed k4_count (%d)", self._k4_count)
        log.info("Block E: computed tailed_triangle_count (%d)", self._tailed_triangle_count)

        log.info("Block E: computing stars (CC star treelet, k=2..10, %d samples)…", _n_samples)
        self._star_counts = self._cc_run_stars(g_und, _n_samples, _rng, _A=_A, _adj=_adj)
        log.info(
            "Block E: computed star_counts (k=2..10 totals=%s)",
            [self._star_counts.get(k, 0) for k in range(2, 11)],
        )

        log.info("Block E: computing 5-cycle (CC k=5, %d samples)…", _n_samples)
        motifs5 = BlockE._cc_run(g_und, 5, _n_samples, _rng, _A=_A, _adj=_adj)
        self._five_cycle_count = motifs5.get((2, 2, 2, 2, 2), 0)
        log.info("Block E: computed five_cycle_count (~%d)", self._five_cycle_count)

        log.info("Block E: computing 6-cycle (CC k=6, %d samples)…", _n_samples)
        motifs6 = BlockE._cc_run(g_und, 6, _n_samples, _rng, _A=_A, _adj=_adj)
        self._six_cycle_count = motifs6.get((2, 2, 2, 2, 2, 2), 0)
        log.info("Block E: computed six_cycle_count (~%d)", self._six_cycle_count)

        # Path templates: for each k, compute Zipf + entropy of the CC graphlet-type
        # distribution at that size.  k=2..6 always run.  k=7..10 are run only when
        # the DP fits in ~1 GB: n × 2^k × k × 4 bytes ≤ 1 GB → n ≤ 1e9 / (k×2^k×4).
        log.info("Block E: computing path templates (color coding k=2..10)…")
        motifs2 = BlockE._cc_run(g_und, 2, _n_samples, _rng, _A=_A, _adj=_adj)
        _cc_by_k = {2: motifs2, 3: motifs3, 4: motifs4, 5: motifs5, 6: motifs6}
        for _k in range(7, 11):
            _dp_bytes = n * (1 << _k) * _k * 4
            if _dp_bytes <= 1_000_000_000:
                log.info("Block E: computing path template k=%d (color coding)…", _k)
                _cc_by_k[_k] = BlockE._cc_run(g_und, _k, _n_samples, _rng, _A=_A, _adj=_adj)
            else:
                log.info(
                    "Block E: skipping path template k=%d (DP would need %.1f GB > 1 GB limit)",
                    _k, _dp_bytes / 1e9,
                )

        self._path_template_zipf    = {}
        self._path_template_entropy = {}
        for _k in range(2, 11):
            if _k in _cc_by_k:
                _z, _e = BlockE._template_stats(_cc_by_k[_k])
            else:
                _z, _e = float("nan"), float("nan")
            self._path_template_zipf[_k]    = _z
            self._path_template_entropy[_k] = _e
        log.info(
            "Block E: computed path_template_zipf (k=2..10 alphas=%s)",
            [round(self._path_template_zipf.get(k, float("nan")), 4) for k in range(2, 11)],
        )
        log.info(
            "Block E: computed path_template_entropy (k=2..10 entropies=%s)",
            [round(self._path_template_entropy.get(k, float("nan")), 4) for k in range(2, 11)],
        )

        # Tree templates: Zipf + entropy of how total motif counts scale across k.
        # Using total-count-per-k (rather than per-type) avoids NaN on small graphs
        # where CC returns only 1-2 distinct graphlet types.
        log.info("Block E: computing tree templates (CC total counts per k)…")
        _totals_by_k: dict[int, int] = {
            _k: sum(_cc_by_k[_k].values())
            for _k in _cc_by_k
            if sum(_cc_by_k[_k].values()) > 0
        }
        self._tree_template_zipf, self._tree_template_entropy = BlockE._template_stats(_totals_by_k)
        log.info(
            "Block E: computed tree_template stats (zipf_alpha=%.4f, entropy=%.4f)",
            self._tree_template_zipf, self._tree_template_entropy,
        )

        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 36-vector for cross-KG comparison."""
        vec = [
            float(self.triangle_count),
            float(self.four_cycle_count),
            float(self.five_cycle_count),
            float(self.six_cycle_count),
            float(self.diamond_count),
            float(self.k4_count),
            float(self.tailed_triangle_count),
        ]
        for k in range(2, 11):
            vec.append(float(self.star_counts.get(k, 0)))
        for k in range(2, 11):
            vec.append(self.path_template_zipf.get(k, float("nan")))
        for k in range(2, 11):
            vec.append(self.path_template_entropy.get(k, float("nan")))
        vec.extend([self.tree_template_zipf, self.tree_template_entropy])
        return vec  # length 7 + 9 + 9 + 9 + 2 = 36

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        names = [
            "triangle_count", "four_cycle_count", "five_cycle_count",
            "six_cycle_count", "diamond_count", "k4_count", "tailed_triangle_count",
        ]
        names += [f"star_count_k{k}" for k in range(2, _MAX_K + 1)]
        names += [f"path_template_zipf_k{k}" for k in range(2, _MAX_K + 1)]
        names += [f"path_template_entropy_k{k}" for k in range(2, _MAX_K + 1)]
        names += ["tree_template_zipf", "tree_template_entropy"]
        return names

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 36-element NaN vector (same length as as_vector())."""
        return [float("nan")] * 36

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for this block's computed features.

        Args:
            mode: "plot" for matplotlib, "text" for CLI summary.
            path: write to file instead of displaying interactively.
        """
        if mode == "text":
            self._visualize_text(path)
        elif mode == "plot":
            self._visualize_plot(path)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'plot' or 'text'.")

    # ── private helpers ───────────────────────────────────────────────────────

    def _visualize_text(self, path: str | None) -> None:
        lines: list[str] = []
        lines.append("=== Block E: Motif Shape Distribution ===\n")

        lines.append("--- Subgraph counts ---")
        lines.append(f"  triangles:           {self.triangle_count}")
        lines.append(f"  4-cycles:            {self.four_cycle_count}")
        lines.append(f"  5-cycles (est):      {self.five_cycle_count}")
        lines.append(f"  6-cycles (est):      {self.six_cycle_count}")
        lines.append(f"  diamonds:            {self.diamond_count}")
        lines.append(f"  K4:                  {self.k4_count}")
        lines.append(f"  tailed triangles:    {self.tailed_triangle_count}")

        lines.append("\n--- Star counts ---")
        star_row = "  " + "  ".join(f"k={k}: {self.star_counts.get(k, 0)}" for k in range(2, 11))
        lines.append(star_row)

        lines.append("\n--- Path templates ---")
        lines.append(f"  {'k':>3}  {'zipf_alpha':>10}  {'entropy':>10}")
        for k in range(2, 11):
            z = self.path_template_zipf.get(k, float("nan"))
            e = self.path_template_entropy.get(k, float("nan"))
            lines.append(f"  {k:>3}  {z:>10.4f}  {e:>10.4f}")

        lines.append("\n--- Tree templates (depth-2) ---")
        lines.append(f"  zipf_alpha={self.tree_template_zipf:.4f}  entropy={self.tree_template_entropy:.4f}")

        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # Motif / cycle counts
            ax = axes[0]
            labels = ["triangles", "4-cycles", "5-cycles\n(est)", "6-cycles\n(est)",
                      "diamonds", "K4", "tailed\ntriangles"]
            values = [
                self.triangle_count, self.four_cycle_count, self.five_cycle_count,
                self.six_cycle_count, self.diamond_count, self.k4_count,
                self.tailed_triangle_count,
            ]
            ax.bar(labels, values, color="steelblue")
            ax.set_ylabel("count")
            ax.set_title("Subgraph motif counts")
            ax.tick_params(axis="x", labelsize=8)

            # Star counts by k — use float to avoid C-long overflow on bigints
            ax = axes[1]
            ks = list(range(2, 11))
            star_vals = [float(self.star_counts.get(k, 0)) for k in ks]
            ax.bar([str(k) for k in ks], star_vals, color="darkorange")
            ax.set_xlabel("k")
            ax.set_ylabel("count")
            ax.set_title("k-star counts (CC induced)")

            # Path template zipf alpha and entropy vs k
            ax = axes[2]
            zipf_vals = [self.path_template_zipf.get(k, float("nan")) for k in ks]
            ent_vals = [self.path_template_entropy.get(k, float("nan")) for k in ks]
            ax.plot(ks, zipf_vals, "o-", label="zipf alpha", color="steelblue")
            ax2 = ax.twinx()
            ax2.plot(ks, ent_vals, "s--", label="entropy", color="darkorange")
            ax.set_xlabel("walk length k")
            ax.set_ylabel("zipf alpha", color="steelblue")
            ax2.set_ylabel("Shannon entropy", color="darkorange")
            ax.set_title("Path template statistics by walk length")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block E: plot failed: %s", exc, exc_info=True)
            plt.close("all")

    @staticmethod
    def _template_stats(counts: dict[tuple, int]) -> tuple[float, float]:
        """Return (Zipf exponent, Shannon entropy) from a {template: count} dict.

        Zipf exponent uses the continuous-MLE estimator (Clauset et al. 2009 §B.2):
          α = 1 + n / Σ ln(xᵢ / xmin)
        This avoids the powerlaw library's discrete normaliser (mpmath + Nelder-Mead),
        which is ~2s per call, while producing essentially the same estimate for the
        frequency counts seen in practice.
        """
        if not counts:
            return float("nan"), float("nan")
        freqs = np.array(list(counts.values()), dtype=float)
        pos = freqs[freqs >= 1.0]
        if pos.size < 2:
            zipf = float("nan")
        else:
            xmin  = float(np.min(pos))
            tail  = pos[pos >= xmin]
            denom = float(np.sum(np.log(tail / xmin)))
            zipf  = float(np.clip(1.0 + tail.size / denom, 1.01, 50.0)) if denom > 0 else float("nan")
        p = freqs / freqs.sum()
        p = p[p > 0]
        entropy = -float(np.sum(p * np.log(p)))
        return zipf, entropy

    @staticmethod
    def _build_out_adj(
        g: igraph.Graph,
    ) -> tuple[dict[int, list[tuple[int, str]]], list[int]]:
        """Build adjacency list for directed walks, skipping literal targets."""
        out_edges: defaultdict[int, list[tuple[int, str]]] = defaultdict(list)
        for e in g.es:
            if not g.vs[e.target]["is_literal"]:
                out_edges[e.source].append((e.target, e["predicate"]))
        start_verts: list[int] = [v for v, adj in out_edges.items() if not g.vs[v]["is_literal"]]
        return dict(out_edges), start_verts

    @staticmethod
    def _sample_all_path_templates(
        out_edges: dict[int, list[tuple[int, str]]],
        start_verts: np.ndarray,
        sample_budget: int,
        rng: np.random.Generator,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Sample walks up to length _MAX_K; record every prefix length in one pass.

        A single walk of length L contributes to the template counts for
        k=2,3,...,L, eliminating the need for 9 separate sampling rounds and
        reducing the total number of rng calls by ~_MAX_K/mean_length.
        """
        # Pre-sample all starting vertices at once — one bulk rng call.
        n_walks = max(1, sample_budget // _MAX_K)
        starts  = rng.choice(start_verts, size=n_walks)

        raw: dict[int, defaultdict[tuple[str, ...], int]] = {
            k: defaultdict(int) for k in range(2, _MAX_K + 1)
        }

        for start in starts:
            v    = int(start)
            rels: list[str] = []
            for _ in range(_MAX_K):
                adj = out_edges.get(v)
                if not adj:
                    break
                nb, rel = adj[int(rng.integers(len(adj)))]
                rels.append(rel)
                v = nb
                if len(rels) >= 2:
                    raw[len(rels)][tuple(rels)] += 1

        path_zipf:    dict[int, float] = {}
        path_entropy: dict[int, float] = {}
        for k in range(2, _MAX_K + 1):
            path_zipf[k], path_entropy[k] = BlockE._template_stats(dict(raw[k]))
        return path_zipf, path_entropy

    @staticmethod
    def _sample_tree_depth2_templates(
        out_edges: dict[int, list[tuple[int, Any]]],
        start_verts: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> dict[tuple[tuple[Any, Any], ...], int]:
        """Sample depth-2 rooted trees; template = sorted tuple of (r1, r2) pairs.

        Caps at 10k samples and at most 10 children / 5 grandchildren per root
        so hub nodes (degree 900+) don't dominate runtime.
        """
        n_samples = min(n_samples, 10_000)
        roots = start_verts[rng.integers(len(start_verts), size=n_samples)]
        counts: defaultdict[tuple, int] = defaultdict(int)
        for root in roots:
            adj1 = out_edges.get(int(root))
            if not adj1:
                continue
            pairs: list[tuple] = []
            for child, r1 in adj1[:10]:  # at most 10 children
                adj2 = out_edges.get(child)
                if adj2:
                    for _, r2 in adj2[:5]:  # at most 5 grandchildren per child
                        pairs.append((r1, r2))
            if pairs:
                counts[tuple(sorted(pairs))] += 1
        return dict(counts)

    @staticmethod
    def _cc_run_multi(
        g_und: igraph.Graph,
        k: int,
        n_samples: int,
        n_colorings: int,
        rng: np.random.Generator,
    ) -> dict[tuple[int, ...], int]:
        """Run _cc_run n_colorings times and return the averaged estimates.

        Each coloring is an independent unbiased estimator; averaging reduces
        variance by √n_colorings without changing the DP cost (which dominates).
        The sampling cost O(n_samples × k × avg_deg) is negligible compared to
        the DP build O(m × 2^k × k), so using more samples per coloring is free.
        """
        totals: dict[tuple[int, ...], float] = defaultdict(float)
        for _ in range(n_colorings):
            for deg_seq, cnt in BlockE._cc_run(g_und, k, n_samples, rng).items():
                totals[deg_seq] += cnt
        return {ds: max(0, int(round(v / n_colorings))) for ds, v in totals.items()}

    @staticmethod
    def _cc_run(
        g_und: igraph.Graph,
        k: int,
        n_samples: int,
        rng: np.random.Generator,
        *,
        _A: "scipy.sparse.csr_matrix | None" = None,
        _adj: "list[np.ndarray] | None" = None,
    ) -> dict[tuple[int, ...], int]:
        """Color coding estimator for k-node graphlet counts (Bressan et al. 2021).

        Randomly assigns k colors, builds a directed path-treelet DP via sparse
        matrix products, samples n_samples colorful k-paths by backtracking, and
        returns {degree_sequence_tuple: estimated_count}.

        Pass pre-built _A (csr_matrix) and _adj (neighbour lists) to avoid
        rebuilding them for every k — they are identical across all CC calls.

        The σ_H correction (number of directed P_k paths spanning motif H) and
        the p_k = k!/k^k colorfulness probability are applied so the returned
        counts estimate the true graphlet frequencies.
        """
        # σ_H: number of directed spanning P_k paths for each graphlet type.
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
        p_k = math.factorial(k) / (k ** k)

        n = g_und.vcount()
        if n < k:
            return {}

        # Assign random colors in {0, ..., k-1} to each node.
        colors = rng.integers(0, k, size=n, dtype=np.int32)

        # Use cached structures if provided; otherwise build them now.
        n_sets   = 1 << k
        full_set = n_sets - 1
        A = _A if _A is not None else scipy.sparse.csr_matrix(
            g_und.get_adjacency_sparse()
        ).astype(np.float32)

        # dp[v, S] = number of colorful paths of length |S| ending at v with color set S.
        dp = np.zeros((n, n_sets), dtype=np.float32)
        dp[np.arange(n), 1 << colors] = 1.0  # vectorised init — avoids O(n) Python loop
        dp_levels = [dp]

        for step in range(1, k):
            dp_next = np.zeros((n, n_sets), dtype=np.float32)
            for c in range(k):
                mc = 1 << c
                # Source sets: size exactly `step`, not containing color c.
                S_src = np.array(
                    [S for S in range(n_sets)
                     if not (S & mc) and bin(S).count('1') == step],
                    dtype=np.int32,
                )
                if len(S_src) == 0:
                    continue
                S_dst = S_src | mc
                # Only nodes with color c can be the new endpoint.
                node_mask = (colors == c).astype(np.float32)[:, None]
                dp_next[:, S_dst] += (A @ dp_levels[-1][:, S_src]) * node_mask
            dp_levels.append(dp_next)

        # Total colorful k-paths (used to scale estimates).
        t = float(dp_levels[-1][:, full_set].sum())
        if t == 0:
            return {}

        # Pre-build neighbor lists for fast sampling.
        adj = _adj if _adj is not None else [
            np.array(g_und.neighbors(v), dtype=np.int32) for v in range(n)
        ]

        # Sampling weights for final level.
        wfinal = dp_levels[-1][:, full_set].astype(np.float64)
        wfinal /= wfinal.sum()

        # Pre-sample all leaf nodes at once — rng.choice(n, p=…) is O(n) per call,
        # so calling it n_samples times in a loop costs O(n × n_samples).
        v_starts = rng.choice(n, size=n_samples, p=wfinal)

        raw_counts: defaultdict[tuple[int, ...], int] = defaultdict(int)
        n_valid = 0

        for _i in range(n_samples):
            v = int(v_starts[_i])
            nodes = [v]
            S = full_set
            ok = True
            # Backtrack through the DP levels to reconstruct a colorful path.
            for level in range(k - 1, 0, -1):
                S_prev = S ^ (1 << int(colors[v]))
                nbrs = adj[v]
                if len(nbrs) == 0:
                    ok = False
                    break
                nw = dp_levels[level - 1][nbrs, S_prev].astype(np.float64)
                tot = nw.sum()
                if tot == 0:
                    ok = False
                    break
                v = int(nbrs[rng.choice(len(nbrs), p=nw / tot)])
                nodes.append(v)
                S = S_prev
            if not ok:
                continue
            node_set = set(nodes)
            if len(node_set) != k:
                # Duplicate nodes → not a valid k-node induced subgraph.
                continue
            n_valid += 1
            # Classify by sorted internal degree sequence of the induced subgraph.
            deg_in = tuple(sorted(
                sum(1 for nb in adj[v] if nb in node_set) for v in node_set
            ))
            raw_counts[deg_in] += 1

        if n_valid == 0:
            return {}

        # Convert raw sample proportions to estimated graphlet counts.
        result: dict[tuple[int, ...], int] = {}
        for deg_seq, cnt in raw_counts.items():
            sigma = _SIGMA.get(deg_seq, 1)
            estimated = (cnt / n_valid) * t / sigma / p_k
            result[deg_seq] = max(0, int(round(estimated)))
        return result

    @staticmethod
    def _count_4node_motifs(
        g_und: igraph.Graph,
        triangles: list,
    ) -> tuple[int, int, int, int]:
        """Count (four_cycle, diamond, k4, tailed_triangle).

        K4 counts use triangle-set-intersection (O(T·d)), not cliques().
        cliques(min=4,max=4) runs Bron-Kerbosch over all vertices even when
        the graph is very sparse, costing 40+ s on 100k-node samples.

        k4      — triangle loop: d ∈ N(a)∩N(b)∩N(c), raw÷4
        diamond — from A²: edge_C − 6·k4
                  where edge_C = Σ_{edges (u,v)} C(A²[u,v], 2)
        C4      — from A²: (total_C − edge_C − diamond) / 2
                  where total_C = Σ_{i<j} C(A²[i,j], 2)
        tailed  — for each triangle vertex, count nodes adjacent only to
                  that vertex (exclusive-neighbour set subtraction)

        Identities:
            total_C = 2·C4 + 2·diamond + 6·K4
            edge_C  =        diamond   + 6·K4
            → diamond = edge_C − 6·K4
            → C4     = (total_C − edge_C − diamond) / 2
        """
        n = g_und.vcount()

        # K4 via triangle intersection: for each triangle find the 4th node
        # adjacent to all three vertices.  Each K4 spans 4 triangles → ÷4.
        adj = [set(g_und.neighbors(v)) for v in range(n)]
        k4_raw = 0
        for a, b, c in triangles:
            abc  = {a, b, c}
            all3 = (adj[a] & adj[b] & adj[c]) - abc
            k4_raw += len(all3)
        k4 = k4_raw // 4

        # A²: (A²)_{ij} = |N(i) ∩ N(j)|; diagonal = degree
        A  = scipy.sparse.csr_matrix(g_und.get_adjacency_sparse())
        A2 = (A @ A).astype(np.float64)

        degrees   = np.array(g_und.degree(), dtype=np.float64)
        edges_arr = np.array(g_und.get_edgelist())
        if edges_arr.shape[0] > 0:
            c_e    = np.asarray(A2[edges_arr[:, 0], edges_arr[:, 1]]).flatten()
            edge_C = float(np.sum(c_e * (c_e - 1)) / 2.0)
        else:
            edge_C = 0.0

        sum_sq  = float(A2.multiply(A2).sum()) - float(np.sum(degrees ** 2))
        sum_a2  = float(np.sum(degrees * (degrees - 1)))
        total_C = (sum_sq - sum_a2) / 4.0

        diamond    = max(0, int(round(edge_C - 6.0 * k4)))
        four_cycle = max(0, int(round((total_C - edge_C - diamond) / 2.0)))

        # Tailed triangle: for each triangle vertex, count nodes adjacent to
        # that vertex only (not the other two).  Each such node contributes one
        # tailed-triangle motif, counted exactly once.
        tailed = 0
        for a, b, c in triangles:
            abc = {a, b, c}
            tailed += len(adj[a] - adj[b] - adj[c] - abc)
            tailed += len(adj[b] - adj[a] - adj[c] - abc)
            tailed += len(adj[c] - adj[a] - adj[b] - abc)

        return four_cycle, diamond, k4, tailed

    @staticmethod
    def _cc_run_stars(
        g_und: igraph.Graph,
        n_samples: int,
        rng: np.random.Generator,
        *,
        _A: "scipy.sparse.csr_matrix | None" = None,
        _adj: "list[np.ndarray] | None" = None,
    ) -> dict[int, int]:
        """Color coding estimator for induced k-star counts, k=2..10.

        A k-star = one center connected to k leaves with NO edges between leaves.

        Star treelet DP (one run per k):
          1. Assign k+1 random colors.
          2. color_hist[v,c] = # neighbours of v with color c  (sparse mat-mul).
          3. dp_star[v] = Π_{c ≠ color[v]} color_hist[v,c]
             (# colourful star choices centred at v).
          4. Sample n_samples centres ∝ dp_star; for each centre pick one
             neighbour per non-own colour.
          5. Classify the induced subgraph: accept as a k-star only if all
             leaves have internal degree 1 (no leaf-leaf edges).
          6. Estimate: ĝ = (raw_star/n_valid) × t / σ / p_{k+1}
             where σ=1 (only the centre can root the star spanning tree)
             and p_{k+1} = (k+1)!/(k+1)^{k+1}.
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
            K   = k + 1                              # total nodes (centre + k leaves)
            p_K = math.factorial(K) / (K ** K)

            colors = rng.integers(0, K, size=n, dtype=np.int32)

            # color_hist[v, c] = # neighbours of v with color c via sparse mul.
            one_hot            = np.zeros((n, K), dtype=np.float32)
            one_hot[np.arange(n), colors] = 1.0
            color_hist         = (A_csr @ one_hot).astype(np.float64)  # (n, K)

            # dp_star[v] = Π_{c ≠ colors[v]} color_hist[v, c]
            # Vectorised: for each colour c, multiply into dp_star only for
            # nodes whose own colour is NOT c.
            dp_star = np.ones(n, dtype=np.float64)
            for c in range(K):
                mask = (colors != c)
                dp_star[mask] *= color_hist[mask, c]

            t = float(dp_star.sum())
            if t == 0:
                results[k] = 0
                continue

            # Pre-sample all centres at once.
            w       = dp_star / t
            centres = rng.choice(n, size=n_samples, p=w)

            # Precompute neighbours-by-colour for every unique centre
            # (avoids repeating the O(deg) filter for high-degree hubs).
            unique_centres = np.unique(centres)
            adj_by_color: dict[int, dict[int, np.ndarray]] = {}
            for v in unique_centres:
                v = int(v)
                nb = adj[v]
                adj_by_color[v] = {
                    c: nb[colors[nb] == c] for c in range(K)
                } if len(nb) > 0 else {c: np.array([], dtype=np.int32) for c in range(K)}

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

                # Accept as induced k-star only if no leaf is connected to another leaf.
                leaf_set = set(leaf_nodes)
                leaves   = leaf_nodes[1:]
                deg_in   = sorted(
                    int(np.sum(np.isin(adj[u], list(leaf_set))))
                    for u in leaf_nodes
                )
                if deg_in == [1] * k + [k]:
                    raw_star += 1

            if n_valid == 0:
                results[k] = 0
            else:
                # σ = 1: only the centre can root the star spanning treelet.
                results[k] = max(0, int(round((raw_star / n_valid) * t / p_K)))

        return results

    @staticmethod
    def _estimate_k_cycle(
        g_und: igraph.Graph,
        k: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> int:
        """Estimate k-cycle count via random walk closure sampling.

        Hub nodes (degree > _HUB_THRESH) are handled with rejection sampling
        and pre-built sets, avoiding O(d) scans for d = 225k-degree nodes.
        """
        _HUB_THRESH  = 200   # degree above which full scan would be too slow
        _MAX_REJECT  = 50    # rejection attempts at hubs (prob failure < 1e-40)

        n = g_und.vcount()
        if n < k:
            return 0

        adj     = [g_und.neighbors(v) for v in range(n)]
        avg_deg = float(sum(len(a) for a in adj)) / n if n > 0 else 0.0
        # Pre-build sets only for the rare high-degree hubs (O(m) total)
        hub_sets = {v: set(nb) for v, nb in enumerate(adj) if len(nb) > _HUB_THRESH}

        n_closed = 0
        n_valid  = 0
        for _ in range(n_samples):
            start = int(rng.integers(n))
            if not adj[start]:
                continue
            v       = start
            visited = {start}
            ok      = True
            for _ in range(k - 1):
                nb_v = adj[v]
                dv   = len(nb_v)
                if dv <= _HUB_THRESH:
                    cands = [u for u in nb_v if u not in visited]
                    if not cands:
                        ok = False
                        break
                    v = cands[int(rng.integers(len(cands)))]
                else:
                    # Rejection sampling: pick random neighbor, retry if visited
                    found = False
                    for _ in range(_MAX_REJECT):
                        u = nb_v[int(rng.integers(dv))]
                        if u not in visited:
                            v = u
                            found = True
                            break
                    if not found:
                        ok = False
                        break
                visited.add(v)
            n_valid += 1
            if ok and len(visited) == k:
                nb_v = adj[v]
                closed = (start in hub_sets[v]) if v in hub_sets else (start in nb_v)
                if closed:
                    n_closed += 1

        if n_valid == 0 or n_closed == 0:
            return 0
        return int((n_closed / n_valid) * n * (avg_deg ** (k - 1)) / (2 * k))
