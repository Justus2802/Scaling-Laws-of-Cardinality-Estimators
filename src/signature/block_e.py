"""Block E — Motif shape distribution features."""

import math
from collections import defaultdict
from typing import Any

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.sparse

from ._logging import get_logger

log = get_logger(__name__)

_SAMPLE_BUDGET = 100_000  # default walk samples for path/tree templates
_MAX_K         = 10       # longest path template walk
_LARGE_N       = 50_000   # above this, sample an induced subgraph for structural counts
_SAMPLE_N      = 100_000  # induced-subgraph size when n > _LARGE_N

_NOT_CALCULATED = object()


class BlockE:
    """Block E — Motif shape distribution of a KG.

    Exact counts for 3- and 4-node motifs on the undirected simplification.
    Path and tree templates are estimated by random walk sampling.

    Usage::

        b = BlockE().calculate(g)
        b.as_vector()                      # fixed-length comparison vector
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

    def _require(self, name: str, value: object) -> Any:
        if value is _NOT_CALCULATED:
            raise RuntimeError(f"Call calculate() before accessing {name}")
        return value

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
        """
        g_und = g.as_undirected(combine_edges="first").simplify()
        n = g_und.vcount()

        # For large graphs, work on a random node-induced subgraph so that
        # list_triangles() O(m√m) and cliques() remain tractable.
        # Counts on the sample are rescaled by (n/s)^k for k-node motifs.
        if n > _LARGE_N:
            _rng_s = np.random.default_rng(42)
            _ids   = _rng_s.choice(n, size=min(n, _SAMPLE_N), replace=False)
            g_motif = g_und.induced_subgraph(_ids)
            _scale  = float(n) / len(_ids)
            log.info(
                "Block E: large graph (%d nodes) — sampling %d for structural counts"
                " (scale=%.2f)",
                n, len(_ids), _scale,
            )
        else:
            g_motif = g_und
            _scale  = 1.0

        nm = g_motif.vcount()

        # Triangles: list_triangles() enumerates each once (O(m√m) on g_motif).
        _tris = g_motif.list_triangles() if nm >= 3 else []
        self._triangle_count = int(round(len(_tris) * _scale ** 3))
        log.info("Block E: computed triangle_count (%d)", self._triangle_count)

        # 4-node motifs on the (possibly sampled) subgraph, then scale by s^4.
        if nm >= 4:
            _c4, _dia, _k4, _tail = self._count_4node_motifs(g_motif, _tris)
            s4 = _scale ** 4
            self._four_cycle_count       = int(round(_c4  * s4))
            self._diamond_count          = int(round(_dia  * s4))
            self._k4_count               = int(round(_k4   * s4))
            self._tailed_triangle_count  = int(round(_tail * s4))
        else:
            self._four_cycle_count = self._diamond_count = 0
            self._k4_count = self._tailed_triangle_count = 0
        log.info("Block E: computed four_cycle_count (%d)", self._four_cycle_count)
        log.info("Block E: computed diamond_count (%d)", self._diamond_count)
        log.info("Block E: computed k4_count (%d)", self._k4_count)
        log.info("Block E: computed tailed_triangle_count (%d)", self._tailed_triangle_count)

        # Stars: exact on g_motif, each k-star (k+1 nodes) scaled by s^(k+1).
        _raw_stars = self._count_stars(g_motif)
        if _scale > 1.0:
            self._star_counts = {
                k: int(round(v * _scale ** (k + 1))) for k, v in _raw_stars.items()
            }
        else:
            self._star_counts = _raw_stars
        log.info(
            "Block E: computed star_counts (k=2..10 totals=%s)",
            [self._star_counts.get(k, 0) for k in range(2, 11)],
        )
        motif_rng = np.random.default_rng(0)
        n_cycle = max(1, sample_budget // 10)
        self._five_cycle_count = self._estimate_k_cycle(g_und, 5, n_cycle, motif_rng)
        log.info("Block E: computed five_cycle_count (~%d, sampled)", self._five_cycle_count)
        self._six_cycle_count = self._estimate_k_cycle(g_und, 6, n_cycle, motif_rng)
        log.info("Block E: computed six_cycle_count (~%d, sampled)", self._six_cycle_count)

        # Path and tree templates from directed graph.
        # One combined walk pass fills all k=2..10 at once (vs 9 separate passes).
        out_edges, start_verts_list = self._build_out_adj(g)
        start_verts = np.array(start_verts_list)

        if start_verts.size > 0:
            rng = np.random.default_rng(1)
            self._path_template_zipf, self._path_template_entropy = (
                self._sample_all_path_templates(out_edges, start_verts, sample_budget, rng)
            )
            log.info(
                "Block E: computed path_template_zipf (k=2..10 alphas=%s)",
                [round(self._path_template_zipf.get(k, float("nan")), 4) for k in range(2, 11)],
            )
            log.info(
                "Block E: computed path_template_entropy (k=2..10 entropies=%s)",
                [round(self._path_template_entropy.get(k, float("nan")), 4) for k in range(2, 11)],
            )
            rng2 = np.random.default_rng(2)
            tree_counts = self._sample_tree_depth2_templates(
                out_edges, start_verts, sample_budget, rng2
            )
            self._tree_template_zipf, self._tree_template_entropy = self._template_stats(tree_counts)
            log.info(
                "Block E: computed tree_template stats (zipf_alpha=%.4f, entropy=%.4f)",
                self._tree_template_zipf, self._tree_template_entropy,
            )
        else:
            self._path_template_zipf = {}
            self._path_template_entropy = {}
            self._tree_template_zipf = float("nan")
            self._tree_template_entropy = float("nan")
            log.info(
                "Block E: computed path/tree template stats (no start vertices, all NaN/empty)"
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

            # Star counts by k
            ax = axes[1]
            ks = list(range(2, 11))
            star_vals = [self.star_counts.get(k, 0) for k in ks]
            ax.bar([str(k) for k in ks], star_vals, color="darkorange")
            ax.set_xlabel("k")
            ax.set_ylabel("count")
            ax.set_title("k-star counts (exact)")

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
    def _count_stars(g_und: igraph.Graph) -> dict[int, int]:
        """Exact k-star counts for k=2..10 via the degree distribution.

        Vectorised: comb(d, k) = prod_{i=0}^{k-1} (d-i)/(i+1) computed with
        numpy over the degree array — avoids a Python-level loop per vertex.
        """
        degrees = np.array(g_und.degree(), dtype=np.float64)
        result: dict[int, int] = {}
        for k in range(2, 11):
            d = degrees[degrees >= k]
            if d.size == 0:
                result[k] = 0
                continue
            vals = np.ones(d.size, dtype=np.float64)
            for i in range(k):
                vals *= (d - i) / (i + 1)
            result[k] = int(float(np.sum(vals)))
        return result

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
