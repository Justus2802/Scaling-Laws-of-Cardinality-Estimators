"""Block E — Motif shape distribution features."""

import functools
import math
from collections import defaultdict
from typing import Any

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.sparse

from ._logging import get_logger
from ._utils import _fit_powerlaw

log = get_logger(__name__)

_SAMPLE_BUDGET = 100_000  # default walk samples for path/tree templates

_NOT_CALCULATED = object()


@functools.lru_cache(maxsize=1)
def _4node_motif_index_map() -> dict[str, int]:
    """Discover which index in motifs_randesu(size=4) maps to each named 4-node motif.

    Creates a canonical 4-node example for each motif, runs full RANDESU enumeration,
    and finds the unique index with count == 1.  Cached so it runs only once per process.
    """
    specs: dict[str, igraph.Graph] = {
        "four_cycle":      igraph.Graph(n=4, edges=[(0,1),(1,2),(2,3),(3,0)]),
        "diamond":         igraph.Graph(n=4, edges=[(0,1),(0,2),(0,3),(1,2),(1,3)]),
        "k4":              igraph.Graph(n=4, edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]),
        "tailed_triangle": igraph.Graph(n=4, edges=[(0,1),(1,2),(0,2),(2,3)]),
    }
    index_map: dict[str, int] = {}
    for name, pattern in specs.items():
        for i, count in enumerate(pattern.motifs_randesu(size=4)):
            if not math.isnan(count) and int(count) == 1:
                index_map[name] = i
                break
    return index_map


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
        # --- Exact motif counts on undirected simple graph ---
        g_und = g.as_undirected(combine_edges="first").simplify()

        # Triangles: A ⊙ A² summed and divided by 6 (each triangle counted 6 times)
        if g_und.vcount() > 0:
            A = scipy.sparse.csr_matrix(g_und.get_adjacency_sparse())
            self._triangle_count = int(A.multiply(A @ A).sum() // 6)
        else:
            self._triangle_count = 0

        # 4-node motifs: full RANDESU enumeration with runtime index discovery
        idx_map = _4node_motif_index_map()
        four_motifs = g_und.motifs_randesu(size=4)

        def _get_motif(name: str) -> int:
            i = idx_map.get(name)
            if i is None or i >= len(four_motifs):
                return 0
            v = four_motifs[i]
            return 0 if math.isnan(v) else int(v)

        self._four_cycle_count = _get_motif("four_cycle")
        self._diamond_count = _get_motif("diamond")
        self._k4_count = _get_motif("k4")
        self._tailed_triangle_count = _get_motif("tailed_triangle")

        # Stars (exact) and 5/6-cycles (sampled)
        self._star_counts = self._count_stars(g_und)
        motif_rng = np.random.default_rng(0)
        n_cycle = max(1, sample_budget // 2)
        self._five_cycle_count = self._estimate_k_cycle(g_und, 5, n_cycle, motif_rng)
        self._six_cycle_count = self._estimate_k_cycle(g_und, 6, n_cycle, motif_rng)

        # --- Path and tree templates from directed graph ---
        out_edges, start_verts_list = self._build_out_adj(g)
        start_verts = np.array(start_verts_list)

        path_template_zipf: dict[int, float] = {}
        path_template_entropy: dict[int, float] = {}
        tree_zipf = float("nan")
        tree_ent = float("nan")

        if start_verts.size > 0:
            rng = np.random.default_rng(1)
            n_per_k = max(1, sample_budget // 9)  # spread budget evenly across k=2..10
            for k in range(2, 11):
                counts = self._sample_path_templates(out_edges, start_verts, k, n_per_k, rng)
                path_template_zipf[k], path_template_entropy[k] = self._template_stats(counts)

            tree_counts = self._sample_tree_depth2_templates(
                out_edges, start_verts, sample_budget, rng
            )
            tree_zipf, tree_ent = self._template_stats(tree_counts)

        self._path_template_zipf = path_template_zipf
        self._path_template_entropy = path_template_entropy
        self._tree_template_zipf = tree_zipf
        self._tree_template_entropy = tree_ent

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
        """Return (Zipf exponent, Shannon entropy) from a {template: count} dict."""
        if not counts:
            return float("nan"), float("nan")
        freqs = np.array(list(counts.values()), dtype=float)
        zipf = _fit_powerlaw(freqs).alpha
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
    def _sample_path_templates(
        out_edges: dict[int, list[tuple[int, str]]],
        start_verts: np.ndarray,
        k: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> dict[tuple[str, ...], int]:
        """Sample n_samples directed walks of length k; count relation-sequence tuples."""
        counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
        for _ in range(n_samples):
            v = int(rng.choice(start_verts))
            rels: list[str] = []
            for _ in range(k):
                adj = out_edges.get(v)
                if not adj:
                    break
                nb, rel = adj[int(rng.integers(len(adj)))]
                rels.append(rel)
                v = nb
            if len(rels) == k:
                counts[tuple(rels)] += 1
        return dict(counts)

    @staticmethod
    def _sample_tree_depth2_templates(
        out_edges: dict[int, list[tuple[int, str]]],
        start_verts: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> dict[tuple[tuple[str, str], ...], int]:
        """Sample depth-2 rooted trees; template = sorted tuple of (r1, r2) pairs."""
        counts: defaultdict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        for _ in range(n_samples):
            root = int(rng.choice(start_verts))
            adj1 = out_edges.get(root)
            if not adj1:
                continue
            pairs: list[tuple[str, str]] = []
            for child, r1 in adj1:
                adj2 = out_edges.get(child)
                if adj2:
                    for _, r2 in adj2:
                        pairs.append((r1, r2))
            if pairs:
                counts[tuple(sorted(pairs))] += 1
        return dict(counts)

    @staticmethod
    def _count_stars(g_und: igraph.Graph) -> dict[int, int]:
        """Exact k-star counts for k=2..10 via the degree distribution."""
        from math import comb
        degrees = g_und.degree()
        return {k: sum(comb(d, k) for d in degrees if d >= k) for k in range(2, 11)}

    @staticmethod
    def _estimate_k_cycle(
        g_und: igraph.Graph,
        k: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> int:
        """Estimate k-cycle count via random walk closure sampling."""
        n = g_und.vcount()
        if n < k:
            return 0
        adj = [list(g_und.neighbors(v)) for v in range(n)]
        degrees = np.array([len(a) for a in adj], dtype=float)
        avg_deg = float(degrees.mean()) if n > 0 else 0.0

        n_closed = 0
        n_valid = 0
        for _ in range(n_samples):
            start = int(rng.integers(n))
            if not adj[start]:
                continue
            v = start
            visited = {start}
            ok = True
            for _ in range(k - 1):
                candidates = [nb for nb in adj[v] if nb not in visited]
                if not candidates:
                    ok = False
                    break
                nb = candidates[int(rng.integers(len(candidates)))]
                visited.add(nb)
                v = nb
            n_valid += 1
            if ok and len(visited) == k and start in adj[v]:
                n_closed += 1

        if n_valid == 0 or n_closed == 0:
            return 0
        return int((n_closed / n_valid) * n * (avg_deg ** (k - 1)) / (2 * k))
