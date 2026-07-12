"""Block E — Motif shape distribution (G5).

Keeps the raw motif counts (triangle, 4-/5-/6-cycle, diamond, k4, tailed
triangle) and the path/tree template zipf+entropy curves. Exact counts for 3-
and 4-node motifs are computed on the undirected simplification; 5/6-node and
path/tree templates come from the color-coding (CC) counter.

The induced ``star_count_k*`` features are no longer part of the signature and
are no longer measured here. The counter's ``count_stars`` helper stays in
``motif_counter`` regardless — it is exercised directly by
``tests/test_generator_motif_counter.py``, ``tests/test_hybrid_motif_counter.py``,
and ``scripts/cc_variance.py``, independent of whether Block E calls it.

Older corpus ``block_e.json`` files measured before this removal may still
carry a stale ``_star_counts`` key; that key is not read by anything and
disappears the next time the graph is re-measured.
"""

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from ..motif_counter import HybridMotifCounter, MotifCounter

from .._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED

log = get_logger(__name__)

_SAMPLE_BUDGET = 100_000  # default walk samples for path/tree templates
_MAX_K         = 10       # longest path template walk

# Counter used for all motif measurement in BlockE.calculate().
# HybridMotifCounter: exact for triangles and k≤3; CC sampling for k≥4. The 3-/4-node
# counts below are therefore exact only for triangles — the rest are estimates at
# every graph size (see tests/test_signature_block_e_vs_library.py).
MOTIF_COUNTER: MotifCounter = HybridMotifCounter(n_samples=_SAMPLE_BUDGET, seed=1)


class BlockE(SignatureBlock):
    """Reduced Block E — motif shape distribution.

    Usage::

        b = BlockE().calculate(g)
        b.as_vector()                # fixed-length comparison vector
        b.as_dict()                  # named key-value pairs
        b.visualize(mode="text")     # CLI summary
        b.visualize(path="out.png")  # save plot to file
    """

    def __init__(self) -> None:
        self._triangle_count = _NOT_CALCULATED
        self._four_cycle_count = _NOT_CALCULATED
        self._five_cycle_count = _NOT_CALCULATED
        self._six_cycle_count = _NOT_CALCULATED
        self._diamond_count = _NOT_CALCULATED
        self._k4_count = _NOT_CALCULATED
        self._tailed_triangle_count = _NOT_CALCULATED
        self._path_template_zipf = _NOT_CALCULATED
        self._path_template_entropy = _NOT_CALCULATED
        self._tree_template_zipf = _NOT_CALCULATED
        self._tree_template_entropy = _NOT_CALCULATED
    # ── properties ────────────────────────────────────────────────────────────

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

    # ── core ──────────────────────────────────────────────────────────────────

    def calculate(
        self,
        g: igraph.Graph,
        sample_budget: int = _SAMPLE_BUDGET,
        skip_stars_and_paths: bool = False,
    ) -> "BlockE":
        """Compute reduced Block E (motif distribution).

        Triangles are counted exactly on the undirected simplification. The 4-node
        motifs, 5/6-cycles and path/tree templates all come from the color-coding
        counter and are estimates; the diamond estimate in particular is biased high
        (see ``KNOWN_CC_DIAMOND_BIAS`` in tests/test_hybrid_motif_counter.py).

        Parameters
        ----------
        skip_stars_and_paths : bool
            When True, skip the path-template and tree-template computations.
            Triangle/4-node motif and 5/6-cycle counts are always measured.
            (Induced star counts are no longer measured regardless of this flag.)
        """
        g_und = g.as_undirected(combine_edges="first").simplify()
        n = g_und.vcount()
        m = g_und.ecount()
        log.info("Block E: graph has %d nodes, %d edges", n, m)

        if n == 0:
            self._triangle_count = 0
            self._four_cycle_count = self._diamond_count = 0
            self._k4_count = self._tailed_triangle_count = 0
            self._five_cycle_count = self._six_cycle_count = 0
            self._path_template_zipf = {}
            self._path_template_entropy = {}
            self._tree_template_zipf = float("nan")
            self._tree_template_entropy = float("nan")
            return self

        self._triangle_count = MOTIF_COUNTER.count_triangles(g_und)
        log.info("Block E: computed triangle_count (%d)", self._triangle_count)

        motifs3 = MOTIF_COUNTER.count_motifs3(g_und)

        motifs4 = MOTIF_COUNTER.count_motifs4(g_und)
        self._four_cycle_count      = motifs4.get((2, 2, 2, 2), 0)
        self._diamond_count         = motifs4.get((2, 2, 3, 3), 0)
        self._k4_count              = motifs4.get((3, 3, 3, 3), 0)
        self._tailed_triangle_count = motifs4.get((1, 2, 2, 3), 0)
        log.info("Block E: computed four_cycle_count (%d)", self._four_cycle_count)
        log.info("Block E: computed diamond_count (%d)", self._diamond_count)
        log.info("Block E: computed k4_count (%d)", self._k4_count)
        log.info("Block E: computed tailed_triangle_count (%d)", self._tailed_triangle_count)

        # 5/6-cycle counts are core motif features, so they are always measured via
        # the hybrid counter (CC-sampled for k>=4, see HybridMotifCounter).
        # motifs5/motifs6 are reused by the path-template block when paths are not skipped.
        motifs5 = MOTIF_COUNTER.count_motifsk(g_und, 5)
        self._five_cycle_count = motifs5.get((2, 2, 2, 2, 2), 0)
        log.info("Block E: computed five_cycle_count (~%d)", self._five_cycle_count)

        motifs6 = MOTIF_COUNTER.count_motifsk(g_und, 6)
        self._six_cycle_count = motifs6.get((2, 2, 2, 2, 2, 2), 0)
        log.info("Block E: computed six_cycle_count (~%d)", self._six_cycle_count)

        if skip_stars_and_paths:
            log.info("Block E: skipping path templates, tree templates.")
            self._path_template_zipf = {}
            self._path_template_entropy = {}
            self._tree_template_zipf = float("nan")
            self._tree_template_entropy = float("nan")
        else:
            # Path templates: for each k, compute Zipf + entropy of the graphlet-type
            # distribution at that size.  k=2..6 always run.  k=7..10 are run only when
            # the DP fits in ~1 GB: n × 2^k × k × 4 bytes ≤ 1 GB → n ≤ 1e9 / (k×2^k×4).
            motifs2 = MOTIF_COUNTER.count_motifsk(g_und, 2)
            _cc_by_k = {2: motifs2, 3: motifs3, 4: motifs4, 5: motifs5, 6: motifs6}
            for _k in range(7, _MAX_K + 1):
                _dp_bytes = n * (1 << _k) * _k * 4
                if _dp_bytes <= 1_000_000_000:
                    _cc_by_k[_k] = MOTIF_COUNTER.count_motifsk(g_und, _k)
                else:
                    log.info(
                        "Block E: skipping path template k=%d (DP would need %.1f GB > 1 GB limit)",
                        _k, _dp_bytes / 1e9,
                    )

            self._path_template_zipf    = {}
            self._path_template_entropy = {}
            for _k in range(2, _MAX_K + 1):
                if _k in _cc_by_k:
                    _z, _e = self._template_stats(_cc_by_k[_k])
                else:
                    _z, _e = float("nan"), float("nan")
                self._path_template_zipf[_k]    = _z
                self._path_template_entropy[_k] = _e
            log.info(
                "Block E: computed path_template_zipf (k=2..10 alphas=%s)",
                [
                    round(self._path_template_zipf.get(k, float("nan")), 4)
                    for k in range(2, _MAX_K + 1)
                ],
            )
            log.info(
                "Block E: computed path_template_entropy (k=2..10 entropies=%s)",
                [
                    round(self._path_template_entropy.get(k, float("nan")), 4)
                    for k in range(2, _MAX_K + 1)
                ],
            )

            # Tree templates: Zipf + entropy of how total motif counts scale across k.
            # Using total-count-per-k (rather than per-type) avoids NaN on small graphs
            # where CC returns only 1-2 distinct graphlet types.
            _totals_by_k: dict[int, int] = {
                _k: sum(_cc_by_k[_k].values())
                for _k in _cc_by_k
                if sum(_cc_by_k[_k].values()) > 0
            }
            self._tree_template_zipf, self._tree_template_entropy = self._template_stats(
                _totals_by_k
            )
            log.info(
                "Block E: computed tree_template stats (zipf_alpha=%.4f, entropy=%.4f)",
                self._tree_template_zipf, self._tree_template_entropy,
            )
        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 27-vector for cross-KG comparison.

        Layout: 7 motif counts; path-template zipf (k=2..10); path-template
        entropy (k=2..10); tree-template (zipf, entropy).

        Attributes absent from stale serialized data are emitted as NaN so
        that sweep_viz.py can still analyse whatever features are present.
        """
        _nan = float("nan")

        def _dict(v) -> dict:
            return {} if v is _NOT_CALCULATED else v

        vec = [
            self._safe_scalar(lambda: self.triangle_count),
            self._safe_scalar(lambda: self.four_cycle_count),
            self._safe_scalar(lambda: self.five_cycle_count),
            self._safe_scalar(lambda: self.six_cycle_count),
            self._safe_scalar(lambda: self.diamond_count),
            self._safe_scalar(lambda: self.k4_count),
            self._safe_scalar(lambda: self.tailed_triangle_count),
        ]
        pzipf = _dict(self._path_template_zipf)
        for k in range(2, _MAX_K + 1):
            vec.append(pzipf.get(k, _nan))
        pent = _dict(self._path_template_entropy)
        for k in range(2, _MAX_K + 1):
            vec.append(pent.get(k, _nan))
        vec.extend([
            self._safe_scalar(lambda: self.tree_template_zipf),
            self._safe_scalar(lambda: self.tree_template_entropy),
        ])
        return vec  # length 7 + 9 + 9 + 2 = 27

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        names = [
            "triangle_count", "four_cycle_count", "five_cycle_count",
            "six_cycle_count", "diamond_count", "k4_count", "tailed_triangle_count",
        ]
        names += [f"path_template_zipf_k{k}" for k in range(2, _MAX_K + 1)]
        names += [f"path_template_entropy_k{k}" for k in range(2, _MAX_K + 1)]
        names += ["tree_template_zipf", "tree_template_entropy"]
        return names

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 27-element NaN vector (same length as as_vector())."""
        return [float("nan")] * 27

    @classmethod
    def _state_from_features(cls, feats: dict[str, float]) -> dict:
        """Rebuild Block E's state from the flat feature dict.

        The per-k path/tree template features are re-keyed into the ``{k: value}``
        dicts the block stores. NaN entries (templates never measured, e.g. when
        ``skip_stars_and_paths`` was set) are dropped rather than stored as NaN,
        reproducing the empty-dict state ``calculate`` leaves in that case.
        """
        ks = range(2, _MAX_K + 1)

        def _by_k(prefix: str) -> dict[int, float]:
            out = {k: feats[f"{prefix}_k{k}"] for k in ks}
            return {k: v for k, v in out.items() if v == v}  # drop NaN

        state = {f"_{name}": cls._int(feats, name) for name in (
            "triangle_count", "four_cycle_count", "five_cycle_count", "six_cycle_count",
            "diamond_count", "k4_count", "tailed_triangle_count",
        )}
        state["_path_template_zipf"] = _by_k("path_template_zipf")
        state["_path_template_entropy"] = _by_k("path_template_entropy")
        state["_tree_template_zipf"] = feats["tree_template_zipf"]
        state["_tree_template_entropy"] = feats["tree_template_entropy"]
        return state

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for reduced Block E.

        Args:
            mode: "plot" for a matplotlib figure, "text" for a CLI summary.
            path: write to this file instead of displaying interactively.
        """
        if mode == "text":
            self._visualize_text(path)
        elif mode == "plot":
            self._visualize_plot(path)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'plot' or 'text'.")

    # ── private helpers ───────────────────────────────────────────────────────

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
            zipf  = (
                float(np.clip(1.0 + tail.size / denom, 1.01, 50.0)) if denom > 0 else float("nan")
            )
        p = freqs / freqs.sum()
        p = p[p > 0]
        entropy = -float(np.sum(p * np.log(p)))
        return zipf, entropy

    def _visualize_text(self, path: str | None) -> None:
        lines: list[str] = []
        lines.append("=== Reduced Block E: Motif Shape Distribution (G5) ===\n")

        lines.append("--- Subgraph counts ---")
        lines.append(f"  triangles:           {self.triangle_count}")
        lines.append(f"  4-cycles:            {self.four_cycle_count}")
        lines.append(f"  5-cycles (est):      {self.five_cycle_count}")
        lines.append(f"  6-cycles (est):      {self.six_cycle_count}")
        lines.append(f"  diamonds:            {self.diamond_count}")
        lines.append(f"  K4:                  {self.k4_count}")
        lines.append(f"  tailed triangles:    {self.tailed_triangle_count}")

        lines.append("\n--- Path templates ---")
        lines.append(f"  {'k':>3}  {'zipf_alpha':>10}  {'entropy':>10}")
        for k in range(2, _MAX_K + 1):
            z = self.path_template_zipf.get(k, float("nan"))
            e = self.path_template_entropy.get(k, float("nan"))
            lines.append(f"  {k:>3}  {z:>10.4f}  {e:>10.4f}")

        lines.append("\n--- Tree templates ---")
        lines.append(
            f"  zipf_alpha={self.tree_template_zipf:.4f}  "
            f"entropy={self.tree_template_entropy:.4f}"
        )

        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            ks = list(range(2, _MAX_K + 1))

            # Motif / cycle counts
            ax = axes[0, 0]
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

            # Path template zipf alpha and entropy vs k
            ax = axes[0, 1]
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

            axes[1, 0].axis("off")  # star-count panel removed; spare cell

            # Tree-template summary scalars (zipf exponent, entropy)
            ax = axes[1, 1]
            ax.bar(["zipf alpha", "entropy"],
                   [self.tree_template_zipf, self.tree_template_entropy],
                   color=["steelblue", "darkorange"])
            ax.set_ylabel("value")
            ax.set_title("Tree template statistics")

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block E: plot failed: %s", exc, exc_info=True)
            plt.close("all")
