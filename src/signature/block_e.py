"""Reduced Block E — Motif shape distribution (G5).

Keeps the raw motif counts (triangle, 4-/5-/6-cycle, diamond, k4, tailed
triangle), the ``star_count_k*`` features, and the path/tree template
zipf+entropy curves, all computed by the original Block E (composed, not
re-implemented).

The ``star_count_k*`` values are *induced* k-stars (a centre plus k leaves with
no edges among the leaves), as actually measured by the original Block E's
``MOTIF_COUNTER.count_stars`` (exact induced counting). They are **not** the
non-induced ``Σ_v C(deg(v), k)``: induced stars are not determined by the degree
sequence alone (they depend on the local edge structure among neighbours), so
they carry information independent of the Block B degree distribution and are
kept here.
"""

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._orig_block_e import BlockE as _OrigBlockE, _SAMPLE_BUDGET, _MAX_K

log = get_logger(__name__)


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
        self._star_counts = _NOT_CALCULATED
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

    # ── core ──────────────────────────────────────────────────────────────────

    def calculate(
        self,
        g: igraph.Graph,
        sample_budget: int = _SAMPLE_BUDGET,
        skip_stars_and_paths: bool = False,
    ) -> "BlockE":
        """Compute reduced Block E (motif distribution).

        Composes the original Block E to reuse its color-coding motif counts,
        induced ``star_count_k*`` values and path/tree template statistics.

        Parameters
        ----------
        skip_stars_and_paths : bool
            When True, skip the path-template and tree-template computations.
            Triangle/4-node motif, 5/6-cycle and induced star counts are always
            measured (mirroring the original Block E).
        """
        orig = _OrigBlockE().calculate(g, sample_budget=sample_budget,
                                       skip_stars_and_paths=skip_stars_and_paths)
        self._triangle_count        = orig.triangle_count
        self._four_cycle_count      = orig.four_cycle_count
        self._five_cycle_count      = orig.five_cycle_count
        self._six_cycle_count       = orig.six_cycle_count
        self._diamond_count         = orig.diamond_count
        self._k4_count              = orig.k4_count
        self._tailed_triangle_count = orig.tailed_triangle_count
        self._star_counts           = orig.star_counts
        self._path_template_zipf    = orig.path_template_zipf
        self._path_template_entropy = orig.path_template_entropy
        self._tree_template_zipf    = orig.tree_template_zipf
        self._tree_template_entropy = orig.tree_template_entropy

        log.info(
            "Block E: triangles=%d, 4-cyc=%d, diamonds=%d, k4=%d, tailed=%d, "
            "tree(zipf=%.4f, entropy=%.4f)",
            self._triangle_count, self._four_cycle_count, self._diamond_count,
            self._k4_count, self._tailed_triangle_count,
            self._tree_template_zipf, self._tree_template_entropy,
        )
        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 36-vector for cross-KG comparison.

        Layout: 7 motif counts; induced star counts (k=2..10); path-template
        zipf (k=2..10); path-template entropy (k=2..10); tree-template
        (zipf, entropy).

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
        star = _dict(self._star_counts)
        for k in range(2, _MAX_K + 1):
            vec.append(float(star.get(k, _nan)))
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

        lines.append("\n--- Star counts (induced) ---")
        star_row = "  " + "  ".join(f"k={k}: {self.star_counts.get(k, 0)}" for k in range(2, _MAX_K + 1))
        lines.append(star_row)

        lines.append("\n--- Path templates ---")
        lines.append(f"  {'k':>3}  {'zipf_alpha':>10}  {'entropy':>10}")
        for k in range(2, _MAX_K + 1):
            z = self.path_template_zipf.get(k, float("nan"))
            e = self.path_template_entropy.get(k, float("nan"))
            lines.append(f"  {k:>3}  {z:>10.4f}  {e:>10.4f}")

        lines.append("\n--- Tree templates ---")
        lines.append(f"  zipf_alpha={self.tree_template_zipf:.4f}  entropy={self.tree_template_entropy:.4f}")

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

            # Induced star counts vs k
            ax = axes[1, 0]
            star_vals = [float(self.star_counts.get(k, float("nan"))) for k in ks]
            ax.bar(ks, star_vals, color="mediumpurple")
            ax.set_xlabel("star size k (leaf count)")
            ax.set_ylabel("induced star count")
            ax.set_title("Induced star counts by size")

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
