"""Reduced Block F — Connectivity (G4).

Keeps the connectivity scalars (component structure, average-local clustering,
degree assortativity) and summarises sampled shortest-path lengths as three
descriptive statistics: max (diameter), mean, and variance.  The sampled path
lengths are kept on the object so ``visualize`` can show their histogram.
"""

import math

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.stats

from .._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED

log = get_logger(__name__)

_SAMPLE_K = 3       # default exponent: 10^k independently sampled pairs
_N_BOOTSTRAP = 999  # accepted for signature compatibility; no longer used


class BlockF(SignatureBlock):
    """Reduced Block F — connectivity features.

    Usage::

        b = BlockF().calculate(g)
        b.as_vector()                # fixed-length comparison vector
        b.as_dict()                  # named key-value pairs
        b.visualize(mode="text")     # CLI summary
        b.visualize(path="out.png")  # save plot to file
    """

    def __init__(self) -> None:
        self._num_components = _NOT_CALCULATED
        self._largest_component_fraction = _NOT_CALCULATED
        self._clustering_coefficient = _NOT_CALCULATED
        self._degree_assortativity = _NOT_CALCULATED
        self._shortest_path_max = _NOT_CALCULATED
        self._shortest_path_mean = _NOT_CALCULATED
        self._shortest_path_var = _NOT_CALCULATED
        # unsummarised data kept for visualization
        self._pair_dists_finite = _NOT_CALCULATED

    def __getattr__(self, name: str):
        """Migrate deserialized instances that still carry _shortest_path_skew."""
        if name in ("_shortest_path_max", "_shortest_path_mean", "_shortest_path_var"):
            skew = self.__dict__.get("_shortest_path_skew", _NOT_CALCULATED)
            if skew is _NOT_CALCULATED:
                return _NOT_CALCULATED
            try:
                loc, scale, shape, _lo, hi = skew
                if any(math.isnan(v) for v in (loc, scale, shape, hi)):
                    return float("nan")
                if name == "_shortest_path_max":
                    return float(hi)
                if name == "_shortest_path_mean":
                    return float(scipy.stats.skewnorm.mean(shape, loc=loc, scale=scale))
                return float(scipy.stats.skewnorm.var(shape, loc=loc, scale=scale))
            except Exception:
                return float("nan")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def num_components(self) -> int:
        return self._require("num_components", self._num_components)

    @property
    def largest_component_fraction(self) -> float:
        return self._require("largest_component_fraction", self._largest_component_fraction)

    @property
    def clustering_coefficient(self) -> float:
        return self._require("clustering_coefficient", self._clustering_coefficient)

    @property
    def degree_assortativity(self) -> float:
        return self._require("degree_assortativity", self._degree_assortativity)

    @property
    def shortest_path_max(self) -> float:
        return self._require("shortest_path_max", self._shortest_path_max)

    @property
    def shortest_path_mean(self) -> float:
        return self._require("shortest_path_mean", self._shortest_path_mean)

    @property
    def shortest_path_var(self) -> float:
        return self._require("shortest_path_var", self._shortest_path_var)

    # ── core ──────────────────────────────────────────────────────────────────

    def calculate(
        self,
        g: igraph.Graph,
        sample_k: int = _SAMPLE_K,
        n_bootstrap: int = _N_BOOTSTRAP,
        skip_shortest_paths: bool = False,
    ) -> "BlockF":
        """Compute reduced Block F (connectivity).

        Shortest-path length is estimated by sampling 10^sample_k independent
        (src, tgt) pairs with replacement from non-literal vertices in the largest
        weakly connected component (LCC); max, mean and variance are taken over the
        finite sampled lengths. Clustering and assortativity use the undirected
        simplification of g (same pattern as Block E).

        Parameters
        ----------
        n_bootstrap : int
            Accepted for signature compatibility; no longer used (the reduced
            block summarises the sampled lengths directly, without bootstrapping).
        skip_shortest_paths : bool
            When True, skip path-length sampling; path stats will be NaN.
        """
        if g.vcount() == 0:
            self._num_components = 0
            self._largest_component_fraction = float("nan")
            self._clustering_coefficient = float("nan")
            self._degree_assortativity = float("nan")
            self._pair_dists_finite = np.array([], dtype=float)
            self._shortest_path_max = float("nan")
            self._shortest_path_mean = float("nan")
            self._shortest_path_var = float("nan")
            log.info("Block F: empty graph — all features set to NaN/0")
            return self

        cc = g.connected_components(mode="weak")
        self._num_components = len(cc)
        log.info("Block F: computed num_components (%d)", self._num_components)
        lcc = cc.giant()
        self._largest_component_fraction = lcc.vcount() / g.vcount()
        log.info(
            "Block F: computed largest_component_fraction (%.4f, %d/%d vertices)",
            self._largest_component_fraction, lcc.vcount(), g.vcount(),
        )

        # --- Sampled shortest-path lengths (over the LCC, undirected BFS) ---
        if skip_shortest_paths:
            log.info("Block F: skipping shortest-path sampling.")
            finite = np.array([], dtype=float)
        else:
            non_lit: list[int] = [v.index for v in lcc.vs if not v["is_literal"]]
            if len(non_lit) >= 2:
                n_samples: int = 10 ** sample_k
                rng = np.random.default_rng(42)
                src_idx = rng.choice(len(non_lit), size=n_samples, replace=True)
                tgt_idx = rng.choice(len(non_lit), size=n_samples, replace=True)
                srcs: list[int] = [non_lit[i] for i in src_idx]
                tgts: list[int] = [non_lit[i] for i in tgt_idx]

                unique_srcs: list[int] = list(dict.fromkeys(srcs))
                unique_tgts: list[int] = list(dict.fromkeys(tgts))
                mat = np.array(
                    lcc.distances(source=unique_srcs, target=unique_tgts, mode="all"),
                    dtype=float,
                )
                src_pos: dict[int, int] = {v: i for i, v in enumerate(unique_srcs)}
                tgt_pos: dict[int, int] = {v: i for i, v in enumerate(unique_tgts)}
                pair_dists = np.array(
                    [mat[src_pos[s], tgt_pos[t]] for s, t in zip(srcs, tgts)],
                    dtype=float,
                )
                pair_dists[pair_dists == np.inf] = np.nan
                finite = pair_dists[pair_dists > 0]  # exclude self-pairs (distance == 0)
            else:
                finite = np.array([], dtype=float)

        self._pair_dists_finite = finite
        if finite.size:
            self._shortest_path_max  = float(np.max(finite))
            self._shortest_path_mean = float(np.mean(finite))
            self._shortest_path_var  = float(np.var(finite))
        else:
            self._shortest_path_max  = float("nan")
            self._shortest_path_mean = float("nan")
            self._shortest_path_var  = float("nan")

        # --- Clustering coefficient and assortativity (undirected simplification) ---
        g_und = g.as_undirected(combine_edges="first").simplify()
        self._clustering_coefficient = float(g_und.transitivity_avglocal_undirected(mode="zero"))
        log.info("Block F: computed clustering_coefficient (%.4f)", self._clustering_coefficient)
        self._degree_assortativity = float(g_und.assortativity_degree(directed=False))
        log.info("Block F: computed degree_assortativity (%.4f)", self._degree_assortativity)

        log.info(
            "Block F: path(max=%.1f, mean=%.3f, var=%.3f)",
            self._shortest_path_max, self._shortest_path_mean, self._shortest_path_var,
        )
        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 7-vector for cross-KG comparison.

        Layout: num_components; largest_component_fraction; clustering_coefficient;
        degree_assortativity; shortest_path_max; shortest_path_mean; shortest_path_var.

        Attributes absent from stale serialized data are emitted as NaN.
        """
        return [
            self._safe_scalar(lambda: self.num_components),
            self._safe_scalar(lambda: self.largest_component_fraction),
            self._safe_scalar(lambda: self.clustering_coefficient),
            self._safe_scalar(lambda: self.degree_assortativity),
            self._safe_scalar(lambda: self.shortest_path_max),
            self._safe_scalar(lambda: self.shortest_path_mean),
            self._safe_scalar(lambda: self.shortest_path_var),
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        return [
            "num_components",
            "largest_component_fraction",
            "clustering_coefficient",
            "degree_assortativity",
            "shortest_path_max",
            "shortest_path_mean",
            "shortest_path_var",
        ]

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 7-element NaN vector (same length as as_vector())."""
        return [float("nan")] * 7

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for reduced Block F.

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
        lines = [
            "=== Reduced Block F: Connectivity (G4) ===",
            f"  num_components            : {self.num_components}",
            f"  largest_component_fraction: {self.largest_component_fraction:.4f}",
            f"  clustering_coefficient    : {self.clustering_coefficient:.4f}",
            f"  degree_assortativity      : {self.degree_assortativity:.4f}",
            f"  shortest path             : max={self.shortest_path_max:.1f}, mean={self.shortest_path_mean:.3f}, var={self.shortest_path_var:.3f}",
        ]
        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            finite = self._require("_pair_dists_finite", self._pair_dists_finite)
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            if finite.size:
                int_max = int(finite.max())
                bins = max(1, int_max)
                ax.hist(finite, bins=bins, label="sampled distances")
                ax.axvline(self.shortest_path_mean, color="red", linestyle="--", label=f"mean={self.shortest_path_mean:.2f}")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "no path data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("shortest-path length")
            ax.set_ylabel("count")
            ax.set_title("Sampled shortest-path lengths")
            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block F: plot failed: %s", exc, exc_info=True)
            plt.close("all")
