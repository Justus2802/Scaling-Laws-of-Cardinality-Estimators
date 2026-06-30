"""Reduced Block F — Connectivity (G4).

Keeps the connectivity scalars (component structure, average-local clustering,
degree assortativity) and summarises sampled shortest-path lengths as three
descriptive statistics: max (diameter), mean, and variance.  The path sampling,
clustering and assortativity computations are reused from the original Block F
(composed, not re-implemented); the sampled path lengths are kept on the object
so ``visualize`` can show their histogram.
"""

import math

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.stats

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._orig_block_f import BlockF as _OrigBlockF, _SAMPLE_K, _N_BOOTSTRAP

log = get_logger(__name__)


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

        Composes the original Block F to reuse its component analysis, sampled
        shortest-path lengths, average-local clustering and degree assortativity,
        then computes max, mean, and variance over the finite sampled path lengths.

        Parameters
        ----------
        skip_shortest_paths : bool
            When True, skip path-length sampling; path stats will be NaN.
        """
        orig = _OrigBlockF().calculate(g, sample_k=sample_k, n_bootstrap=n_bootstrap,
                                       skip_shortest_paths=skip_shortest_paths)
        self._num_components = orig.num_components
        self._largest_component_fraction = orig.largest_component_fraction
        self._clustering_coefficient = orig.clustering_coefficient
        self._degree_assortativity = orig.degree_assortativity

        finite = orig._pair_dists_finite
        finite = np.asarray(finite, dtype=float) if finite is not None else np.array([], dtype=float)
        self._pair_dists_finite = finite
        if finite.size:
            self._shortest_path_max  = float(np.max(finite))
            self._shortest_path_mean = float(np.mean(finite))
            self._shortest_path_var  = float(np.var(finite))
        else:
            self._shortest_path_max  = float("nan")
            self._shortest_path_mean = float("nan")
            self._shortest_path_var  = float("nan")

        log.info(
            "Block F: components=%d, lcc=%.4f, clustering=%.4f, path(max=%.1f, mean=%.3f, var=%.3f)",
            self._num_components, self._largest_component_fraction,
            self._clustering_coefficient, self._shortest_path_max,
            self._shortest_path_mean, self._shortest_path_var,
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
