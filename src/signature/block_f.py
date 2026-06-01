"""Block F — Connectivity features."""

import warnings

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.stats

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED

log = get_logger(__name__)

_SAMPLE_K = 3       # default exponent: 10^k independently sampled pairs
_N_BOOTSTRAP = 999   # default bootstrap resamples for SE estimation


class BlockF(SignatureBlock):
    """Block F — Connectivity features of a KG.

    Usage::

        b = BlockF().calculate(g)
        b.as_vector()                      # fixed-length comparison vector
        b.as_dict()                        # named key-value pairs
        b.visualize()                      # interactive matplotlib figure
        b.visualize(mode="text")           # CLI summary
        b.visualize(path="out.png")        # save plot to file
    """

    def __init__(self) -> None:
        self._num_components = _NOT_CALCULATED
        self._largest_component_fraction = _NOT_CALCULATED
        self._avg_shortest_path_length = _NOT_CALCULATED
        self._avg_shortest_path_length_se = _NOT_CALCULATED
        self._clustering_coefficient = _NOT_CALCULATED
        self._degree_assortativity = _NOT_CALCULATED
        self._pair_dists_finite: np.ndarray | None = None  # for histogram in visualize

    @property
    def num_components(self) -> int:
        return self._require("num_components", self._num_components)

    @property
    def largest_component_fraction(self) -> float:
        return self._require("largest_component_fraction", self._largest_component_fraction)

    @property
    def avg_shortest_path_length(self) -> float:
        return self._require("avg_shortest_path_length", self._avg_shortest_path_length)

    @property
    def avg_shortest_path_length_se(self) -> float:
        return self._require("avg_shortest_path_length_se", self._avg_shortest_path_length_se)

    @property
    def clustering_coefficient(self) -> float:
        return self._require("clustering_coefficient", self._clustering_coefficient)

    @property
    def degree_assortativity(self) -> float:
        return self._require("degree_assortativity", self._degree_assortativity)

    def calculate(
        self,
        g: igraph.Graph,
        sample_k: int = _SAMPLE_K,
        n_bootstrap: int = _N_BOOTSTRAP,
    ) -> "BlockF":
        """Compute Block F (connectivity) of the graph signature.

        Shortest-path length is estimated by sampling 10^sample_k independent
        (src, tgt) pairs with replacement from non-literal vertices in the largest
        weakly connected component (LCC). Unique sources and targets are
        deduplicated into a single distances() call; per-pair distances are then
        looked up from the resulting matrix. Undirected BFS (mode='all') ensures
        every pair within the weakly connected LCC is reachable.

        Clustering and assortativity both use the undirected simplification of g
        (same pattern as Block E).
        """
        if g.vcount() == 0:
            self._num_components = 0
            self._largest_component_fraction = float("nan")
            self._avg_shortest_path_length = float("nan")
            self._avg_shortest_path_length_se = float("nan")
            self._clustering_coefficient = float("nan")
            self._degree_assortativity = float("nan")
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

        # --- Sampled avg shortest-path length ---
        non_lit: list[int] = [v.index for v in lcc.vs if not v["is_literal"]]
        avg_sp = float("nan")
        sp_se = float("nan")
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
            self._pair_dists_finite = finite
            if finite.size >= 2:
                avg_sp = float(np.mean(finite))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = scipy.stats.bootstrap(
                        (finite,), np.mean, n_resamples=n_bootstrap, rng=42
                    )
                sp_se = float(res.standard_error)
            elif finite.size == 1:
                avg_sp = float(finite[0])
                sp_se = float("nan")

        self._avg_shortest_path_length = avg_sp
        self._avg_shortest_path_length_se = sp_se
        log.info(
            "Block F: computed avg_shortest_path_length (%.4f ± %.4f SE)",
            avg_sp, sp_se,
        )

        # --- Clustering coefficient and assortativity (undirected simplification) ---
        g_und = g.as_undirected(combine_edges="first").simplify()
        self._clustering_coefficient = float(g_und.transitivity_avglocal_undirected(mode="zero"))
        log.info(
            "Block F: computed clustering_coefficient (%.4f)", self._clustering_coefficient
        )
        self._degree_assortativity = float(g_und.assortativity_degree(directed=False))
        log.info(
            "Block F: computed degree_assortativity (%.4f)", self._degree_assortativity
        )

        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 6-vector for cross-KG comparison."""
        return [
            float(self.num_components),
            self.largest_component_fraction,
            self.avg_shortest_path_length,
            self.avg_shortest_path_length_se,
            self.clustering_coefficient,
            self.degree_assortativity,
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        return [
            "num_components",
            "largest_component_fraction",
            "avg_shortest_path_length",
            "avg_shortest_path_length_se",
            "clustering_coefficient",
            "degree_assortativity",
        ]

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 6-element NaN vector (same length as as_vector())."""
        return [float("nan")] * 6

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics.

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

    def _visualize_text(self, path: str | None) -> None:
        lines = [
            "Block F — Connectivity",
            f"  num_components:              {self.num_components}",
            f"  largest_component_fraction:  {self.largest_component_fraction:.4f}",
            f"  avg_shortest_path_length:    {self.avg_shortest_path_length:.4f}",
            f"  avg_shortest_path_length_se: {self.avg_shortest_path_length_se:.4f}",
            f"  clustering_coefficient:      {self.clustering_coefficient:.4f}",
            f"  degree_assortativity:        {self.degree_assortativity:.4f}",
        ]
        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

            finite = self._pair_dists_finite
            if finite is not None and finite.size > 0:
                # distances are integers — one bin per integer value
                int_max = int(finite.max())
                bins = np.arange(1, int_max + 2) - 0.5
                ax.hist(finite, bins=bins, color="steelblue", edgecolor="white")
                ax.axvline(self.avg_shortest_path_length, color="crimson", linestyle="--",
                           label=f"mean={self.avg_shortest_path_length:.2f}")
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "no path data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Sampled shortest-path lengths")
            ax.set_xlabel("distance")
            ax.set_ylabel("count")

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block F: plot failed: %s", exc, exc_info=True)
            plt.close("all")
