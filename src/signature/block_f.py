"""Block F — Connectivity features."""

import warnings
from typing import Any

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.stats

from ._logging import get_logger

log = get_logger(__name__)

_SAMPLE_K = 0        # default exponent: 10^k independently sampled pairs
_N_BOOTSTRAP = 999   # default bootstrap resamples for SE estimation

_NOT_CALCULATED = object()


class BlockF:
    """Block F — Connectivity features of a KG.

    Usage::

        b = BlockF().calculate(g)
        b.as_vector()                      # fixed-length comparison vector
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

    def _require(self, name: str, value: object) -> Any:
        if value is _NOT_CALCULATED:
            raise RuntimeError(f"Call calculate() before accessing {name}")
        return value

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
        log.info("Block F: starting calculation (vertices=%d, edges=%d)", g.vcount(), g.ecount())

        if g.vcount() == 0:
            log.warning("Block F: empty graph — all metrics set to NaN")
            self._num_components = 0
            self._largest_component_fraction = float("nan")
            self._avg_shortest_path_length = float("nan")
            self._avg_shortest_path_length_se = float("nan")
            self._clustering_coefficient = float("nan")
            self._degree_assortativity = float("nan")
            return self

        cc = g.connected_components(mode="weak")
        self._num_components = len(cc)
        lcc = cc.giant()
        self._largest_component_fraction = lcc.vcount() / g.vcount()
        log.info(
            "Block F: %d weakly connected component(s); LCC fraction=%.4f (%d/%d vertices)",
            self._num_components,
            self._largest_component_fraction,
            lcc.vcount(),
            g.vcount(),
        )

        # --- Sampled avg shortest-path length ---
        non_lit: list[int] = [v.index for v in lcc.vs if not v["is_literal"]]
        avg_sp = float("nan")
        sp_se = float("nan")
        log.debug("Block F: %d non-literal vertices in LCC available for path sampling", len(non_lit))
        if len(non_lit) < 2:
            log.warning(
                "Block F: only %d non-literal vertex/vertices in LCC — "
                "skipping shortest-path estimation",
                len(non_lit),
            )
        else:
            n_samples: int = 10 ** sample_k
            log.debug("Block F: sampling %d (src, tgt) pairs (sample_k=%d)", n_samples, sample_k)
            rng = np.random.default_rng(42)
            src_idx = rng.choice(len(non_lit), size=n_samples, replace=True)
            tgt_idx = rng.choice(len(non_lit), size=n_samples, replace=True)
            srcs: list[int] = [non_lit[i] for i in src_idx]
            tgts: list[int] = [non_lit[i] for i in tgt_idx]

            unique_srcs: list[int] = list(dict.fromkeys(srcs))
            unique_tgts: list[int] = list(dict.fromkeys(tgts))
            log.debug(
                "Block F: BFS distance matrix — %d unique sources × %d unique targets",
                len(unique_srcs),
                len(unique_tgts),
            )
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
            n_inf = int(np.sum(np.isnan(pair_dists)))
            log.debug(
                "Block F: %d finite path(s), %d unreachable pair(s) out of %d sampled",
                finite.size,
                n_inf,
                n_samples,
            )
            if finite.size >= 2:
                avg_sp = float(np.mean(finite))
                log.debug(
                    "Block F: bootstrapping SE with n_resamples=%d", n_bootstrap
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = scipy.stats.bootstrap(
                        (finite,), np.mean, n_resamples=n_bootstrap, rng=42
                    )
                sp_se = float(res.standard_error)
                log.info(
                    "Block F: avg shortest-path length=%.4f ± %.4f (SE)",
                    avg_sp,
                    sp_se,
                )
            elif finite.size == 1:
                avg_sp = float(finite[0])
                sp_se = float("nan")
                log.warning("Block F: only 1 finite path found; SE is NaN")

        self._avg_shortest_path_length = avg_sp
        self._avg_shortest_path_length_se = sp_se

        # --- Clustering coefficient and assortativity (undirected simplification) ---
        log.debug("Block F: computing clustering coefficient and degree assortativity")
        g_und = g.as_undirected(combine_edges="first").simplify()
        self._clustering_coefficient = float(g_und.transitivity_avglocal_undirected(mode="zero"))
        self._degree_assortativity = float(g_und.assortativity_degree(directed=False))
        log.info(
            "Block F: clustering_coefficient=%.4f, degree_assortativity=%.4f",
            self._clustering_coefficient,
            self._degree_assortativity,
        )

        log.info("Block F: calculation complete")
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
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Shortest-path length histogram
            ax = axes[0]
            finite = self._pair_dists_finite
            if finite is not None and finite.size > 0:
                ax.hist(finite, bins=20, color="steelblue", edgecolor="white")
                ax.axvline(self.avg_shortest_path_length, color="crimson", linestyle="--",
                           label=f"mean={self.avg_shortest_path_length:.2f}")
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "no path data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Sampled shortest-path lengths")
            ax.set_xlabel("distance")
            ax.set_ylabel("count")

            # Scalar summary bar chart
            ax = axes[1]
            labels = ["LCC fraction", "clustering", "assortativity"]
            values = [
                self.largest_component_fraction,
                self.clustering_coefficient,
                self.degree_assortativity,
            ]
            colors = ["steelblue" if v >= 0 else "tomato" for v in values]
            ax.bar(labels, values, color=colors, edgecolor="white")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"Connectivity scalars  (components={self.num_components})")
            ax.set_ylim(-1, 1)

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block F: plot failed: %s", exc, exc_info=True)
            plt.close("all")
