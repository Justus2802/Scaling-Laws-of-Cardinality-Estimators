"""Colour-coding motif counter (Bressan et al. 2021)."""

import igraph
import numpy as np

from ._base import MotifCounter
from ._common import cc_run, cc_run_stars


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
    """

    def __init__(self, n_samples: int = 10_000, seed: int = 1, n_colorings: int = 16) -> None:
        self._n_samples = n_samples
        self._n_colorings = n_colorings
        self._rng = np.random.default_rng(seed)

    def count_triangles(self, g: igraph.Graph) -> int:
        return len(g.list_triangles()) if g.vcount() >= 3 else 0

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        return cc_run(g, k, self._n_samples, self._rng, n_colorings=self._n_colorings)

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        return cc_run_stars(g, self._n_samples, self._rng, n_colorings=self._n_colorings)
