"""Colour-coding motif counter (Bressan et al. 2021)."""

import igraph
import numpy as np

from ._base import MotifCounter
from ._common import cc_run, cc_run_stars


class CCMotifCounter(MotifCounter):
    """Colour-coding estimator (Bressan et al. 2021).

    Triangle count is exact (via igraph ``list_triangles``); all graphlet and
    star counts are estimated by the colour-coding sampler.
    """

    def __init__(self, n_samples: int = 10_000, seed: int = 1) -> None:
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    def count_triangles(self, g: igraph.Graph) -> int:
        return len(g.list_triangles()) if g.vcount() >= 3 else 0

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        return cc_run(g, k, self._n_samples, self._rng)

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        return cc_run_stars(g, self._n_samples, self._rng)
