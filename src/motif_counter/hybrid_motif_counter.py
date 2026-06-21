"""Hybrid motif counter: exact for k≤4, ESCAPE for k=5, CC for k≥6."""

import igraph
import numpy as np

from ._base import MotifCounter
from ._common import cc_run
from .exact_motif_counter import ExactMotifCounter
from ._logging import get_logger

log = get_logger(__name__)


class HybridMotifCounter(MotifCounter):
    """Exact counting for triangles, k=3, k=4; ESCAPE-exact for k=5; CC sampling for k≥6.

    Recommended for signature measurement — avoids CC variance on the most
    common motifs while staying tractable for large k.

    ``n_colorings`` is forwarded to the colour-coding estimator used for k≥6 (and
    the k=5 dense fallback): the per-type estimate is averaged over that many
    independent colourings to escape the single-colouring all-zero failure at k=6
    and reduce variance ~``1/n_colorings`` (Alon–Yuster–Zwick 1995; Motivo /
    Bressan et al. 2021).  The exact k≤5 paths ignore it.
    """

    def __init__(self, n_samples: int = 10_000, seed: int = 1, n_colorings: int = 16) -> None:
        self._n_samples = n_samples
        self._n_colorings = n_colorings
        self._rng = np.random.default_rng(seed)
        self._exact = ExactMotifCounter()

    def count_triangles(self, g: igraph.Graph) -> int:
        return self._exact.count_triangles(g)

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        if k <= 4:
            return self._exact.count_motifsk(g, k)
        if k == 5:
            try:
                return self._exact.count_motifsk(g, 5)
            except RuntimeError:
                # High-degree hub nodes make exact enumeration impractical; fall back to CC.
                return cc_run(g, k, self._n_samples, self._rng, n_colorings=self._n_colorings)
        return cc_run(g, k, self._n_samples, self._rng, n_colorings=self._n_colorings)

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        return self._exact.count_stars(g)
