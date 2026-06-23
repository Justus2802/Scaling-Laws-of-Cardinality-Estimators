"""Hybrid motif counter: exact for k≤3, ESCAPE for k=5, CC for k=4 and k≥6."""

import igraph
import numpy as np

from ._base import MotifCounter

from .exact_motif_counter import ExactMotifCounter
from .cc_motif_counter import CCMotifCounter
from ._logging import get_logger

log = get_logger(__name__)


class HybridMotifCounter(MotifCounter):
    """Exact counting for triangles and k=3; ESCAPE-exact for k=5; CC sampling for k=4 and k≥6.

    Recommended for signature measurement — exact on triangles/3-node graphlets,
    CC-sampled for 4-node motifs, while staying tractable for large k.

    ``n_colorings`` is forwarded to the colour-coding estimator used for k=4, k≥6
    (and the k=5 dense fallback): the per-type estimate is averaged over that many
    independent colourings to escape the single-colouring all-zero failure at k=6
    and reduce variance ~``1/n_colorings`` (Alon–Yuster–Zwick 1995; Motivo /
    Bressan et al. 2021).  The exact k≤5 paths ignore it.
    """

    def __init__(self, n_samples: int = 10_000, seed: int = 1, n_colorings: int = 16) -> None:
        self._n_samples = n_samples
        self._n_colorings = n_colorings
        self._rng = np.random.default_rng(seed)
        self._exact = ExactMotifCounter()
        self._cc = CCMotifCounter(n_samples=n_samples, seed=seed, n_colorings=n_colorings)

    def count_triangles(self, g: igraph.Graph) -> int:
        return self._exact.count_triangles(g)

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        if k <= 3:
            return self._exact.count_motifsk(g, k)
        if k == 4:
            return self._cc.count_motifsk(g, k)
        if k == 5:
            try:
                return self._cc.count_motifsk(g, 5)
            except RuntimeError:
                # High-degree hub nodes make exact enumeration impractical; fall back to CC.
                return self._cc.count_motifsk(g, 5)
        return self._cc.count_motifsk(g, k)

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        return self._cc.count_stars(g)
