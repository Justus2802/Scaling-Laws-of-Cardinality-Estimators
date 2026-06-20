"""Hybrid motif counter: exact for k≤4, ESCAPE for k=5, CC for k≥6."""

import igraph
import numpy as np

from ._base import MotifCounter
from ._common import cc_run, cc_run_stars, count_motifs5_escape
from .exact_motif_counter import ExactMotifCounter
from ._logging import get_logger

log = get_logger(__name__)


class HybridMotifCounter(MotifCounter):
    """Exact counting for triangles, k=3, k=4; ESCAPE-exact for k=5; CC sampling for k≥6.

    Recommended for signature measurement — avoids CC variance on the most
    common motifs while staying tractable for large k.
    """

    def __init__(self, n_samples: int = 10_000, seed: int = 1) -> None:
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)
        self._exact = ExactMotifCounter()

    def count_triangles(self, g: igraph.Graph) -> int:
        return self._exact.count_triangles(g)

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        if k <= 4:
            return self._exact.count_motifsk(g, k)
        if k == 5:
            try:
                return count_motifs5_escape(g)
            except RuntimeError as exc:
                log.warning("ESCAPE k=5 fell back to CC sampling: %s", exc)
                return cc_run(g, k, self._n_samples, self._rng)
        return cc_run(g, k, self._n_samples, self._rng)

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        return cc_run_stars(g, self._n_samples, self._rng)
