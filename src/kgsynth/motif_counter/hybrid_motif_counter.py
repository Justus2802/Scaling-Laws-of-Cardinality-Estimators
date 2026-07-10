"""Hybrid motif counter: exact for k≤3, colour-coding for k≥4."""

import igraph

from ._base import MotifCounter

from .exact_motif_counter import ExactMotifCounter
from .cc_motif_counter import CCMotifCounter
from .._logging import get_logger

log = get_logger(__name__)


class HybridMotifCounter(MotifCounter):
    """Exact counting for triangles and k≤3; colour-coding sampling for k≥4.

    Recommended for signature measurement — exact on triangles/3-node graphlets,
    CC-sampled from k=4 up, which keeps it tractable on large, hub-heavy KGs.

    **Everything from k=4 up is an estimate**, at every graph size. In particular
    the diamond graphlet (degree sequence ``(2,2,3,3)``) is systematically
    over-counted by ~50%, and the bias does not shrink with sample budget — see
    ``KNOWN_CC_DIAMOND_BIAS`` in ``tests/test_hybrid_motif_counter.py``. Callers
    needing exact 3-/4-node counts should use ``ExactMotifCounter`` directly.

    k=5 used to route to ``ExactMotifCounter`` (ESCAPE); commit ``c8fdd4e``
    changed it to CC. On real KGs this is moot — ESCAPE's ``_ESCAPE_MAX_DEGREE``
    guard (50) rejects every graph in the corpus, whose minimum max-degree is 68 —
    so the exact path would have fallen back to CC regardless.

    ``n_colorings`` is forwarded to the colour-coding estimator: the per-type
    estimate is averaged over that many independent colourings to escape the
    single-colouring all-zero failure at k=6 and reduce variance ~``1/n_colorings``
    (Alon–Yuster–Zwick 1995; Motivo / Bressan et al. 2021). The exact k≤3 path
    ignores it.

    ``adaptive`` is forwarded to the CC sampler: when True its per-call path-sample
    count scales with graph size (see ``CCMotifCounter``); the exact path ignores it.
    """

    def __init__(self, n_samples: int = 20000, seed: int = 1, n_colorings: int = 16,
                 adaptive: bool = False) -> None:
        self._n_samples = n_samples
        self._n_colorings = n_colorings
        self._exact = ExactMotifCounter()
        # Seeding lives entirely on the CC sampler; this class draws no randomness itself.
        self._cc = CCMotifCounter(n_samples=n_samples, seed=seed,
                                  n_colorings=n_colorings, adaptive=adaptive)

    def count_triangles(self, g: igraph.Graph) -> int:
        return self._exact.count_triangles(g)

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        if k <= 3:
            return self._exact.count_motifsk(g, k)
        return self._cc.count_motifsk(g, k)

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        return self._cc.count_stars(g)
