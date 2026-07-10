"""Pluggable motif counting strategies.

``MotifCounter`` is the shared counting interface used by both Stage 3 rewiring
(``generator.stage3``) and the Block E signature measurement (``signature.block_e``).
Implementations can be swapped in via module-level constants in each consumer.

Public API
----------
cc_run                  — colour-coding graphlet estimator (Bressan et al. 2021)
cc_run_stars            — colour-coding star-treelet estimator (vectorised)
cc_run_stars_loop       — un-vectorised reference star estimator (benchmark baseline)
count_motifs5_escape    — exact 5-node graphlet counter (ESCAPE, WWW 2017)
count_motifsk_escape    — exact k-node graphlet counter (ESCAPE, k=5/6)
MotifCounter            — abstract base class
CCMotifCounter          — colour-coding implementation (sampling-based)
ExactMotifCounter       — exact enumeration for k ≤ 6 (ESCAPE for k=5/6)
HybridMotifCounter      — exact for k ≤ 3, CC for k ≥ 4

Incremental SA delta helpers live in ``generator.local_updates``.
"""

from ._base import MotifCounter
from ._common import (
    count_motifs5_escape,
    count_motifsk_escape,
)
from .cc_motif_counter import (
    CCMotifCounter,
    cc_run,
    cc_run_stars,
    cc_run_stars_loop,
)
from .exact_motif_counter import ExactMotifCounter
from .hybrid_motif_counter import HybridMotifCounter

__all__ = [
    "MotifCounter",
    "CCMotifCounter",
    "ExactMotifCounter",
    "HybridMotifCounter",
    "cc_run",
    "cc_run_stars",
    "cc_run_stars_loop",
    "count_motifs5_escape",
    "count_motifsk_escape",
]
