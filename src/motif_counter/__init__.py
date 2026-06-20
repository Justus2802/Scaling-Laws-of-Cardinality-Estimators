"""Pluggable motif counting strategies.

``MotifCounter`` is the shared counting interface used by both Stage 3 rewiring
(``generator.stage3``) and the Block E signature measurement (``signature.block_e``).
Implementations can be swapped in via module-level constants in each consumer.

Public API
----------
cc_run                  — colour-coding graphlet estimator (Bressan et al. 2021)
cc_run_stars            — colour-coding star-treelet estimator
count_motifs5_escape    — exact 5-node graphlet counter (ESCAPE, WWW 2017)
MotifCounter            — abstract base class
CCMotifCounter          — colour-coding implementation (sampling-based)
ExactMotifCounter       — exact enumeration for k ≤ 4
HybridMotifCounter      — exact for k ≤ 5 (ESCAPE), CC for k ≥ 6

Private helpers (used by stage3 incremental delta)
--------------------------------------------------
MOTIF4_DS
_count_motifs4_through_edge
_motif4_delta
"""

from ._base import MotifCounter
from ._common import (
    _count_motifs4_through_edge,
    _motif4_delta,
    cc_run,
    cc_run_stars,
    count_motifs5_escape,
)
from .cc_motif_counter import CCMotifCounter
from .exact_motif_counter import ExactMotifCounter
from .hybrid_motif_counter import HybridMotifCounter

__all__ = [
    "MotifCounter",
    "CCMotifCounter",
    "ExactMotifCounter",
    "HybridMotifCounter",
    "cc_run",
    "cc_run_stars",
    "count_motifs5_escape",
    "_count_motifs4_through_edge",
    "_motif4_delta",
]
