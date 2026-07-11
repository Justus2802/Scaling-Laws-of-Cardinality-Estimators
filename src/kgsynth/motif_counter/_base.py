"""Abstract base class for motif counting strategies."""

from abc import ABC, abstractmethod

import igraph


class MotifCounter(ABC):
    """Counts motif instances in an undirected simple graph, grouped by family.

    Implementations are selected via module-level constants in each consumer
    (``INITIAL_MOTIF_COUNTER``/``REMEASURE_MOTIF_COUNTER`` in stage3.py,
    ``MOTIF_COUNTER`` in block_e.py).

    Class-level constants define the graphlet types shared across all implementations.
    """

    # 4-node connected motif types (sorted degree sequences).
    MOTIF4_DS: frozenset[tuple] = frozenset(
        {(2, 2, 2, 2), (2, 2, 3, 3), (3, 3, 3, 3), (1, 2, 2, 3)}
    )

    # σ_H: number of directed spanning P_k paths for each graphlet type H.
    # Used by cc_run to convert raw sample proportions to estimated counts.
    SIGMA: dict[tuple, int] = {
        # k=3
        (2, 2, 2): 6,        # triangle (C3): 3 spanning P3 paths × 2 directions
        # k=4
        (1, 1, 2, 2): 2,    # P4 path
        (2, 2, 2, 2): 8,    # C4
        (1, 2, 2, 3): 4,    # tailed triangle
        (2, 2, 3, 3): 8,    # diamond
        (3, 3, 3, 3): 24,   # K4
        # k=5
        (2, 2, 2, 2, 2): 10,    # C5
        # k=6
        (2, 2, 2, 2, 2, 2): 12,  # C6
    }

    # Degree sequences of the 5- and 6-cycle, used by count_cycles.
    C5_DS: tuple = (2, 2, 2, 2, 2)
    C6_DS: tuple = (2, 2, 2, 2, 2, 2)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def count_triangles(self, g: igraph.Graph) -> int:
        """Count triangles (3-node cycles) exactly."""

    @abstractmethod
    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        """Count k-node connected graphlets.

        Returns ``{sorted_degree_sequence_tuple: count}``.
        ``ExactMotifCounter`` only supports k ≤ 4; raises ``NotImplementedError``
        for larger k.
        """

    @abstractmethod
    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        """Count induced k-stars for k=2..10. Returns ``{k: count}``."""

    # ── Concrete convenience wrappers ────────────────────────────────────────

    def count_motifs3(self, g: igraph.Graph) -> dict[tuple, int]:
        """Count 3-node graphlets: (2,2,2)=triangle, (1,1,2)=open wedge."""
        return self.count_motifsk(g, 3)

    def count_motifs4(self, g: igraph.Graph) -> dict[tuple, int]:
        """Count 4-node connected motifs (C4, diamond, K4, paw)."""
        return self.count_motifsk(g, 4)

    def count_cycles(
        self,
        g: igraph.Graph,
        k5: bool = True,
        k6: bool = True,
    ) -> tuple[int, int]:
        """Estimate 5- and 6-cycle counts via count_motifsk.

        Returns (c5, c6); skips a size if the corresponding flag is False.
        Delegates to ``count_motifsk(g, 5)`` and ``count_motifsk(g, 6)``,
        which each implementation resolves (exact or CC) as appropriate.
        """
        c5 = self.count_motifsk(g, 5).get(self.C5_DS, 0) if k5 else 0
        c6 = self.count_motifsk(g, 6).get(self.C6_DS, 0) if k6 else 0
        return c5, c6
