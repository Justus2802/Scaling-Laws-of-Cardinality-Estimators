"""Exact motif counter via full subgraph enumeration (k ≤ 4)."""

import igraph

from ._base import MotifCounter
from ._common import _count_motifs4_through_edge


class ExactMotifCounter(MotifCounter):
    # Per-edge divisors for 4-node exact enumeration: each subgraph is found
    # once per edge contributing a valid (w,x) pair via _count_motifs4_through_edge.
    MOTIF4_DIVISORS: dict[tuple, int] = {
        (2, 2, 2, 2): 4,   # C4: 4 edges
        (2, 2, 3, 3): 5,   # diamond: 5 edges
        (3, 3, 3, 3): 6,   # K4: 6 edges
        (1, 2, 2, 3): 3,   # paw: 3 edges (pendant base-edge doesn't produce a valid pair)
    }
    """Exact motif counter via full subgraph enumeration (k ≤ 4 only).

    Triangle count uses igraph's ``list_triangles``; k=3 and k=4 graphlets are
    counted by direct enumeration.  Cost is O(m·Δ²) for k=4 where Δ is the
    maximum degree.

    Raises ``NotImplementedError`` for k ≥ 5 and for star counting.
    """

    def count_triangles(self, g: igraph.Graph) -> int:
        return len(g.list_triangles()) if g.vcount() >= 3 else 0

    def count_motifsk(self, g: igraph.Graph, k: int) -> dict[tuple, int]:
        if k == 2:
            m = g.ecount()
            return {(1, 1): m} if m > 0 else {}
        if k == 3:
            return self._count_motifs3(g)
        if k == 4:
            return self._count_motifs4_exact(g)
        raise NotImplementedError(
            f"ExactMotifCounter does not support k={k}; use CCMotifCounter for k ≥ 5"
        )

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        raise NotImplementedError(
            "ExactMotifCounter does not implement star counting; use CCMotifCounter"
        )

    def _count_motifs3(self, g: igraph.Graph) -> dict[tuple, int]:
        n = g.vcount()
        if n < 2:
            return {}
        tris = len(g.list_triangles()) if n >= 3 else 0
        degs = g.degree()
        wedges = sum(d * (d - 1) // 2 for d in degs) - 3 * tris
        result: dict[tuple, int] = {}
        if tris > 0:
            result[(2, 2, 2)] = tris
        if wedges > 0:
            result[(1, 1, 2)] = wedges
        return result

    def _count_motifs4_exact(self, g: igraph.Graph) -> dict[tuple, int]:
        n = g.vcount()
        adj: list[dict] = [{} for _ in range(n)]
        for e in g.es:
            u, v = e.source, e.target
            adj[u][v] = 1
            adj[v][u] = 1

        counts: dict[tuple, int] = {}
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for ds, cnt in _count_motifs4_through_edge(adj, u, v).items():
                    counts[ds] = counts.get(ds, 0) + cnt

        return {
            ds: cnt // self.MOTIF4_DIVISORS[ds]
            for ds, cnt in counts.items()
            if ds in self.MOTIF4_DIVISORS and cnt > 0
        }
