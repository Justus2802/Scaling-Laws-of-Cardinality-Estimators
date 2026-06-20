"""Exact motif counter via full subgraph enumeration (k ≤ 4)."""

import math
from itertools import combinations

import igraph

from ._base import MotifCounter
from ._common import count_motifs5_escape, _count_motifs4_through_edge


class ExactMotifCounter(MotifCounter):
    """Exact motif counter via full subgraph enumeration (k ≤ 5).

    Triangle count uses igraph's ``list_triangles``; k=3 and k=4 graphlets are
    counted by direct enumeration.  Cost is O(m·Δ²) for k=4 where Δ is the
    maximum degree.  Star counts (k=2..10) use a triangle-node fast path:
    triangle-free nodes contribute C(d,k) directly; only triangle nodes need
    inclusion-exclusion over neighbourhood-induced edges.

    k=5 enumeration raises ``RuntimeError`` on graphs with a very high-degree
    hub (see ``count_motifs5_escape``); ``NotImplementedError`` is raised for
    k ≥ 6.
    """

    # Per-edge divisors for 4-node exact enumeration: each subgraph is found
    # once per edge contributing a valid (w,x) pair via _count_motifs4_through_edge.
    MOTIF4_DIVISORS: dict[tuple, int] = {
        (2, 2, 2, 2): 4,   # C4: 4 edges
        (2, 2, 3, 3): 5,   # diamond: 5 edges
        (3, 3, 3, 3): 6,   # K4: 6 edges
        (1, 2, 2, 3): 3,   # paw: 3 edges (pendant base-edge doesn't produce a valid pair)
    }

    # Nodes with degree above this threshold use direct subset enumeration
    # instead of inclusion-exclusion (avoids exponential blow-up on dense
    # neighbourhoods while staying tractable up to MAX_STAR_K=10).
    _HUB_THRESH = 50
    _MAX_STAR_K = 10

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
        if k == 5:
            # ESCAPE exact enumeration; raises RuntimeError on high-degree hubs.
            return count_motifs5_escape(g)
        raise NotImplementedError(
            f"ExactMotifCounter does not support k={k}; use CCMotifCounter for k ≥ 6"
        )

    def count_stars(self, g: igraph.Graph) -> dict[int, int]:
        """Count induced k-stars exactly for k=2..10.

        A k-star is one centre node connected to k leaves with NO edges between
        leaves (induced subgraph condition).

        Fast path: nodes that appear in NO triangle are guaranteed to have zero
        edges among their neighbours, so every k-subset of N(v) is a valid
        induced k-star — contribute C(d, k) directly.

        Slow path (triangle nodes only): use inclusion-exclusion over the
        neighbourhood-induced edge set E_v, or direct subset enumeration for
        high-degree nodes (degree > _HUB_THRESH).

        Cost: O(m + |triangle_nodes|·Δ²) — the triangle-free majority of KG
        nodes is handled in O(1) per node after the triangle listing.
        """
        MAX_K = self._MAX_STAR_K
        n = g.vcount()
        if n == 0:
            return {k: 0 for k in range(2, MAX_K + 1)}

        nbr: list[set[int]] = [set() for _ in range(n)]
        for e in g.es:
            nbr[e.source].add(e.target)
            nbr[e.target].add(e.source)

        # Nodes that appear in at least one triangle need the slow path.
        in_triangle: set[int] = set()
        if g.vcount() >= 3:
            for tri in g.list_triangles():
                in_triangle.update(tri)

        totals = [0] * (MAX_K + 1)

        for v in range(n):
            d = len(nbr[v])
            if d < 2:
                continue

            if v not in in_triangle:
                # Triangle-free centre: N(v) has no internal edges by definition.
                for k in range(2, min(d, MAX_K) + 1):
                    totals[k] += math.comb(d, k)
                continue

            nb_list = list(nbr[v])
            inner_edges: list[tuple[int, int]] = [
                (u, w)
                for u in nb_list
                for w in nbr[u]
                if w in nbr[v] and w > u
            ]

            if not inner_edges:
                # In a triangle but no inner edges from this centre's perspective.
                for k in range(2, min(d, MAX_K) + 1):
                    totals[k] += math.comb(d, k)
                continue

            if d > self._HUB_THRESH:
                inner_set: set[tuple[int, int]] = set(inner_edges)
                for k in range(2, min(d, MAX_K) + 1):
                    cnt = 0
                    for subset in combinations(nb_list, k):
                        ok = all(
                            (min(a, b), max(a, b)) not in inner_set
                            for i, a in enumerate(subset)
                            for b in subset[i + 1:]
                        )
                        if ok:
                            cnt += 1
                    totals[k] += cnt
                continue

            # Inclusion-exclusion over subsets of inner_edges.
            ie = len(inner_edges)
            correction = [0] * (MAX_K + 1)
            for mask in range(1, 1 << ie):
                verts: set[int] = set()
                bits = mask
                sign_exp = 0
                while bits:
                    idx = (bits & -bits).bit_length() - 1
                    u, w = inner_edges[idx]
                    verts.add(u)
                    verts.add(w)
                    sign_exp += 1
                    bits &= bits - 1
                s = len(verts)
                sign = (-1) ** (sign_exp + 1)
                for k in range(s, min(d, MAX_K) + 1):
                    correction[k] += sign * math.comb(d - s, k - s)

            for k in range(2, min(d, MAX_K) + 1):
                totals[k] += math.comb(d, k) - correction[k]

        return {k: totals[k] for k in range(2, MAX_K + 1)}

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
