"""Tests for generator.motif_counter — ExactMotifCounter and helpers."""

import os
import sys
import unittest

import igraph
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from motif_counter import (
    CCMotifCounter,
    ExactMotifCounter,
)
from generator.local_updates import _count_motifs4_through_edge, _motif4_delta


def _und(n: int, edges: list[tuple[int, int]]) -> igraph.Graph:
    """Simple undirected graph."""
    g = igraph.Graph(n=n, directed=False)
    g.add_edges(edges)
    return g


def _adj(n: int, edges: list[tuple[int, int]]) -> list[dict]:
    """Adjacency dict list matching the format used in stage3."""
    a: list[dict] = [{} for _ in range(n)]
    for u, v in edges:
        a[u][v] = a[u].get(v, 0) + 1
        a[v][u] = a[v].get(u, 0) + 1
    return a


_EXACT = ExactMotifCounter()
_CC = CCMotifCounter(n_samples=200_000, seed=0)


# ── Triangle counting ─────────────────────────────────────────────────────────

class TestCountTriangles(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(_EXACT.count_triangles(_und(0, [])), 0)

    def test_single_node(self):
        self.assertEqual(_EXACT.count_triangles(_und(1, [])), 0)

    def test_single_edge(self):
        self.assertEqual(_EXACT.count_triangles(_und(2, [(0, 1)])), 0)

    def test_path_no_triangle(self):
        # 0-1-2 — no triangle
        self.assertEqual(_EXACT.count_triangles(_und(3, [(0, 1), (1, 2)])), 0)

    def test_one_triangle(self):
        self.assertEqual(_EXACT.count_triangles(_und(3, [(0, 1), (1, 2), (0, 2)])), 1)

    def test_two_triangles_sharing_edge(self):
        # Diamond has 2 triangles
        g = _und(4, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
        self.assertEqual(_EXACT.count_triangles(g), 2)

    def test_k4(self):
        # K4 has 4 triangles
        g = _und(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        self.assertEqual(_EXACT.count_triangles(g), 4)


# ── 4-node motif counting — single canonical graphs ──────────────────────────

class TestCountMotifs4Canonical(unittest.TestCase):
    """Each test graph contains exactly one instance of the target motif type."""

    def _assert_only(self, g: igraph.Graph, ds: tuple, expected: int) -> None:
        counts = _EXACT.count_motifs4(g)
        self.assertEqual(
            counts.get(ds, 0), expected,
            f"expected {ds}={expected}, got {counts}",
        )
        # No other motif type present
        for k, v in counts.items():
            if k != ds:
                self.assertEqual(v, 0, f"unexpected motif {k}={v}")

    def test_c4_single(self):
        # 4-cycle: 0-1-2-3-0
        g = _und(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        self._assert_only(g, (2, 2, 2, 2), 1)

    def test_diamond_single(self):
        # Diamond: K4 minus one edge (missing 2-3)
        g = _und(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
        self._assert_only(g, (2, 2, 3, 3), 1)

    def test_k4_single(self):
        # Complete K4
        g = _und(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        self._assert_only(g, (3, 3, 3, 3), 1)

    def test_paw_single(self):
        # Paw: triangle 0-1-2 plus pendant 0-3
        g = _und(4, [(0, 1), (1, 2), (0, 2), (0, 3)])
        self._assert_only(g, (1, 2, 2, 3), 1)

    def test_empty_no_motifs(self):
        g = _und(4, [])
        self.assertEqual(_EXACT.count_motifs4(g), {})

    def test_star_no_4motif(self):
        # Star K_{1,3}: not a connected 4-node motif in MOTIF4_DS
        g = _und(4, [(0, 1), (0, 2), (0, 3)])
        self.assertEqual(_EXACT.count_motifs4(g), {})

    def test_path4_no_4motif(self):
        # Path 0-1-2-3 has degree sequence (1,2,2,1), not in MOTIF4_DS
        g = _und(4, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(_EXACT.count_motifs4(g), {})

    def test_multiple_c4(self):
        # Two disjoint 4-cycles on 8 nodes
        g = _und(8, [(0,1),(1,2),(2,3),(0,3), (4,5),(5,6),(6,7),(4,7)])
        counts = _EXACT.count_motifs4(g)
        self.assertEqual(counts.get((2,2,2,2), 0), 2)


# ── ExactMotifCounter vs CCMotifCounter agreement ────────────────────────────

class TestExactVsCC(unittest.TestCase):
    """Exact and CC counters should agree on 4-motif counts within CC variance."""

    def _petersen_like(self) -> igraph.Graph:
        """A moderately connected graph with several motif types present."""
        # Petersen graph: 10 nodes, 15 edges; has many 4-node subgraphs
        edges = [
            (0,1),(1,2),(2,3),(3,4),(4,0),   # outer pentagon
            (0,5),(1,6),(2,7),(3,8),(4,9),   # spokes
            (5,7),(7,9),(9,6),(6,8),(8,5),   # inner pentagram
        ]
        return _und(10, edges)

    def test_triangles_agree(self):
        g = self._petersen_like()
        self.assertEqual(_EXACT.count_triangles(g), _CC.count_triangles(g))

    def test_motifs4_close(self):
        g = self._petersen_like()
        exact = _EXACT.count_motifs4(g)
        cc = CCMotifCounter(n_samples=500_000, seed=7).count_motifs4(g)
        for ds in exact:
            e_val = exact[ds]
            c_val = cc.get(ds, 0)
            if e_val == 0:
                continue
            rel_err = abs(e_val - c_val) / e_val
            self.assertLess(
                rel_err, 0.15,
                f"CC and exact disagree too much on {ds}: exact={e_val}, cc={c_val}",
            )


# ── _count_motifs4_through_edge ───────────────────────────────────────────────

class TestCountMotifs4ThroughEdge(unittest.TestCase):

    def test_c4_through_one_edge(self):
        # 4-cycle 0-1-2-3-0; count through edge 0-1
        a = _adj(4, [(0,1),(1,2),(2,3),(0,3)])
        counts = _count_motifs4_through_edge(a, 0, 1)
        self.assertEqual(counts.get((2,2,2,2), 0), 1)

    def test_no_motif_through_dangling_edge(self):
        # Star: 0 connected to 1,2,3 — no 4-node motif through any edge
        a = _adj(4, [(0,1),(0,2),(0,3)])
        self.assertEqual(_count_motifs4_through_edge(a, 0, 1), {})


# ── _motif4_delta ─────────────────────────────────────────────────────────────

class TestMotif4Delta(unittest.TestCase):
    """_motif4_delta should match the difference of full counts before/after swap."""

    def _full_count(self, a: list[dict]) -> dict[tuple, int]:
        n = len(a)
        counts: dict[tuple, int] = {}
        for u in range(n):
            for v in a[u]:
                if v <= u:
                    continue
                for ds, c in _count_motifs4_through_edge(a, u, v).items():
                    counts[ds] = counts.get(ds, 0) + c
        _DIV = {(2,2,2,2):4, (2,2,3,3):5, (3,3,3,3):6, (1,2,2,3):3}
        return {ds: c//_DIV[ds] for ds,c in counts.items() if ds in _DIV and c>0}

    def _apply_swap(self, a: list[dict], s1:int, o1:int, s2:int, o2:int) -> None:
        def inc(u,v): a[u][v]=a[u].get(v,0)+1; a[v][u]=a[v].get(u,0)+1
        def dec(u,v):
            a[u][v]-=1;
            if not a[u][v]: del a[u][v]
            a[v][u]-=1;
            if not a[v][u]: del a[v][u]
        dec(s1,o1); dec(s2,o2); inc(s1,o2); inc(s2,o1)

    def _check_delta(self, n, edges, s1, o1, s2, o2):
        a_before = _adj(n, edges)
        before = self._full_count(a_before)

        delta = _motif4_delta(a_before, s1, o1, s2, o2)

        a_after = _adj(n, edges)
        self._apply_swap(a_after, s1, o1, s2, o2)
        after = self._full_count(a_after)

        all_keys = set(before) | set(after) | set(delta)
        for ds in all_keys:
            expected = after.get(ds, 0) - before.get(ds, 0)
            got = delta.get(ds, 0)
            self.assertEqual(
                got, expected,
                f"delta mismatch for {ds}: expected {expected}, got {got} "
                f"(before={before}, after={after})",
            )

    def test_swap_breaks_c4(self):
        # C4: 0-1-2-3-0, swap (0,1) and (2,3) → path 0-3-2-1 (no c4)
        self._check_delta(4, [(0,1),(1,2),(2,3),(0,3)], 0, 1, 2, 3)

    def test_swap_creates_triangle(self):
        # Path 0-1-2-3, swap (1,2) and (0,3) → 0-2, 1-3 — changes motif landscape
        self._check_delta(4, [(0,1),(1,2),(2,3),(0,3)], 1, 2, 0, 3)

    def test_swap_k4_to_diamond(self):
        # K4 minus edge 2-3, swap (0,2) with (1,3) → 0-3, 1-2 added (still K4 minus something)
        edges = [(0,1),(0,2),(0,3),(1,2),(1,3)]
        self._check_delta(5, edges + [(0,4),(1,4)], 0, 2, 1, 3)

    def test_zero_delta_when_no_change(self):
        # Swap creates edges that were already there conceptually; use disjoint components
        edges = [(0,1),(2,3),(1,2),(0,3)]  # c4
        self._check_delta(4, edges, 0, 1, 2, 3)


if __name__ == "__main__":
    unittest.main()
