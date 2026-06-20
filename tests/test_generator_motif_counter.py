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
from itertools import combinations

from motif_counter._common import _count_motifs4_through_edge
from generator.local_updates import (
    _motif4_delta,
    _triangle_node_delta,
    _induced_cycles_through_nodes,
    _cycle_delta,
)


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

    def test_random_fuzz(self):
        # Random simple graphs + random valid double-edge swaps; the incremental
        # 4-node motif delta must match the brute-force before/after recount.
        rng = np.random.default_rng(98765)
        for _ in range(300):
            n = int(rng.integers(6, 11))
            edges = [
                (u, v)
                for u in range(n) for v in range(u + 1, n)
                if rng.random() < 0.35
            ]
            if len(edges) < 2:
                continue
            i, j = rng.choice(len(edges), size=2, replace=False)
            s1, o1 = edges[i]
            s2, o2 = edges[j]
            if rng.random() < 0.5:
                s2, o2 = o2, s2
            if len({s1, o1, s2, o2}) < 4:
                continue
            new1 = (min(s1, o2), max(s1, o2))
            new2 = (min(s2, o1), max(s2, o1))
            eset = {(min(a, b), max(a, b)) for a, b in edges}
            if new1 in eset or new2 in eset or new1 == new2:
                continue
            self._check_delta(n, edges, s1, o1, s2, o2)


# ── triangle delta ────────────────────────────────────────────────────────────

def _brute_tri_counts(a: list[dict], n: int) -> tuple[int, dict[int, int]]:
    """Brute-force (total triangles, per-node triangle counts) on adj membership."""
    per_node: dict[int, int] = {v: 0 for v in range(n)}
    total = 0
    for u in range(n):
        nbrs = [w for w in a[u] if w > u]
        for ai in range(len(nbrs)):
            for bi in range(ai + 1, len(nbrs)):
                w, x = nbrs[ai], nbrs[bi]
                if x in a[w]:           # u<w<x triangle
                    total += 1
                    per_node[u] += 1
                    per_node[w] += 1
                    per_node[x] += 1
    return total, per_node


class TestTriangleDelta(unittest.TestCase):
    """_triangle_node_delta must match brute-force triangle recount (aggregate + per node)."""

    def _apply_swap(self, a: list[dict], s1, o1, s2, o2) -> None:
        def inc(u, v): a[u][v] = a[u].get(v, 0) + 1; a[v][u] = a[v].get(u, 0) + 1
        def dec(u, v):
            a[u][v] -= 1
            if not a[u][v]: del a[u][v]
            a[v][u] -= 1
            if not a[v][u]: del a[v][u]
        dec(s1, o1); dec(s2, o2); inc(s1, o2); inc(s2, o1)

    def _check(self, n, edges, s1, o1, s2, o2):
        a = _adj(n, edges)
        t_before, pn_before = _brute_tri_counts(a, n)

        dT, node_delta = _triangle_node_delta(a, s1, o1, s2, o2)

        a_after = _adj(n, edges)
        self._apply_swap(a_after, s1, o1, s2, o2)
        t_after, pn_after = _brute_tri_counts(a_after, n)

        self.assertEqual(dT, t_after - t_before, "aggregate ΔT mismatch")
        for v in range(n):
            self.assertEqual(
                node_delta.get(v, 0), pn_after[v] - pn_before[v],
                f"per-node Δt mismatch at {v}",
            )

    def test_close_triangle(self):
        # Wedge 1-0-2 plus stubs; swap to create edge 1-2 closing a triangle
        self._check(5, [(0,1),(0,2),(1,3),(2,4)], 1,3, 2,4)

    def test_random_fuzz(self):
        rng = np.random.default_rng(2468)
        for _ in range(300):
            n = int(rng.integers(5, 10))
            edges = [
                (u, v)
                for u in range(n) for v in range(u + 1, n)
                if rng.random() < 0.4
            ]
            if len(edges) < 2:
                continue
            i, j = rng.choice(len(edges), size=2, replace=False)
            s1, o1 = edges[i]
            s2, o2 = edges[j]
            if rng.random() < 0.5:
                s2, o2 = o2, s2
            if len({s1, o1, s2, o2}) < 4:
                continue
            new1 = (min(s1, o2), max(s1, o2))
            new2 = (min(s2, o1), max(s2, o1))
            eset = {(min(a, b), max(a, b)) for a, b in edges}
            if new1 in eset or new2 in eset or new1 == new2:
                continue
            self._check(n, edges, s1, o1, s2, o2)


# ── induced-cycle delta ───────────────────────────────────────────────────────

def _brute_induced_cycles(a: list[dict], k: int) -> int:
    """Count induced (chordless) k-cycles by brute force over all k-subsets.

    A k-subset induces a k-cycle iff every vertex has exactly 2 neighbours
    inside the subset (2-regular) and the subset is connected (a single cycle,
    not e.g. two disjoint triangles for k=6).
    """
    n = len(a)
    count = 0
    for verts in combinations(range(n), k):
        vs = set(verts)
        degs = {v: sum(1 for u in vs if u != v and u in a[v]) for v in vs}
        if any(d != 2 for d in degs.values()):
            continue
        # connectivity check via BFS within the subset
        start = verts[0]
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in a[cur]:
                if nb in vs and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(seen) == k:
            count += 1
    return count


class TestInducedCyclesThroughNodes(unittest.TestCase):

    def test_single_c5(self):
        a = _adj(5, [(0,1),(1,2),(2,3),(3,4),(0,4)])
        cycles = _induced_cycles_through_nodes(a, range(5), 5)
        self.assertEqual(cycles, {frozenset({0,1,2,3,4})})

    def test_chord_destroys_induced(self):
        # C5 with a chord 0-2 — no induced 5-cycle anymore
        a = _adj(5, [(0,1),(1,2),(2,3),(3,4),(0,4),(0,2)])
        self.assertEqual(_induced_cycles_through_nodes(a, range(5), 5), set())

    def test_c6_not_two_triangles(self):
        # Two disjoint triangles: 2-regular on 6 nodes but NOT a 6-cycle
        a = _adj(6, [(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)])
        self.assertEqual(_induced_cycles_through_nodes(a, range(6), 6), set())

    def test_anchor_filter(self):
        # C5 on {0..4} plus isolated edge; anchoring on node 7 finds nothing
        a = _adj(8, [(0,1),(1,2),(2,3),(3,4),(0,4),(6,7)])
        self.assertEqual(_induced_cycles_through_nodes(a, [7], 5), set())
        self.assertEqual(_induced_cycles_through_nodes(a, [0], 5),
                         {frozenset({0,1,2,3,4})})


class TestCycleDelta(unittest.TestCase):
    """_cycle_delta should match the difference of full induced-cycle counts."""

    def _apply_swap(self, a: list[dict], s1, o1, s2, o2) -> None:
        def inc(u, v): a[u][v] = a[u].get(v, 0) + 1; a[v][u] = a[v].get(u, 0) + 1
        def dec(u, v):
            a[u][v] -= 1
            if not a[u][v]: del a[u][v]
            a[v][u] -= 1
            if not a[v][u]: del a[v][u]
        dec(s1, o1); dec(s2, o2); inc(s1, o2); inc(s2, o1)

    def _check(self, n, edges, s1, o1, s2, o2):
        a_before = _adj(n, edges)
        c5_b, c6_b = _brute_induced_cycles(a_before, 5), _brute_induced_cycles(a_before, 6)

        dc5, dc6 = _cycle_delta(a_before, s1, o1, s2, o2)

        a_after = _adj(n, edges)
        self._apply_swap(a_after, s1, o1, s2, o2)
        c5_a, c6_a = _brute_induced_cycles(a_after, 5), _brute_induced_cycles(a_after, 6)

        self.assertEqual(dc5, c5_a - c5_b, f"Δc5 mismatch (before={c5_b}, after={c5_a})")
        self.assertEqual(dc6, c6_a - c6_b, f"Δc6 mismatch (before={c6_b}, after={c6_a})")

    def test_create_c5_by_closing(self):
        # Path 0-1-2-3-4 plus edge 0-5, 4-5; swap to close 0-1-2-3-4-0
        # Start: edges form an open structure; swap creates the closing edge.
        self._check(6, [(0,1),(1,2),(2,3),(3,4),(0,5),(4,5)], 0, 5, 4, 5)

    def test_break_c5(self):
        # C5 0-1-2-3-4-0 with spare edges to swap with; remove a cycle edge
        self._check(7, [(0,1),(1,2),(2,3),(3,4),(0,4),(5,6)], 0,1, 5,6)

    def test_chord_added_destroys_c5(self):
        # C5 + two extra nodes; a swap that drops chord 0-2 onto the cycle
        self._check(7, [(0,1),(1,2),(2,3),(3,4),(0,4),(0,5),(2,6)], 0,5, 2,6)

    def test_c6_create(self):
        # Path 0-1-2-3-4-5 (missing closing 0-5) plus stubs (0,6) and (7,5).
        # Swap (0,6)↔(7,5) → adds (0,5) [closes the induced C6] and (7,6).
        self._check(8, [(0,1),(1,2),(2,3),(3,4),(4,5),(0,6),(7,5)], 0,6, 7,5)

    def test_no_change(self):
        # Swap in a region with no 5/6-cycles
        self._check(6, [(0,1),(2,3),(1,2),(0,3),(4,5)], 0,1, 4,5)

    def test_random_fuzz(self):
        # Random simple graphs + random valid double-edge swaps; the incremental
        # delta must match the brute-force induced-cycle recount every time.
        rng = np.random.default_rng(12345)
        for _ in range(300):
            n = int(rng.integers(7, 11))
            # random simple edge set
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.32:
                        edges.append((u, v))
            if len(edges) < 2:
                continue
            i, j = rng.choice(len(edges), size=2, replace=False)
            s1, o1 = edges[i]
            s2, o2 = edges[j]
            # orient endpoints randomly to exercise both pairings
            if rng.random() < 0.5:
                s2, o2 = o2, s2
            # skip swaps that would create a self-loop or a duplicate edge
            if len({s1, o1, s2, o2}) < 4:
                continue
            new1 = (min(s1, o2), max(s1, o2))
            new2 = (min(s2, o1), max(s2, o1))
            eset = {(min(a, b), max(a, b)) for a, b in edges}
            if new1 in eset or new2 in eset or new1 == new2:
                continue
            self._check(n, edges, s1, o1, s2, o2)


if __name__ == "__main__":
    unittest.main()
