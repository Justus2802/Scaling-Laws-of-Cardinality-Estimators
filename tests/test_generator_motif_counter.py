"""Tests for generator.motif_counter — ExactMotifCounter and helpers."""

import itertools
import unittest

import igraph
import numpy as np

from kgsynth.motif_counter import (
    CCMotifCounter,
    ExactMotifCounter,
    cc_run_stars,
    cc_run_stars_loop,
)

from kgsynth.motif_counter._common import _count_motifs4_through_edge
from kgsynth.generator.local_updates import (
    _motif4_delta,
    _triangle_node_delta,
    _induced_cycles_through_pair,
    _induced_cycles_through_pair_mitm,
    _cycle_delta,
)

# Shared brute-force oracles (see tests/_brute_motifs.py). Aliased to the
# underscore-prefixed names this module has always used.
from _brute_motifs import (
    und as _und,
    adj as _adj,
    brute_tri_counts as _brute_tri_counts,
    brute_induced_cycles as _brute_induced_cycles,
)


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
        def inc(u,v):
            a[u][v]=a[u].get(v,0)+1
            a[v][u]=a[v].get(u,0)+1
        def dec(u,v):
            a[u][v]-=1
            if not a[u][v]:
                del a[u][v]
            a[v][u]-=1
            if not a[v][u]:
                del a[v][u]
        dec(s1,o1)
        dec(s2,o2)
        inc(s1,o2)
        inc(s2,o1)

    def _check_delta(self, n, edges, s1, o1, s2, o2, types=None):
        a_before = _adj(n, edges)
        before = self._full_count(a_before)

        kwargs = {} if types is None else {"types": types}
        delta = _motif4_delta(a_before, s1, o1, s2, o2, **kwargs)

        a_after = _adj(n, edges)
        self._apply_swap(a_after, s1, o1, s2, o2)
        after = self._full_count(a_after)

        # When a restricted `types` set is requested, only those types are checked.
        all_keys = set(before) | set(after) | set(delta)
        if types is not None:
            all_keys &= types
        for ds in all_keys:
            expected = after.get(ds, 0) - before.get(ds, 0)
            got = delta.get(ds, 0)
            self.assertEqual(
                got, expected,
                f"delta mismatch for {ds} (types={types}): expected {expected}, "
                f"got {got} (before={before}, after={after})",
            )
        # The restricted delta must never report a type outside `types`.
        if types is not None:
            self.assertTrue(set(delta) <= types, f"delta leaked types outside {types}: {delta}")

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

    def test_random_fuzz_fast_path(self):
        # Same fuzz, but request only C4/diamond/K4 (paw excluded) → exercises the
        # fast O(Δ²) pair-enumeration path; must match brute force for those types.
        fast_types = frozenset({(2, 2, 2, 2), (2, 2, 3, 3), (3, 3, 3, 3)})
        rng = np.random.default_rng(13579)
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
            self._check_delta(n, edges, s1, o1, s2, o2, types=fast_types)


# ── triangle delta ────────────────────────────────────────────────────────────

class TestTriangleDelta(unittest.TestCase):
    """_triangle_node_delta must match brute-force triangle recount (aggregate + per node)."""

    def _apply_swap(self, a: list[dict], s1, o1, s2, o2) -> None:
        def inc(u, v):
            a[u][v] = a[u].get(v, 0) + 1
            a[v][u] = a[v].get(u, 0) + 1
        def dec(u, v):
            a[u][v] -= 1
            if not a[u][v]:
                del a[u][v]
            a[v][u] -= 1
            if not a[v][u]:
                del a[v][u]
        dec(s1, o1)
        dec(s2, o2)
        inc(s1, o2)
        inc(s2, o1)

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

class TestInducedCyclesThroughPair(unittest.TestCase):

    def test_single_c5_adjacent_pair(self):
        # C5 0-1-2-3-4-0; pair (0,1) is a cycle edge
        a = _adj(5, [(0,1),(1,2),(2,3),(3,4),(0,4)])
        self.assertEqual(_induced_cycles_through_pair(a, 0, 1, 5),
                         {frozenset({0,1,2,3,4})})

    def test_single_c5_nonadjacent_pair(self):
        # Same C5; pair (0,2) are non-adjacent cycle vertices (two arcs 0-1-2 and 0-4-3-2)
        a = _adj(5, [(0,1),(1,2),(2,3),(3,4),(0,4)])
        self.assertEqual(_induced_cycles_through_pair(a, 0, 2, 5),
                         {frozenset({0,1,2,3,4})})

    def test_chord_destroys_induced(self):
        # C5 with a chord 0-2 — no induced 5-cycle through any pair
        a = _adj(5, [(0,1),(1,2),(2,3),(3,4),(0,4),(0,2)])
        self.assertEqual(_induced_cycles_through_pair(a, 0, 1, 5), set())
        self.assertEqual(_induced_cycles_through_pair(a, 3, 4, 5), set())

    def test_c6_not_two_triangles(self):
        # Two disjoint triangles: 2-regular on 6 nodes but NOT a 6-cycle
        a = _adj(6, [(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)])
        self.assertEqual(_induced_cycles_through_pair(a, 0, 1, 6), set())

    def test_pair_not_in_cycle(self):
        # C5 on {0..4} plus isolated edge 6-7; no induced 5-cycle through (6,7)
        a = _adj(8, [(0,1),(1,2),(2,3),(3,4),(0,4),(6,7)])
        self.assertEqual(_induced_cycles_through_pair(a, 6, 7, 5), set())

    def test_mitm_matches_dfs_and_oracle(self):
        # The anchored meet-in-the-middle enumerator (the default behind
        # _cycle_delta) must return exactly the same induced-cycle sets as the
        # recursive DFS, and both must match the brute-force oracle filtered to
        # subsets containing the pair.
        rng = np.random.default_rng(2024)
        from _brute_motifs import _subset_connected

        def _oracle_pair(a, x, y, k):
            n = len(a)
            out = set()
            for verts in itertools.combinations(range(n), k):
                vs = set(verts)
                if x not in vs or y not in vs:
                    continue
                if any(sum(1 for u in vs if u != v and u in a[v]) != 2 for v in vs):
                    continue
                if _subset_connected(a, verts, vs):
                    out.add(frozenset(vs))
            return out

        for _ in range(200):
            n = int(rng.integers(6, 12))
            a = _adj(n, [(u, v) for u in range(n) for v in range(u + 1, n)
                         if rng.random() < 0.35])
            x, y = int(rng.integers(0, n)), int(rng.integers(0, n))
            if x == y:
                continue
            for k in (5, 6):
                dfs = _induced_cycles_through_pair(a, x, y, k)
                mitm = _induced_cycles_through_pair_mitm(a, x, y, k)
                self.assertEqual(mitm, dfs, f"MITM≠DFS n={n} k={k} pair=({x},{y})")
                self.assertEqual(mitm, _oracle_pair(a, x, y, k),
                                 f"MITM≠oracle n={n} k={k} pair=({x},{y})")


class TestCycleDelta(unittest.TestCase):
    """_cycle_delta should match the difference of full induced-cycle counts."""

    def _apply_swap(self, a: list[dict], s1, o1, s2, o2) -> None:
        def inc(u, v):
            a[u][v] = a[u].get(v, 0) + 1
            a[v][u] = a[v].get(u, 0) + 1
        def dec(u, v):
            a[u][v] -= 1
            if not a[u][v]:
                del a[u][v]
            a[v][u] -= 1
            if not a[v][u]:
                del a[v][u]
        dec(s1, o1)
        dec(s2, o2)
        inc(s1, o2)
        inc(s2, o1)

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


# ── CC star estimator (cc_run_stars / cc_run_stars_loop) ──────────────────────

class TestCCStars(unittest.TestCase):
    """The CC star estimators must agree with the exact induced-star counts.

    These estimators are unbiased Monte-Carlo samplers, so a single run is noisy.
    Statistical assertions therefore average each per-k estimate over several
    independent seeds (the mean of an unbiased estimator converges to the true
    count) and assert only where the exact count is large enough for the relative
    error to be meaningful.

    Validation strategy: the vectorised ``cc_run_stars`` (the production path) is
    checked thoroughly against the exact oracle with a generous sampling budget,
    since it is fast. The un-vectorised ``cc_run_stars_loop`` is the same
    estimator but ~orders slower, so it is validated with a small budget on a
    tiny graph (correctness vs exact) and against ``cc_run_stars`` directly
    (equivalence) — running it at the thorough budget would time out.
    """

    _IMPLS = (("vec", cc_run_stars), ("loop", cc_run_stars_loop))
    _VEC = (("vec", cc_run_stars),)

    def _mean_estimate(self, fn, g, *, seeds=16, n_colorings=12, n_samples=10000):
        """Per-k mean estimate over ``seeds`` independent runs of ``fn``."""
        acc: dict[int, float] = {k: 0.0 for k in range(2, 11)}
        for s in range(seeds):
            est = fn(g, n_samples, np.random.default_rng(1000 + s),
                     n_colorings=n_colorings)
            for k in range(2, 11):
                acc[k] += est.get(k, 0)
        return {k: acc[k] / seeds for k in acc}

    def _assert_close_to_exact(self, g, *, min_count, rel_tol, impls=None,
                               ks=None, **kw):
        """Selected impls' mean estimates must match exact within ``rel_tol``.

        Only star sizes whose exact count is ≥ ``min_count`` are asserted (rarer
        sizes are too noisy for a meaningful relative-error bound); ``ks``, if
        given, further restricts to those sizes. ``impls`` defaults to both.
        """
        exact = _EXACT.count_stars(g)
        impls = impls or self._IMPLS
        sizes = ks if ks is not None else range(2, 11)
        for label, fn in impls:
            mean = self._mean_estimate(fn, g, **kw)
            for k in sizes:
                e = exact.get(k, 0)
                if e < min_count:
                    continue
                rel = abs(mean[k] - e) / e
                self.assertLess(
                    rel, rel_tol,
                    f"[{label}] k={k}: mean={mean[k]:.1f} exact={e} rel={rel:.3f}",
                )

    # ── degenerate graphs (deterministic, exact) ─────────────────────────────

    def test_empty_graph(self):
        for label, fn in self._IMPLS:
            res = fn(_und(0, []), 1000, np.random.default_rng(0), n_colorings=2)
            self.assertEqual(res, {k: 0 for k in range(2, 11)}, label)

    def test_no_edges(self):
        for label, fn in self._IMPLS:
            res = fn(_und(5, []), 1000, np.random.default_rng(0), n_colorings=2)
            self.assertEqual(res, {k: 0 for k in range(2, 11)}, label)

    def test_single_edge(self):
        # One edge → no stars (need ≥2 leaves).
        for label, fn in self._IMPLS:
            res = fn(_und(2, [(0, 1)]), 1000, np.random.default_rng(0), n_colorings=2)
            self.assertEqual({k: res.get(k, 0) for k in range(2, 11)},
                             {k: 0 for k in range(2, 11)}, label)

    # ── triangle-free star hub: closed form C(d, k) ──────────────────────────

    def test_star_hub_closed_form(self):
        # Single hub of degree d (triangle-free) → induced k-stars = C(d, k).
        # Assert (vec) on sizes where C(d, k) is large enough to be a tight target.
        from math import comb
        d = 8
        g = _und(d + 1, [(0, i) for i in range(1, d + 1)])
        self._assert_close_to_exact(g, min_count=40, rel_tol=0.20,
                                    impls=self._VEC, seeds=24, n_colorings=32)
        # Sanity: the closed form is what the exact oracle reports.
        exact = _EXACT.count_stars(g)
        self.assertEqual({k: exact[k] for k in range(2, d + 1)},
                         {k: comb(d, k) for k in range(2, d + 1)})

    # ── leaf-leaf edge must be rejected (induced, not raw, star) ──────────────

    def test_leaf_leaf_edge_rejected(self):
        # Hub 0 with many leaves plus one leaf-leaf chord. Any star subset that
        # includes BOTH chord endpoints is not an induced star, so the induced
        # counts are strictly below C(d, k). Compare the vec mean estimate to the
        # exact oracle, which already excludes the chorded subsets.
        d = 9
        edges = [(0, i) for i in range(1, d + 1)] + [(1, 2)]  # chord 1-2
        g = _und(d + 1, edges)
        exact = _EXACT.count_stars(g)
        # The chord must actually reduce some count below the chord-free C(d,k).
        from math import comb
        self.assertLess(exact[3], comb(d, 3))
        self._assert_close_to_exact(g, min_count=40, rel_tol=0.20,
                                    impls=self._VEC, seeds=24, n_colorings=32)

    # ── statistical agreement with exact on random graphs (vectorised) ────────

    def test_random_graphs_agree_with_exact(self):
        # Heavier graphs validate the vectorised estimator on non-trivial
        # structure. Run vec only — the un-vectorised loop is ~orders slower here
        # (its equivalence to vec is covered by the small-graph tests above).
        rng = np.random.default_rng(2024)
        for trial in range(3):
            n = int(rng.integers(14, 20))
            edges = [
                (u, v)
                for u in range(n) for v in range(u + 1, n)
                if rng.random() < 0.28
            ]
            g = _und(n, edges)
            # Mean over seeds tames variance; only assert on abundant sizes.
            self._assert_close_to_exact(g, min_count=40, rel_tol=0.20,
                                        impls=self._VEC, seeds=24, n_colorings=24)

    # ── reference (loop) implementation: correctness + equivalence ────────────

    def test_loop_matches_exact_small(self):
        # The un-vectorised reference is slow, so validate it cheaply on a tiny
        # hub, asserting only the low-variance 2-/3-stars (high colourful prob →
        # fast convergence; larger k needs far more samples than the loop affords).
        d = 6
        g = _und(d + 1, [(0, i) for i in range(1, d + 1)])  # C(6,2)=15, C(6,3)=20
        self._assert_close_to_exact(
            g, min_count=15, rel_tol=0.25, ks=(2, 3),
            impls=(("loop", cc_run_stars_loop),), seeds=8, n_colorings=8,
        )

    def test_vec_matches_loop(self):
        # Same estimator → same distribution: with an equal (small) budget the
        # two implementations' mean estimates must agree with each other on the
        # low-variance 2-/3-stars. Confirms the vectorised rewrite is faithful.
        d = 6
        g = _und(d + 1, [(0, i) for i in range(1, d + 1)])
        exact = _EXACT.count_stars(g)
        kw = dict(seeds=10, n_colorings=8)
        vec_mean = self._mean_estimate(cc_run_stars, g, **kw)
        loop_mean = self._mean_estimate(cc_run_stars_loop, g, **kw)
        for k in (2, 3):
            diff = abs(vec_mean[k] - loop_mean[k]) / exact[k]
            self.assertLess(
                diff, 0.25,
                f"k={k}: vec={vec_mean[k]:.1f} loop={loop_mean[k]:.1f} "
                f"exact={exact[k]} reldiff={diff:.3f}",
            )


class TestEscapeExactCyclesVsBrute(unittest.TestCase):
    """The ESCAPE enumerator (ExactMotifCounter k=5/6) must match the brute oracle.

    Validates both the pre-existing c5 path and the new c6 generalization against
    the independent ``_brute_induced_cycles`` ground truth — small graphs never
    trip ``_ESCAPE_MAX_DEGREE``, so the exact path is always exercised.
    """

    C5_DS = ExactMotifCounter.C5_DS
    C6_DS = ExactMotifCounter.C6_DS

    def _exact_cycles(self, n, edges):
        g = _und(n, edges)
        c5 = _EXACT.count_motifsk(g, 5).get(self.C5_DS, 0)
        c6 = _EXACT.count_motifsk(g, 6).get(self.C6_DS, 0)
        return c5, c6

    def test_single_c6(self):
        # One induced 6-cycle → exactly 1; no 5-cycle.
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
        self.assertEqual(self._exact_cycles(6, edges), (0, 1))

    def test_chorded_c6_not_induced(self):
        # 6-cycle with a chord 0-3 → no induced 6-cycle.
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 3)]
        self.assertEqual(self._exact_cycles(6, edges)[1], 0)

    def test_two_triangles_not_c6(self):
        # Two disjoint triangles: 2-regular on 6 nodes but NOT a 6-cycle.
        edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
        self.assertEqual(self._exact_cycles(6, edges)[1], 0)

    def test_single_c5(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
        self.assertEqual(self._exact_cycles(5, edges), (1, 0))

    def test_random_fuzz(self):
        # Same random-graph generator as TestCycleDelta.test_random_fuzz; assert
        # the exact ESCAPE c5/c6 counts equal the brute-force induced-cycle oracle.
        rng = np.random.default_rng(54321)
        for _ in range(300):
            n = int(rng.integers(6, 12))
            edges = [
                (u, v)
                for u in range(n)
                for v in range(u + 1, n)
                if rng.random() < 0.32
            ]
            a = _adj(n, edges)
            c5_exact, c6_exact = self._exact_cycles(n, edges)
            self.assertEqual(c5_exact, _brute_induced_cycles(a, 5),
                             f"c5 mismatch on n={n} edges={edges}")
            self.assertEqual(c6_exact, _brute_induced_cycles(a, 6),
                             f"c6 mismatch on n={n} edges={edges}")


if __name__ == "__main__":
    unittest.main()
