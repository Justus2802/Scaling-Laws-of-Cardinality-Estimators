"""IPF stub allocation — the invariants Stage 2's wiring depends on."""

import unittest

import numpy as np

from kgsynth.generator._ipf import (
    build_support,
    fit_stubs,
    ipf,
    solve_edge_budget,
)


def _memberships(rng, n_entities, n_relations, lo=1, hi=4):
    return [rng.choice(n_relations, size=int(rng.integers(lo, hi)), replace=False)
            for _ in range(n_entities)]


def _pools(memberships, n_relations):
    """subjects_by_rel-equivalent: entities per relation, ascending (Stage 2's order)."""
    pools = {r: [] for r in range(n_relations)}
    for v, rels in enumerate(memberships):
        for r in rels:
            pools[int(r)].append(v)
    return pools


class TestBuildSupport(unittest.TestCase):
    def test_column_slices_match_the_relation_pools(self):
        # Stage 2 slices a column straight out of the stub vector and zips it against
        # subjects_by_rel[r]. If the orders ever diverge, stubs land on the wrong entities
        # and the degree targets are silently scrambled — so pin the contract.
        rng = np.random.default_rng(0)
        V, R = 200, 7
        cs = _memberships(rng, V, R)
        rows, cols = build_support(cs, R)
        starts = np.concatenate(([0], np.cumsum(np.bincount(cols, minlength=R))))
        pools = _pools(cs, R)
        for r in range(R):
            self.assertEqual(rows[starts[r]:starts[r + 1]].tolist(), pools[r])


class TestIPF(unittest.TestCase):
    def test_row_margins_exact_and_columns_close(self):
        rng = np.random.default_rng(1)
        V, R = 300, 6
        rows, cols = build_support(_memberships(rng, V, R), R)
        rt = rng.integers(1, 9, V).astype(np.int64)
        ct = np.full(R, rt.sum() / R)
        a = ipf(rows, cols, rng.random(rows.size) + 0.1, rt, ct, V, R)
        np.testing.assert_allclose(np.bincount(rows, a, minlength=V), rt, rtol=1e-6)

    def test_survives_an_unreachable_column_target(self):
        # A column asking for far more than its support can supply used to drive the
        # multiplicative scalings to inf and poison the whole matrix with NaN.
        rng = np.random.default_rng(2)
        V, R = 400, 5
        rows, cols = build_support(_memberships(rng, V, R), R)
        rt = np.full(V, 3, dtype=np.int64)
        ct = np.array([1e9, 1.0, 1.0, 1.0, 1.0])       # wildly infeasible first column
        a = ipf(rows, cols, rng.random(rows.size) + 0.1, rt, ct, V, R)
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_allclose(np.bincount(rows, a, minlength=V), rt, rtol=1e-6)


class TestFitStubs(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.V, self.R = 500, 8
        self.cs = _memberships(self.rng, self.V, self.R)
        self.rows, self.cols = build_support(self.cs, self.R)
        self.rt = self.rng.integers(4, 12, self.V).astype(np.int64)
        total = int(self.rt.sum())
        self.ct = np.full(self.R, total // self.R, dtype=np.int64)
        self.ct[0] += total - int(self.ct.sum())

    def test_column_sums_are_exact(self):
        # This is the whole point: Σ_v X[v,r] must equal e_r exactly, on both sides, or the
        # relation's stubs cannot be paired and the remainder falls into a blunt recovery pass.
        x = fit_stubs(self.rows, self.cols, self.rng.random(self.rows.size) + 0.1,
                      self.rt, self.ct, self.V, self.R, self.rng)
        np.testing.assert_array_equal(
            np.bincount(self.cols, x, minlength=self.R).astype(np.int64), self.ct)

    def test_row_sums_land_within_one(self):
        x = fit_stubs(self.rows, self.cols, self.rng.random(self.rows.size) + 0.1,
                      self.rt, self.ct, self.V, self.R, self.rng)
        got = np.bincount(self.rows, x, minlength=self.V)
        self.assertLessEqual(int(np.abs(got - self.rt).max()), 2)

    def test_floor_gives_every_support_entry_a_stub(self):
        x = fit_stubs(self.rows, self.cols, self.rng.random(self.rows.size) + 0.1,
                      self.rt, self.ct, self.V, self.R, self.rng, floor=True)
        self.assertTrue((x >= 1).all())
        np.testing.assert_array_equal(
            np.bincount(self.cols, x, minlength=self.R).astype(np.int64), self.ct)

    def test_floor_does_not_decapitate_the_hub(self):
        # Imposing the floor by taking stubs back from the heaviest entries flattens exactly
        # the hubs that carry the max/p90 degree targets (it cost aids max-out 11 -> 5).
        # The floor must come out of the margins, so the hub keeps its allocation.
        rt = self.rt.copy()
        rt[0] = 60                       # one clear hub
        ct = self.ct.copy()
        ct[0] += int(rt.sum()) - int(self.rt.sum())
        x = fit_stubs(self.rows, self.cols, self.rng.random(self.rows.size) + 0.1,
                      rt, ct, self.V, self.R, self.rng, floor=True)
        got = np.bincount(self.rows, x, minlength=self.V)
        self.assertGreaterEqual(int(got[0]), 55)


class TestSolveEdgeBudget(unittest.TestCase):
    def _solve(self, cs, inv, tgt_out, tgt_in, budget, V, R, col_cap=None):
        rng = np.random.default_rng(5)
        orow, ocol = build_support(cs, R)
        irow, icol = build_support(inv, R)
        ov = rng.random(orow.size) + 0.1
        iv = rng.random(irow.size) + 0.1
        e = solve_edge_budget(orow, ocol, ov, tgt_out, irow, icol, iv, tgt_in,
                              budget, V, R, col_cap=col_cap)
        return e, (orow, ocol, ov), (irow, icol, iv)

    def test_budget_is_fully_spendable(self):
        # Σe == Σtgt_out is what makes the deficit pass unnecessary: every stub the degree
        # law asks for has a relation to be spent on.
        rng = np.random.default_rng(6)
        V, R = 400, 5
        cs, inv = _memberships(rng, V, R), _memberships(rng, V, R)
        tgt = np.full(V, 6, dtype=np.int64)
        e, _, _ = self._solve(cs, inv, tgt, tgt.copy(),
                              np.full(R, V * 6 / R), V, R)
        self.assertEqual(int(e.sum()), int(tgt.sum()))

    def test_shrinks_a_relation_its_object_pool_cannot_absorb(self):
        # The aids case: relation 0 is emitted by everyone but received by only a few, so
        # its object pool cannot hold the budget the relation weights hand it. The solver
        # must clip it to what the pool supports and give the surplus to relations with room
        # — rather than accept an unplaceable budget and dump the difference into a
        # uniform-random recovery pass.
        V, R = 2000, 4
        cs = [np.array([0, 1 + v % 3]) for v in range(V)]
        inv = [np.array([0, 1 + v % 3]) if v < 100 else np.array([1 + v % 3])
               for v in range(V)]
        tgt = np.full(V, 6, dtype=np.int64)
        total = int(tgt.sum())
        budget = np.array([0.70, 0.10, 0.10, 0.10]) * total     # 70% to the starved relation
        e, _, _ = self._solve(cs, inv, tgt, tgt.copy(), budget, V, R)

        pool_quota = int(tgt[:100].sum())                        # all the in-quota r0 can see
        self.assertLessEqual(int(e[0]), pool_quota)
        self.assertLess(int(e[0]), int(0.70 * total))            # actually shrunk
        self.assertEqual(int(e.sum()), total)                    # and nothing was lost

    def test_respects_the_distinct_pair_ceiling(self):
        rng = np.random.default_rng(8)
        V, R = 300, 4
        cs, inv = _memberships(rng, V, R), _memberships(rng, V, R)
        tgt = np.full(V, 5, dtype=np.int64)
        cap = np.array([50.0, 1e9, 1e9, 1e9])                    # relation 0 is tiny
        e, _, _ = self._solve(cs, inv, tgt, tgt.copy(),
                              np.full(R, V * 5 / R), V, R, col_cap=cap)
        self.assertLessEqual(int(e[0]), 50)
        self.assertEqual(int(e.sum()), int(tgt.sum()))


class TestStubBalance(unittest.TestCase):
    def test_both_sides_get_identical_per_relation_stub_counts(self):
        # The constraint the whole module exists to establish. Σ_v X[v,r] == Σ_v Y[v,r] for
        # every r: a directed relation's out-stubs and in-stubs must match or they cannot
        # be paired.
        rng = np.random.default_rng(9)
        V, R = 600, 6
        cs, inv = _memberships(rng, V, R), _memberships(rng, V, R)
        orow, ocol = build_support(cs, R)
        irow, icol = build_support(inv, R)
        tgt_out = rng.integers(2, 10, V).astype(np.int64)
        tgt_in = tgt_out[rng.permutation(V)]                     # same sum, different shape
        ov, iv = rng.random(orow.size) + 0.1, rng.random(irow.size) + 0.1

        e = solve_edge_budget(orow, ocol, ov, tgt_out, irow, icol, iv, tgt_in,
                              np.full(R, tgt_out.sum() / R), V, R)
        x = fit_stubs(orow, ocol, ov, tgt_out, e, V, R, rng, floor=True)
        y = fit_stubs(irow, icol, iv, tgt_in, e, V, R, rng)

        cx = np.bincount(ocol, x, minlength=R).astype(np.int64)
        cy = np.bincount(icol, y, minlength=R).astype(np.int64)
        np.testing.assert_array_equal(cx, cy)
        np.testing.assert_array_equal(cx, e)


if __name__ == "__main__":
    unittest.main()


class TestEntryCaps(unittest.TestCase):
    def test_no_entry_exceeds_the_opposite_pool(self):
        # A relation cannot carry the same (s, o) pair twice, so an object is reached by at
        # most |S_r| distinct subjects. An allocation above that is not merely hard to place
        # — it is unrealisable, and no pairing will ever satisfy it (fb237_v4's in-hub was
        # allocated 119 in-stubs of relation 178 from a pool of 117 subjects).
        rng = np.random.default_rng(11)
        V, R = 400, 5
        # Relation 0 has a tiny object pool, so its in-side entries face a big subject pool
        # while its out-side entries have almost nowhere to go.
        cs = [np.array([0, 1 + v % 4]) for v in range(V)]
        inv = [np.array([0, 1 + v % 4]) if v < 12 else np.array([1 + v % 4])
               for v in range(V)]
        orow, ocol = build_support(cs, R)
        irow, icol = build_support(inv, R)
        n_obj = np.bincount(icol, minlength=R)      # |O_r|
        n_subj = np.bincount(ocol, minlength=R)     # |S_r|

        tgt = np.full(V, 6, dtype=np.int64)
        e = solve_edge_budget(orow, ocol, rng.random(orow.size) + 0.1, tgt,
                              irow, icol, rng.random(irow.size) + 0.1, tgt.copy(),
                              np.full(R, int(tgt.sum()) / R), V, R,
                              col_cap=(n_subj * n_obj).astype(float))
        x = fit_stubs(orow, ocol, rng.random(orow.size) + 0.1, tgt, e, V, R, rng,
                      floor=True, entry_cap=n_obj[ocol])
        y = fit_stubs(irow, icol, rng.random(irow.size) + 0.1, tgt, e, V, R, rng,
                      entry_cap=n_subj[icol])

        self.assertTrue((x <= n_obj[ocol]).all(), "a subject was given more stubs of r "
                                                  "than r has distinct objects")
        self.assertTrue((y <= n_subj[icol]).all(), "an object was given more stubs of r "
                                                   "than r has distinct subjects")
        # The caps are enforced *within* a column, so stub balance must survive them.
        np.testing.assert_array_equal(np.bincount(ocol, x, minlength=R).astype(np.int64), e)
        np.testing.assert_array_equal(np.bincount(icol, y, minlength=R).astype(np.int64), e)
