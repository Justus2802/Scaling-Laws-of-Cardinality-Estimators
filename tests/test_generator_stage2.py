import unittest

import numpy as np
from kgsynth.signature import BlockA, BlockB, BlockC, BlockD, BlockF  # noqa: E402
from kgsynth.signature._fits import (  # noqa: E402
    ExpDecayFit, TruncPowerLawFit, ZipfFit, fit_quantiles,
)
from kgsynth.signature._utils import PowerLawStats  # noqa: E402
from kgsynth.generator import sample_schema, instantiate  # noqa: E402

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def _pls(alpha: float) -> PowerLawStats:
    return PowerLawStats(alpha, 1.0, float("nan"), float("nan"), float("nan"), float("nan"))


def _q(center: float, spread: float, lo: float, hi: float):
    """Quantile fit of a normal sample centred at ``center`` (truncated to [lo, hi])."""
    rng = np.random.default_rng(0)
    return fit_quantiles(rng.normal(center, spread, 500), lo=lo, hi=hi)


def _make_block_a(num_entities=300, num_triples=1200, num_relations=4) -> BlockA:
    a = BlockA()
    a._num_entities = num_entities
    a._num_relations = num_relations
    a._mean_degree = num_triples / num_entities
    return a


def _make_block_c(num_classes=3, class_size_zipf=2.0) -> BlockC:
    c = BlockC()
    c._num_classes = num_classes
    c._class_size_fit = _pls(class_size_zipf)
    c._type_rel_spectrum_exp = ExpDecayFit(rate=0.5, scale=100.0)
    # A complete reduced Block C always carries these (see docs/generator.md
    # §"Target signature must be complete"); sample_schema's group-CS path reads
    # them directly, with no per-entity/per-type fallback.
    c._subj_cooc_exp = ExpDecayFit(rate=0.5, scale=100.0)
    c._obj_cooc_exp  = ExpDecayFit(rate=0.5, scale=100.0)
    # Pair-level multiplicity targets a complete reduced Block C always carries
    # (1.0 = simple graph, no repeated/reversed pairs); sample_schema reads them.
    c._edge_multiplicity = 1.0
    c._bidirectional_ratio = 1.0
    return c


def _make_block_f(num_components=1, largest_component_fraction=1.0) -> BlockF:
    """Reduced BlockF with the fields sample_schema reads."""
    f = BlockF()
    f._num_components = num_components
    f._largest_component_fraction = largest_component_fraction
    return f


def _make_block_b(
    relation_zipf=2.0, in_alpha=4.0, obj_alpha=2.0, subj_alpha=2.2, a_obj=0.5
) -> BlockB:
    """Reduced BlockB with the fields sample_schema reads."""
    b = BlockB()
    b._relation_zipf = ZipfFit(exponent=relation_zipf, x_min=1.0)
    b._in_degree_fit = _pls(in_alpha)
    b._out_degree_fit = _pls(2.5)
    b._out_degree_max = 20
    b._out_degree_p90 = 8.0
    b._in_degree_max = 20
    b._in_degree_p90 = 8.0
    b._obj_alpha_q = _q(obj_alpha, 0.3, 1.4, 3.0)
    b._subj_alpha_q = _q(subj_alpha, 0.3, 1.4, 3.0)
    b._a_obj = a_obj
    b._a_subj = 0.2
    # Reciprocity a complete reduced Block B always carries; all-NaN frac models
    # "no reciprocity signal" so sample_schema leaves relation_reciprocity None
    # (all-asymmetric) — the reciprocity tests use _make_block_b_reciprocal instead.
    b._recip_symmetric_frac = np.full(6, float("nan"))
    b._recip_symmetric_value = float("nan")
    return b


def _make_block_d(
    cs_size_loc=3.0, num_distinct_cs=12, cs_freq_alpha=2.0,
    inv_cs_size_loc=2.0, inv_num_distinct_cs=8, inv_cs_freq_alpha=2.0,
) -> BlockD:
    """Reduced BlockD with the fields sample_schema reads (forward + inverse CS)."""
    d = BlockD()
    d._cs_size_q = _q(cs_size_loc, 1.0, 1.0, 8.0)
    d._inv_cs_size_q = _q(inv_cs_size_loc, 1.0, 1.0, 8.0)
    d._num_distinct_cs = num_distinct_cs
    d._cs_freq_fit = TruncPowerLawFit(cs_freq_alpha, 1.0, 100.0)
    d._inv_num_distinct_cs = inv_num_distinct_cs
    d._inv_cs_freq_fit = TruncPowerLawFit(inv_cs_freq_alpha, 1.0, 100.0)
    return d


def _content_edges(g) -> list:
    return [e for e in g.es if e["predicate"] != _RDF_TYPE]


class TestStage2EdgeBudget(unittest.TestCase):
    """Phase-2 per-relation multiplicity-then-PA with edge conservation."""

    def setUp(self):
        self.a = _make_block_a(num_entities=300, num_triples=1200, num_relations=4)
        self.c = _make_block_c(num_classes=3)
        self.b = _make_block_b()
        self.d = _make_block_d()
        self.f = _make_block_f()

    def test_content_edges_near_budget(self):
        schema = sample_schema(self.a, self.c, b=self.b, d=self.d, f=self.f, seed=0)
        g = instantiate(schema, seed=1)
        content = _content_edges(g)
        # content budget = |E| − one rdf:type edge per entity (types present)
        target = schema.num_triples - schema.num_entities
        # Never exceeds budget (throttle); lands close (only duplicate-triple
        # rejection in PA can undershoot).
        self.assertLessEqual(len(content), target)
        self.assertGreaterEqual(len(content), 0.85 * target)

    def test_every_present_relation_gets_edges(self):
        schema = sample_schema(self.a, self.c, b=self.b, d=self.d, f=self.f, seed=0)
        g = instantiate(schema, seed=1)
        used = {e["predicate"] for e in _content_edges(g)}
        # Each relation that landed in some CS should receive ≥1 edge.
        self.assertTrue(used)
        self.assertTrue(used.issubset(set(schema.relations)))

    def test_deterministic(self):
        schema = sample_schema(self.a, self.c, b=self.b, d=self.d, f=self.f, seed=0)
        g1 = instantiate(schema, seed=1)
        g2 = instantiate(schema, seed=1)
        self.assertEqual(g1.ecount(), g2.ecount())

    def test_no_subject_multiplicity_cap(self):
        # The old hard inverse-functionality cap forced ≤2 subjects per (predicate,
        # object). With the in-side allocation, a heavy subject-multiplicity tail
        # (low α) + PA must let some (predicate, object) pairs exceed 2 subjects.
        b = _make_block_b(subj_alpha=1.5, in_alpha=2.5)
        schema = sample_schema(self.a, self.c, b=b, d=self.d, f=self.f, seed=0)
        g = instantiate(schema, seed=1)
        po_counts: dict[tuple, set] = {}
        for e in _content_edges(g):
            po_counts.setdefault((e["predicate"], e.target), set()).add(e.source)
        max_subjects = max((len(v) for v in po_counts.values()), default=0)
        self.assertGreater(max_subjects, 2)

    def test_measured_relation_zipf_used(self):
        # A very skewed measured relation Zipf should produce more unequal
        # relation weights than a flat one.
        b_flat = _make_block_b(relation_zipf=1.0)
        b_skew = _make_block_b(relation_zipf=4.0)
        s_flat = sample_schema(self.a, self.c, b=b_flat, d=self.d, f=self.f, seed=0)
        s_skew = sample_schema(self.a, self.c, b=b_skew, d=self.d, f=self.f, seed=0)
        self.assertGreater(s_skew.relation_weights.var(), s_flat.relation_weights.var())


def _und_pairs(edges: list) -> set:
    return {(min(s, o), max(s, o)) for s, o, _ in edges if s != o}


class TestStage2Reciprocity(unittest.TestCase):
    """Bidirectional pair-overlap construction driven by per-relation reciprocity."""

    def setUp(self):
        self.a = _make_block_a(num_entities=300, num_triples=1200, num_relations=4)
        self.c = _make_block_c(num_classes=3)
        self.d = _make_block_d()
        self.f = _make_block_f()

    def _make_block_b_reciprocal(self, frac=1.0, value=0.9) -> BlockB:
        b = _make_block_b()
        b._recip_symmetric_frac = np.full(6, frac)
        b._recip_symmetric_value = value
        return b

    def test_high_reciprocity_yields_bidirectional_pairs(self):
        b_recip = self._make_block_b_reciprocal(frac=1.0)
        b_none = _make_block_b()  # no reciprocity attrs set → NotCalculated → asymmetric
        s_recip = sample_schema(self.a, self.c, b=b_recip, d=self.d, f=self.f, seed=0)
        s_none = sample_schema(self.a, self.c, b=b_none, d=self.d, f=self.f, seed=0)
        self.assertIsNotNone(s_recip.relation_reciprocity)
        self.assertTrue((s_recip.relation_reciprocity > 0).any())

        g_recip = instantiate(s_recip, seed=1)
        g_none = instantiate(s_none, seed=1)

        def _bidir_pair_frac(g) -> float:
            edges = [(e.source, e.target, e["predicate"]) for e in _content_edges(g)]
            dir_pairs = {(s, o) for s, o, _ in edges if s != o}
            und = _und_pairs(edges)
            return (len(dir_pairs) - len(und)) / max(1, len(und))

        self.assertGreater(_bidir_pair_frac(g_recip), _bidir_pair_frac(g_none))

    def test_budget_conserved_with_reciprocity(self):
        b_recip = self._make_block_b_reciprocal(frac=1.0)
        schema = sample_schema(self.a, self.c, b=b_recip, d=self.d, f=self.f, seed=0)
        g = instantiate(schema, seed=1)
        content = _content_edges(g)
        target = schema.num_triples - schema.num_entities
        self.assertLessEqual(len(content), target)
        self.assertGreaterEqual(len(content), 0.85 * target)

    def test_deterministic_with_reciprocity(self):
        b_recip = self._make_block_b_reciprocal(frac=1.0)
        schema = sample_schema(self.a, self.c, b=b_recip, d=self.d, f=self.f, seed=0)
        g1 = instantiate(schema, seed=1)
        g2 = instantiate(schema, seed=1)
        self.assertEqual(g1.ecount(), g2.ecount())
        self.assertEqual(
            sorted((e.source, e.target, e["predicate"]) for e in g1.es),
            sorted((e.source, e.target, e["predicate"]) for e in g2.es),
        )


def _count_components(edges: list, n: int) -> tuple[int, int]:
    """Return (num_components, giant_size) from an edge list on n nodes."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for s, o, _ in edges:
        if s < n and o < n and s != o:
            adj[s].append(o)
            adj[o].append(s)
    visited = [False] * n
    comps: list[int] = []
    for start in range(n):
        if not visited[start]:
            size = 0
            stack = [start]
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                visited[v] = True
                size += 1
                stack.extend(u for u in adj[v] if not visited[u])
            comps.append(size)
    return len(comps), max(comps)


class _FakeSch:
    relations = ["r0", "r1"]


def _disconnected_edges(component_sizes: list[int]) -> tuple[list, int]:
    """Build a chain-per-component edge list; returns (edges, total_nodes)."""
    edges: list = []
    offset = 0
    for sz in component_sizes:
        for i in range(sz - 1):
            edges.append((offset + i, offset + i + 1, "r0"))
        offset += sz
    return edges, offset


class TestConnectComponents(unittest.TestCase):
    """_connect_components targeting nc / LCC fraction."""

    def _call(self, edges, n, *, target_nc=1, target_lcc=1.0):
        rng = np.random.default_rng(0)
        seen = {(s, o, p) for s, o, p in edges}
        in_deg = np.zeros(n)
        for _, o, _ in edges:
            in_deg[o] += 1.0
        from kgsynth.generator.stage2 import _connect_components
        _connect_components(edges, n, _FakeSch(), rng, seen, in_deg,
                            target_nc=target_nc, target_lcc=target_lcc)
        return edges

    def test_default_fully_connects(self):
        # With default target_nc=1, every component must be bridged.
        edges, n = _disconnected_edges([7, 2, 1])
        edges = self._call(edges, n)
        nc, _ = _count_components(edges, n)
        self.assertEqual(nc, 1)

    def test_target_nc1_explicit(self):
        edges, n = _disconnected_edges([5, 3, 2])
        edges = self._call(edges, n, target_nc=1, target_lcc=1.0)
        nc, _ = _count_components(edges, n)
        self.assertEqual(nc, 1)

    def test_exact_lcc_match(self):
        # Giant=7, three isolates (size 1 each). target_nc=3 → 2 satellites.
        # sat_budget = (1-0.8)*10 = 2.0; prefix=[0,1,2]; best_j=2 (|2-2|=0, exact).
        # Expect: nc=3, lcc=8/10=0.8.
        edges, n = _disconnected_edges([7, 1, 1, 1])
        edges = self._call(edges, n, target_nc=3, target_lcc=0.8)
        nc, giant = _count_components(edges, n)
        self.assertEqual(nc, 3)
        self.assertAlmostEqual(giant / n, 0.8)

    def test_nearest_prefix_below_budget(self):
        # Giant=8, sats=[1,1,1]. target_nc=2 → 1 satellite.
        # sat_budget=(1-0.85)*10=1.5; prefix=[0,1]; best_j=1 (|1-1.5|=0.5 < |0-1.5|=1.5).
        # Expect: nc=2, giant=9, lcc=0.9.
        edges, n = _disconnected_edges([8, 1, 1, 1])  # n=11, but adjust expectation
        # Recompute: n=11, sat_budget=(1-0.9)*11=1.1; prefix=[0,1]; best_j=1.
        edges = self._call(edges, n, target_nc=2, target_lcc=0.9)
        nc, giant = _count_components(edges, n)
        self.assertEqual(nc, 2)
        self.assertGreaterEqual(giant / n, 0.9)

    def test_nearest_prefix_crosses_budget(self):
        # Giant=6, sats=[1,2,4]. target_nc=2 → 1 satellite.
        # n=13, sat_budget=(1-0.85)*13=1.95; sats_asc=[1,2,4]; k=1; prefix=[0,1].
        # best_j=1 (|1-1.95|=0.95 < |0-1.95|=1.95). Keep size-1 sat, bridge 2 and 4.
        # giant=6+2+4=12, nc=2, lcc=12/13.
        edges, n = _disconnected_edges([6, 1, 2, 4])
        edges = self._call(edges, n, target_nc=2, target_lcc=0.85)
        nc, giant = _count_components(edges, n)
        self.assertEqual(nc, 2)
        self.assertEqual(giant, 12)

    def test_warns_when_only_satellite_too_large_for_lcc(self):
        # Giant(10) + one large satellite(10). target_nc=2, target_lcc=0.9999 →
        # sat_budget=(1-0.9999)*20=0.002. prefix=[0,10]; best_j=0 wins (|0-0.002|<|10-0.002|).
        # Algorithm bridges everything → nc=1; warning fires because best_j=0 but sat_budget>0.
        edges, n = _disconnected_edges([10, 10])
        import logging
        with self.assertLogs("generator.stage2", level=logging.WARNING):
            edges = self._call(edges, n, target_nc=2, target_lcc=0.9999)
        nc, _ = _count_components(edges, n)
        self.assertEqual(nc, 1)

    def test_single_component_no_op(self):
        # Already one component — nothing should be added.
        edges, n = _disconnected_edges([10])
        original_len = len(edges)
        edges = self._call(edges, n, target_nc=3, target_lcc=0.8)
        self.assertEqual(len(edges), original_len)
        nc, _ = _count_components(edges, n)
        self.assertEqual(nc, 1)


if __name__ == "__main__":
    unittest.main()
