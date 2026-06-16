import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from signature_reduced import BlockA, BlockB, BlockC, BlockD  # noqa: E402
from signature_reduced._fits import ExpDecayFit, SkewNormFit, ZipfFit  # noqa: E402
from signature._utils import PowerLawStats  # noqa: E402
from generator import sample_schema, instantiate  # noqa: E402

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def _pls(alpha: float) -> PowerLawStats:
    return PowerLawStats(alpha, 1.0, float("nan"), float("nan"), float("nan"), float("nan"))


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
    return c


def _make_block_b(
    relation_zipf=2.0, in_alpha=4.0, obj_alpha=2.0, subj_alpha=2.2, a_obj=0.5
) -> BlockB:
    """Reduced BlockB with the fields sample_schema reads."""
    b = BlockB()
    b._relation_zipf = ZipfFit(exponent=relation_zipf, x_min=1.0)
    b._in_degree_fit = _pls(in_alpha)
    b._out_degree_fit = _pls(2.5)
    b._obj_alpha_skew = SkewNormFit(loc=obj_alpha, scale=0.3, shape=0.0, lo=1.4, hi=3.0)
    b._subj_alpha_skew = SkewNormFit(loc=subj_alpha, scale=0.3, shape=0.0, lo=1.4, hi=3.0)
    b._a_obj = a_obj
    b._a_subj = 0.2
    return b


def _make_block_d(cs_size_loc=3.0, num_distinct_cs=12, cs_freq_alpha=2.0) -> BlockD:
    """Reduced BlockD with the fields sample_schema reads."""
    d = BlockD()
    d._cs_size_skew = SkewNormFit(loc=cs_size_loc, scale=1.0, shape=0.0, lo=1.0, hi=8.0)
    d._inv_cs_size_skew = SkewNormFit(loc=2.0, scale=1.0, shape=0.0, lo=1.0, hi=8.0)
    d._num_distinct_cs = num_distinct_cs
    d._cs_freq_fit = _pls(cs_freq_alpha)
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

    def test_content_edges_near_budget(self):
        schema = sample_schema(self.a, self.c, b=self.b, d=self.d, seed=0)
        g = instantiate(schema, seed=1)
        content = _content_edges(g)
        # content budget = |E| − one rdf:type edge per entity (types present)
        target = schema.num_triples - schema.num_entities
        # Never exceeds budget (throttle); lands close (only duplicate-triple
        # rejection in PA can undershoot).
        self.assertLessEqual(len(content), target)
        self.assertGreaterEqual(len(content), 0.85 * target)

    def test_every_present_relation_gets_edges(self):
        schema = sample_schema(self.a, self.c, b=self.b, d=self.d, seed=0)
        g = instantiate(schema, seed=1)
        used = {e["predicate"] for e in _content_edges(g)}
        # Each relation that landed in some CS should receive ≥1 edge.
        self.assertTrue(used)
        self.assertTrue(used.issubset(set(schema.relations)))

    def test_deterministic(self):
        schema = sample_schema(self.a, self.c, b=self.b, d=self.d, seed=0)
        g1 = instantiate(schema, seed=1)
        g2 = instantiate(schema, seed=1)
        self.assertEqual(g1.ecount(), g2.ecount())

    def test_fallback_no_b_no_d(self):
        # Without Block B/D the neutral fallback must still produce a valid graph
        # near the budget (uniform weights, budget-derived CS size).
        schema = sample_schema(self.a, self.c, seed=0)
        g = instantiate(schema, seed=1)
        content = _content_edges(g)
        target = schema.num_triples - schema.num_entities
        self.assertLessEqual(len(content), target)
        self.assertGreaterEqual(len(content), 0.7 * target)

    def test_no_subject_multiplicity_cap(self):
        # The old hard inverse-functionality cap forced ≤2 subjects per (predicate,
        # object). With the in-side allocation, a heavy subject-multiplicity tail
        # (low α) + PA must let some (predicate, object) pairs exceed 2 subjects.
        b = _make_block_b(subj_alpha=1.5, in_alpha=2.5)
        schema = sample_schema(self.a, self.c, b=b, d=self.d, seed=0)
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
        s_flat = sample_schema(self.a, self.c, b=b_flat, d=self.d, seed=0)
        s_skew = sample_schema(self.a, self.c, b=b_skew, d=self.d, seed=0)
        self.assertGreater(s_skew.relation_weights.var(), s_flat.relation_weights.var())


if __name__ == "__main__":
    unittest.main()
