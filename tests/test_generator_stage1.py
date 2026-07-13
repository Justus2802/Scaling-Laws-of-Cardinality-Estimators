import unittest

import numpy as np
from kgsynth.signature import BlockA, BlockB, BlockC, BlockD, BlockF
from kgsynth.signature._fits import (
    ExpDecayFit, TruncPowerLawFit, ZipfFit, fit_quantiles, nan_exp_decay,
)
from kgsynth.signature._utils import PowerLawStats
from kgsynth.generator import Schema, sample_schema
from kgsynth.generator.stage1 import COOC_NUM_GROUPS, _validate_target

# Default P(r|t) type-relation spectrum and co-occurrence spectra: positive,
# decaying exp curves that reconstruct to a usable low-rank signal.
_DEFAULT_SPECTRUM = ExpDecayFit(rate=0.5, scale=100.0)


def _q(center: float, spread: float, lo: float, hi: float):
    """Quantile fit of a normal sample centred at ``center`` (truncated to [lo, hi])."""
    rng = np.random.default_rng(0)
    return fit_quantiles(rng.normal(center, spread, 500), lo=lo, hi=hi)


def _make_block_a(
    num_entities: int = 100,
    num_triples: int = 500,
    num_relations: int = 10,
) -> BlockA:
    """Build a reduced BlockA by setting its measured fields directly.

    Reduced BlockA stores mean degree (E/V); the generator recovers the edge
    budget as round(V × mean_degree).
    """
    a = BlockA()
    a._num_entities = num_entities
    a._num_relations = num_relations
    a._mean_degree = (num_triples / num_entities) if num_entities else 0.0
    return a


def _make_block_c(
    num_classes: int = 5,
    class_size_zipf: float = 2.0,
    spectrum: ExpDecayFit = _DEFAULT_SPECTRUM,
    subj_cooc: ExpDecayFit = _DEFAULT_SPECTRUM,
    obj_cooc: ExpDecayFit = _DEFAULT_SPECTRUM,
) -> BlockC:
    """Build a reduced BlockC by setting the fields sample_schema reads.

    ``class_size_zipf`` becomes the class-size power-law α; ``spectrum`` is the
    P(r|t) type-relation exp-decay spectrum the generator reconstructs singular
    values from. Pass ``nan_exp_decay()`` to model "too few classes to fit" —
    the legitimate small-R/untyped-KG fallback (see ``docs/generator.md``).
    ``subj_cooc`` / ``obj_cooc`` set the co-occurrence spectra used for the
    forward/inverse CS group prototypes; a real graph always measures these
    (``_validate_target`` rejects NaN here — see ``TestValidateTarget``).
    """
    c = BlockC()
    c._num_classes = num_classes
    c._class_size_fit = PowerLawStats(
        class_size_zipf, 1.0, float("nan"), float("nan"), float("nan"), float("nan")
    )
    c._type_rel_spectrum_exp = spectrum
    c._subj_cooc_exp = subj_cooc
    c._obj_cooc_exp  = obj_cooc
    # Pair-level multiplicity targets a complete reduced Block C always carries
    # (1.0 = simple graph, no repeated/reversed pairs); sample_schema reads them.
    c._edge_multiplicity = 1.0
    c._bidirectional_ratio = 1.0
    return c


def _make_block_b(
    relation_zipf=2.0, in_alpha=4.0, out_alpha=2.5, obj_alpha=2.0, subj_alpha=2.2, a_obj=0.5,
) -> BlockB:
    """Reduced BlockB with the fields sample_schema/_validate_target read."""
    b = BlockB()
    b._relation_zipf = ZipfFit(exponent=relation_zipf, x_min=1.0)
    b._in_degree_fit = PowerLawStats(in_alpha, 1.0, *([float("nan")] * 4))
    b._out_degree_fit = PowerLawStats(out_alpha, 1.0, *([float("nan")] * 4))
    b._out_degree_max = 20
    b._out_degree_p90 = 8.0
    b._in_degree_max = 20
    b._in_degree_p90 = 8.0
    b._obj_alpha_q = _q(obj_alpha, 0.3, 1.4, 3.0)
    b._subj_alpha_q = _q(subj_alpha, 0.3, 1.4, 3.0)
    b._obj_mult_max = 12      # upper bound of the per-relation multiplicity draws
    b._subj_mult_max = 12
    b._a_obj = a_obj
    b._a_subj = 0.2
    # Reciprocity a complete reduced Block B always carries; all-NaN frac models
    # "no reciprocity signal" (small-R fallback) so sample_schema leaves
    # relation_reciprocity None (all-asymmetric).
    b._recip_symmetric_frac = np.full(6, float("nan"))
    b._recip_symmetric_value = float("nan")
    return b


def _make_block_d(
    cs_size_loc=3.0, num_distinct_cs=12, cs_freq_alpha=2.0,
    inv_cs_size_loc=2.0, inv_num_distinct_cs=8, inv_cs_freq_alpha=2.0,
) -> BlockD:
    """Reduced BlockD with the fields sample_schema/_validate_target read."""
    d = BlockD()
    d._cs_size_q = _q(cs_size_loc, 1.0, 1.0, 8.0)
    d._inv_cs_size_q = _q(inv_cs_size_loc, 1.0, 1.0, 8.0)
    d._num_distinct_cs = num_distinct_cs
    d._cs_freq_fit = TruncPowerLawFit(cs_freq_alpha, 1.0, 100.0)
    d._inv_num_distinct_cs = inv_num_distinct_cs
    d._inv_cs_freq_fit = TruncPowerLawFit(inv_cs_freq_alpha, 1.0, 100.0)
    return d


def _make_block_f(num_components=1, largest_component_fraction=1.0) -> BlockF:
    """Reduced BlockF with the fields sample_schema/_validate_target read."""
    f = BlockF()
    f._num_components = num_components
    f._largest_component_fraction = largest_component_fraction
    return f


class TestSampleSchemaStructure(unittest.TestCase):
    """Structural invariants: correct shapes, sums, and field wiring."""

    def setUp(self):
        self.a = _make_block_a(num_entities=100, num_triples=500, num_relations=10)
        self.c = _make_block_c(num_classes=5, class_size_zipf=2.0)
        self.b = _make_block_b()
        self.d = _make_block_d()
        self.f = _make_block_f()

    def _schema(self, **kw):
        return sample_schema(self.a, self.c, d=self.d, b=self.b, f=self.f, seed=0, **kw)

    def test_returns_schema_instance(self):
        schema = self._schema()
        self.assertIsInstance(schema, Schema)

    def test_relations_count_matches_block_a(self):
        schema = self._schema()
        self.assertEqual(len(schema.relations), self.a.num_relations)

    def test_types_count_matches_block_c(self):
        schema = self._schema()
        self.assertEqual(len(schema.types), self.c.num_classes)

    def test_relation_weights_sum_to_one(self):
        schema = self._schema()
        self.assertAlmostEqual(schema.relation_weights.sum(), 1.0, places=10)

    def test_relation_weights_all_positive(self):
        schema = self._schema()
        self.assertTrue(np.all(schema.relation_weights > 0))

    def test_type_weights_sum_to_one(self):
        schema = self._schema()
        self.assertAlmostEqual(schema.type_weights.sum(), 1.0, places=10)

    def test_type_weights_all_positive(self):
        schema = self._schema()
        self.assertTrue(np.all(schema.type_weights > 0))

    def test_type_relation_probs_shape(self):
        schema = self._schema()
        self.assertEqual(
            schema.type_relation_probs.shape,
            (self.c.num_classes, self.a.num_relations),
        )

    def test_type_relation_probs_rows_sum_to_one(self):
        schema = self._schema()
        row_sums = schema.type_relation_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_type_relation_probs_nonnegative(self):
        schema = self._schema()
        self.assertTrue(np.all(schema.type_relation_probs >= 0.0))

    def test_num_entities_passed_through(self):
        schema = self._schema()
        self.assertEqual(schema.num_entities, self.a.num_entities)

    def test_num_triples_derived_from_mean_degree(self):
        schema = self._schema()
        expected = round(self.a.num_entities * self.a.mean_degree)
        self.assertEqual(schema.num_triples, expected)

    def test_relation_uris_are_strings(self):
        schema = self._schema()
        self.assertTrue(all(isinstance(r, str) for r in schema.relations))
        self.assertTrue(all(r.startswith("http://") for r in schema.relations))

    def test_type_uris_are_strings(self):
        schema = self._schema()
        self.assertTrue(all(isinstance(t, str) for t in schema.types))


class TestSampleSchemaEdgeCases(unittest.TestCase):
    """Degenerate inputs: zero types, NaN class-size Zipf, single relation."""

    def _bdf(self):
        return dict(b=_make_block_b(), d=_make_block_d(), f=_make_block_f())

    def test_zero_types_gives_empty_type_fields(self):
        a = _make_block_a(num_relations=5)
        c = _make_block_c(num_classes=0, class_size_zipf=float("nan"))
        schema = sample_schema(a, c, seed=0, **self._bdf())
        self.assertEqual(len(schema.types), 0)
        self.assertEqual(schema.type_weights.shape, (0,))
        self.assertEqual(schema.type_relation_probs.shape, (0, 5))

    def test_nan_class_zipf_falls_back_to_uniform_type_weights(self):
        # Block C could not fit a Zipf (too few classes) → uniform weights
        a = _make_block_a(num_relations=4)
        c = _make_block_c(num_classes=3, class_size_zipf=float("nan"))
        schema = sample_schema(a, c, seed=0, **self._bdf())
        self.assertEqual(len(schema.type_weights), 3)
        np.testing.assert_allclose(schema.type_weights, 1 / 3, atol=1e-10)

    def test_single_relation(self):
        a = _make_block_a(num_relations=1)
        c = _make_block_c(num_classes=2)
        schema = sample_schema(a, c, seed=0, **self._bdf())
        self.assertEqual(len(schema.relations), 1)
        self.assertAlmostEqual(schema.relation_weights[0], 1.0)
        # P(r|t): 2 types × 1 relation, each row must be [1.0]
        np.testing.assert_allclose(schema.type_relation_probs, 1.0, atol=1e-10)

    def test_single_type(self):
        a = _make_block_a(num_relations=5)
        c = _make_block_c(num_classes=1, class_size_zipf=2.0)
        schema = sample_schema(a, c, seed=0, **self._bdf())
        self.assertEqual(len(schema.types), 1)
        self.assertAlmostEqual(schema.type_weights[0], 1.0)
        self.assertAlmostEqual(schema.type_relation_probs[0].sum(), 1.0)

    def test_no_spectrum_signal_falls_back_to_relation_weights(self):
        # With no P(r|t) spectrum (untyped-KG fallback) the low-rank path
        # degenerates; each type row should equal the global relation_weights.
        a = _make_block_a(num_relations=4)
        c = _make_block_c(num_classes=3, spectrum=nan_exp_decay())
        schema = sample_schema(a, c, seed=0, **self._bdf())
        for row in schema.type_relation_probs:
            np.testing.assert_allclose(row, schema.relation_weights, atol=1e-10)


class TestSampleSchemaReproducibility(unittest.TestCase):
    """Seed and determinism guarantees."""

    def setUp(self):
        self.a = _make_block_a()
        self.c = _make_block_c()
        self.b = _make_block_b()
        self.d = _make_block_d()
        self.f = _make_block_f()

    def test_same_seed_same_output(self):
        s1 = sample_schema(self.a, self.c, d=self.d, b=self.b, f=self.f, seed=42)
        s2 = sample_schema(self.a, self.c, d=self.d, b=self.b, f=self.f, seed=42)
        np.testing.assert_array_equal(s1.relation_weights, s2.relation_weights)
        np.testing.assert_array_equal(s1.type_weights, s2.type_weights)
        np.testing.assert_array_equal(s1.type_relation_probs, s2.type_relation_probs)

    def test_different_seeds_give_different_weights(self):
        s1 = sample_schema(self.a, self.c, d=self.d, b=self.b, f=self.f, seed=0)
        s2 = sample_schema(self.a, self.c, d=self.d, b=self.b, f=self.f, seed=1)
        # With 10 relations the chance of an exact match by coincidence is negligible
        self.assertFalse(np.allclose(s1.relation_weights, s2.relation_weights))


class TestSampleSchemaZipfEffect(unittest.TestCase):
    """Higher Zipf exponent → more skewed relation weights.

    ``relation_zipf_exponent`` is only used as a fallback when Block B's own
    measured exponent is unavailable (small R) — see sample_schema's
    docstring — so these fixtures give Block B a NaN relation_zipf to exercise
    the parameter path directly, rather than the (higher-precedence) measured one.
    """

    def _nan_zipf_block_b(self):
        b = _make_block_b()
        b._relation_zipf = ZipfFit(exponent=float("nan"), x_min=1.0)
        return b

    def test_higher_exponent_more_skewed(self):
        a = _make_block_a(num_relations=20)
        c = _make_block_c()
        bdf = dict(b=self._nan_zipf_block_b(), d=_make_block_d(), f=_make_block_f())
        low = sample_schema(a, c, relation_zipf_exponent=1.0, seed=0, **bdf)
        high = sample_schema(a, c, relation_zipf_exponent=3.0, seed=0, **bdf)
        # Gini coefficient: higher Zipf → higher Gini (more skewed)
        def gini(w):
            w = np.sort(w)
            n = len(w)
            return (2 * np.sum(np.arange(1, n + 1) * w) / (n * w.sum())) - (n + 1) / n
        self.assertGreater(gini(high.relation_weights), gini(low.relation_weights))

    def test_relation_weights_variance_increases_with_exponent(self):
        a = _make_block_a(num_relations=15)
        c = _make_block_c()
        bdf = dict(b=self._nan_zipf_block_b(), d=_make_block_d(), f=_make_block_f())
        low = sample_schema(a, c, relation_zipf_exponent=1.0, seed=7, **bdf)
        high = sample_schema(a, c, relation_zipf_exponent=4.0, seed=7, **bdf)
        self.assertGreater(high.relation_weights.var(), low.relation_weights.var())


class TestSampleSchemaCoocGroups(unittest.TestCase):
    """Co-occurrence group prototypes built from subj_cooc_exp / obj_cooc_exp.

    A real graph always measures these (never NaN — see docs/generator.md), so
    group prototypes are always built; there is no "no groups" fallback path
    left to test here (see TestValidateTarget for the NaN-rejection behaviour).
    """

    _COOC = ExpDecayFit(rate=0.5, scale=100.0)

    def _schema_with_groups(self, num_relations=10, num_classes=5):
        a = _make_block_a(num_relations=num_relations)
        c = _make_block_c(num_classes=num_classes, subj_cooc=self._COOC, obj_cooc=self._COOC)
        bdf = dict(b=_make_block_b(), d=_make_block_d(), f=_make_block_f())
        return sample_schema(a, c, seed=0, **bdf)

    def test_group_probs_built_when_cooc_available(self):
        schema = self._schema_with_groups()
        self.assertIsNotNone(schema.subj_group_probs)
        self.assertIsNotNone(schema.obj_group_probs)

    def test_group_probs_shape(self):
        num_relations = 8
        schema = self._schema_with_groups(num_relations=num_relations)
        self.assertEqual(schema.subj_group_probs.shape, (COOC_NUM_GROUPS, num_relations))
        self.assertEqual(schema.obj_group_probs.shape,  (COOC_NUM_GROUPS, num_relations))

    def test_group_probs_rows_sum_to_one(self):
        schema = self._schema_with_groups()
        np.testing.assert_allclose(schema.subj_group_probs.sum(axis=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(schema.obj_group_probs.sum(axis=1),  1.0, atol=1e-10)

    def test_group_weights_sum_to_one(self):
        schema = self._schema_with_groups()
        self.assertAlmostEqual(schema.subj_group_weights.sum(), 1.0, places=10)
        self.assertAlmostEqual(schema.obj_group_weights.sum(),  1.0, places=10)

    def test_group_weights_all_positive(self):
        schema = self._schema_with_groups()
        self.assertTrue(np.all(schema.subj_group_weights > 0))
        self.assertTrue(np.all(schema.obj_group_weights  > 0))


class TestValidateTarget(unittest.TestCase):
    """_validate_target rejects a target signature missing a required feature."""

    def setUp(self):
        self.a = _make_block_a()
        self.c = _make_block_c()
        self.b = _make_block_b()
        self.d = _make_block_d()
        self.f = _make_block_f()

    def test_complete_target_passes(self):
        _validate_target(self.a, self.b, self.c, self.d, self.f)  # no raise

    def test_nan_num_entities_raises(self):
        self.a._num_entities = float("nan")
        with self.assertRaisesRegex(ValueError, "num_entities"):
            _validate_target(self.a, self.b, self.c, self.d, self.f)

    def test_nan_cs_size_q_raises(self):
        self.d._cs_size_q = (float("nan"),) * len(self.d._cs_size_q)
        with self.assertRaisesRegex(ValueError, "cs_size_q"):
            _validate_target(self.a, self.b, self.c, self.d, self.f)

    def test_nan_out_degree_p90_raises(self):
        self.b._out_degree_p90 = float("nan")
        with self.assertRaisesRegex(ValueError, "out_degree_p90"):
            _validate_target(self.a, self.b, self.c, self.d, self.f)

    def test_nan_subj_cooc_exp_raises(self):
        self.c._subj_cooc_exp = nan_exp_decay()
        with self.assertRaisesRegex(ValueError, "subj_cooc_exp"):
            _validate_target(self.a, self.b, self.c, self.d, self.f)

    def test_nan_num_components_raises(self):
        self.f._num_components = float("nan")
        with self.assertRaisesRegex(ValueError, "num_components"):
            _validate_target(self.a, self.b, self.c, self.d, self.f)

    def test_sample_schema_propagates_validation_error(self):
        self.d._num_distinct_cs = float("nan")
        with self.assertRaises(ValueError):
            sample_schema(self.a, self.c, d=self.d, b=self.b, f=self.f, seed=0)


if __name__ == "__main__":
    unittest.main()
