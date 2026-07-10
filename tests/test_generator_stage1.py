import unittest

import numpy as np
from kgsynth.signature import BlockA, BlockC
from kgsynth.signature._fits import ExpDecayFit, nan_exp_decay
from kgsynth.signature._utils import PowerLawStats
from kgsynth.generator import Schema, sample_schema
from kgsynth.generator.stage1 import COOC_NUM_GROUPS

# Default P(r|t) type-relation spectrum: a positive, decaying exp curve that
# reconstructs to a usable low-rank signal for P(r|t) synthesis.
_DEFAULT_SPECTRUM = ExpDecayFit(rate=0.5, scale=100.0)


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
    subj_cooc: ExpDecayFit | None = None,
    obj_cooc: ExpDecayFit | None = None,
) -> BlockC:
    """Build a reduced BlockC by setting the fields sample_schema reads.

    ``class_size_zipf`` becomes the class-size power-law α; ``spectrum`` is the
    P(r|t) type-relation exp-decay spectrum the generator reconstructs singular
    values from. Pass ``nan_exp_decay()`` to model "no P(r|t) signal".
    ``subj_cooc`` / ``obj_cooc`` set the co-occurrence spectra; default to NaN
    (no group prototypes built).
    """
    c = BlockC()
    c._num_classes = num_classes
    c._class_size_fit = PowerLawStats(
        class_size_zipf, 1.0, float("nan"), float("nan"), float("nan"), float("nan")
    )
    c._type_rel_spectrum_exp = spectrum
    c._subj_cooc_exp = subj_cooc if subj_cooc is not None else nan_exp_decay()
    c._obj_cooc_exp  = obj_cooc  if obj_cooc  is not None else nan_exp_decay()
    return c


class TestSampleSchemaStructure(unittest.TestCase):
    """Structural invariants: correct shapes, sums, and field wiring."""

    def setUp(self):
        self.a = _make_block_a(num_entities=100, num_triples=500, num_relations=10)
        self.c = _make_block_c(num_classes=5, class_size_zipf=2.0)

    def test_returns_schema_instance(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertIsInstance(schema, Schema)

    def test_relations_count_matches_block_a(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertEqual(len(schema.relations), self.a.num_relations)

    def test_types_count_matches_block_c(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertEqual(len(schema.types), self.c.num_classes)

    def test_relation_weights_sum_to_one(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertAlmostEqual(schema.relation_weights.sum(), 1.0, places=10)

    def test_relation_weights_all_positive(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertTrue(np.all(schema.relation_weights > 0))

    def test_type_weights_sum_to_one(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertAlmostEqual(schema.type_weights.sum(), 1.0, places=10)

    def test_type_weights_all_positive(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertTrue(np.all(schema.type_weights > 0))

    def test_type_relation_probs_shape(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertEqual(
            schema.type_relation_probs.shape,
            (self.c.num_classes, self.a.num_relations),
        )

    def test_type_relation_probs_rows_sum_to_one(self):
        schema = sample_schema(self.a, self.c, seed=0)
        row_sums = schema.type_relation_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_type_relation_probs_nonnegative(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertTrue(np.all(schema.type_relation_probs >= 0.0))

    def test_num_entities_passed_through(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertEqual(schema.num_entities, self.a.num_entities)

    def test_num_triples_derived_from_mean_degree(self):
        schema = sample_schema(self.a, self.c, seed=0)
        expected = round(self.a.num_entities * self.a.mean_degree)
        self.assertEqual(schema.num_triples, expected)

    def test_relation_uris_are_strings(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertTrue(all(isinstance(r, str) for r in schema.relations))
        self.assertTrue(all(r.startswith("http://") for r in schema.relations))

    def test_type_uris_are_strings(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertTrue(all(isinstance(t, str) for t in schema.types))


class TestSampleSchemaEdgeCases(unittest.TestCase):
    """Degenerate inputs: zero types, NaN Zipf, single relation."""

    def test_zero_types_gives_empty_type_fields(self):
        a = _make_block_a(num_relations=5)
        c = _make_block_c(num_classes=0, class_size_zipf=float("nan"))
        schema = sample_schema(a, c, seed=0)
        self.assertEqual(len(schema.types), 0)
        self.assertEqual(schema.type_weights.shape, (0,))
        self.assertEqual(schema.type_relation_probs.shape, (0, 5))

    def test_nan_class_zipf_falls_back_to_uniform_type_weights(self):
        # Block C could not fit a Zipf (too few classes) → uniform weights
        a = _make_block_a(num_relations=4)
        c = _make_block_c(num_classes=3, class_size_zipf=float("nan"))
        schema = sample_schema(a, c, seed=0)
        self.assertEqual(len(schema.type_weights), 3)
        np.testing.assert_allclose(schema.type_weights, 1 / 3, atol=1e-10)

    def test_single_relation(self):
        a = _make_block_a(num_relations=1)
        c = _make_block_c(num_classes=2)
        schema = sample_schema(a, c, seed=0)
        self.assertEqual(len(schema.relations), 1)
        self.assertAlmostEqual(schema.relation_weights[0], 1.0)
        # P(r|t): 2 types × 1 relation, each row must be [1.0]
        np.testing.assert_allclose(schema.type_relation_probs, 1.0, atol=1e-10)

    def test_single_type(self):
        a = _make_block_a(num_relations=5)
        c = _make_block_c(num_classes=1, class_size_zipf=2.0)
        schema = sample_schema(a, c, seed=0)
        self.assertEqual(len(schema.types), 1)
        self.assertAlmostEqual(schema.type_weights[0], 1.0)
        self.assertAlmostEqual(schema.type_relation_probs[0].sum(), 1.0)

    def test_no_spectrum_signal_falls_back_to_relation_weights(self):
        # With no P(r|t) spectrum the low-rank path degenerates;
        # each type row should equal the global relation_weights.
        a = _make_block_a(num_relations=4)
        c = _make_block_c(num_classes=3, spectrum=nan_exp_decay())
        schema = sample_schema(a, c, seed=0)
        for row in schema.type_relation_probs:
            np.testing.assert_allclose(row, schema.relation_weights, atol=1e-10)


class TestSampleSchemaReproducibility(unittest.TestCase):
    """Seed and determinism guarantees."""

    def setUp(self):
        self.a = _make_block_a()
        self.c = _make_block_c()

    def test_same_seed_same_output(self):
        s1 = sample_schema(self.a, self.c, seed=42)
        s2 = sample_schema(self.a, self.c, seed=42)
        np.testing.assert_array_equal(s1.relation_weights, s2.relation_weights)
        np.testing.assert_array_equal(s1.type_weights, s2.type_weights)
        np.testing.assert_array_equal(s1.type_relation_probs, s2.type_relation_probs)

    def test_different_seeds_give_different_weights(self):
        s1 = sample_schema(self.a, self.c, seed=0)
        s2 = sample_schema(self.a, self.c, seed=1)
        # With 10 relations the chance of an exact match by coincidence is negligible
        self.assertFalse(np.allclose(s1.relation_weights, s2.relation_weights))


class TestSampleSchemaZipfEffect(unittest.TestCase):
    """Higher Zipf exponent → more skewed relation weights."""

    def test_higher_exponent_more_skewed(self):
        a = _make_block_a(num_relations=20)
        c = _make_block_c()
        low = sample_schema(a, c, relation_zipf_exponent=1.0, seed=0)
        high = sample_schema(a, c, relation_zipf_exponent=3.0, seed=0)
        # Gini coefficient: higher Zipf → higher Gini (more skewed)
        def gini(w):
            w = np.sort(w)
            n = len(w)
            return (2 * np.sum(np.arange(1, n + 1) * w) / (n * w.sum())) - (n + 1) / n
        self.assertGreater(gini(high.relation_weights), gini(low.relation_weights))

    def test_relation_weights_variance_increases_with_exponent(self):
        a = _make_block_a(num_relations=15)
        c = _make_block_c()
        low = sample_schema(a, c, relation_zipf_exponent=1.0, seed=7)
        high = sample_schema(a, c, relation_zipf_exponent=4.0, seed=7)
        self.assertGreater(high.relation_weights.var(), low.relation_weights.var())


class TestSampleSchemaCoocGroups(unittest.TestCase):
    """Co-occurrence group prototypes built from subj_cooc_exp / obj_cooc_exp."""

    _COOC = ExpDecayFit(rate=0.5, scale=100.0)

    def _schema_with_groups(self, num_relations=10, num_classes=5):
        a = _make_block_a(num_relations=num_relations)
        c = _make_block_c(num_classes=num_classes, subj_cooc=self._COOC, obj_cooc=self._COOC)
        return sample_schema(a, c, seed=0)

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

    def test_nan_cooc_exp_gives_none_groups(self):
        # Default _make_block_c has NaN cooc fits → no groups built.
        a = _make_block_a()
        c = _make_block_c()
        schema = sample_schema(a, c, seed=0)
        self.assertIsNone(schema.subj_group_probs)
        self.assertIsNone(schema.obj_group_probs)


if __name__ == "__main__":
    unittest.main()
