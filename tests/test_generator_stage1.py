import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from signature import BlockA, BlockC
from generator import Schema, sample_schema

_TOP_K_SV = 10


def _make_block_a(
    num_entities: int = 100,
    num_triples: int = 500,
    num_relations: int = 10,
) -> BlockA:
    return BlockA(
        num_entities=num_entities,
        num_triples=num_triples,
        num_relations=num_relations,
        density=num_triples / (num_entities ** 2) if num_entities else 0.0,
        triples_per_entity=num_triples / num_entities if num_entities else 0.0,
        relation_reuse=num_triples / num_relations if num_relations else 0.0,
    )


def _make_block_c(
    num_classes: int = 5,
    class_size_zipf: float = 2.0,
    singular_values: np.ndarray | None = None,
) -> BlockC:
    if singular_values is None:
        # Plausible decaying singular values
        singular_values = np.array(
            [100.0, 60.0, 30.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
        )
    return BlockC(
        subj_singular_values=singular_values,
        subj_cooc_density=0.4,
        subj_row_entropies=np.ones(5),
        obj_singular_values=singular_values.copy(),
        obj_cooc_density=0.3,
        obj_row_entropies=np.ones(5),
        num_classes=num_classes,
        class_size_zipf_exponent=class_size_zipf,
        class_sizes={f"T{i}": max(1, 10 - i) for i in range(num_classes)},
        type_relation_conditional={},
    )


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

    def test_num_triples_passed_through(self):
        schema = sample_schema(self.a, self.c, seed=0)
        self.assertEqual(schema.num_triples, self.a.num_triples)

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

    def test_zero_singular_values_falls_back_to_relation_weights(self):
        # When all singular values are zero the low-rank path degenerates;
        # each type row should equal the global relation_weights.
        a = _make_block_a(num_relations=4)
        c = _make_block_c(num_classes=3, singular_values=np.zeros(_TOP_K_SV))
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


if __name__ == "__main__":
    unittest.main()
