import math
import os
import sys
import tempfile
import unittest

import igraph
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_io import load_kg
from signature import BlockC

_VECTOR_LEN = 29   # 10 subj_SVs + 3 subj_stats + 10 obj_SVs + 3 obj_stats + 3 type_stats
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_EX = "http://example.org/"


class TestBlockCSmallFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(content)
        return load_kg(path)

    def test_empty_graph(self):
        g = igraph.Graph(directed=True)
        g.vs["is_literal"] = []
        c = BlockC().calculate(g)
        self.assertEqual(c.num_classes, 0)
        self.assertTrue(math.isnan(c.class_size_zipf_exponent))
        self.assertEqual(c.class_sizes, {})
        self.assertEqual(c.type_relation_conditional, {})
        self.assertTrue(np.all(c.subj_singular_values == 0.0))
        self.assertTrue(np.all(c.obj_singular_values == 0.0))
        self.assertEqual(len(c.subj_singular_values), 10)
        self.assertEqual(len(c.obj_singular_values), 10)
        self.assertEqual(len(c.as_vector()), _VECTOR_LEN)

    def test_single_triple_no_type(self):
        # One predicate → 1×1 co-occurrence matrix, no type info
        c = BlockC().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:s ex:p ex:o .\n"
        ))
        self.assertEqual(c.num_classes, 0)
        self.assertEqual(len(c.subj_singular_values), 10)
        self.assertEqual(len(c.obj_singular_values), 10)
        self.assertEqual(len(c.as_vector()), _VECTOR_LEN)

    def test_cooc_subject_side_full_density_when_subject_uses_both(self):
        # ex:a uses both ex:p and ex:q → all 4 entries of 2×2 M_subj are nonzero
        c = BlockC().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:x .\n"
            "ex:a ex:q ex:y .\n"
        ))
        self.assertAlmostEqual(c.subj_cooc_density, 1.0)
        self.assertEqual(len(c.as_vector()), _VECTOR_LEN)

    def test_cooc_subject_side_half_density_when_subjects_disjoint(self):
        # ex:a uses ex:p only; ex:b uses ex:q only → off-diagonal of M_subj is 0
        # nnz = 2 (diagonal), total_cells = 4 → density = 0.5
        c = BlockC().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:x .\n"
            "ex:b ex:q ex:y .\n"
        ))
        self.assertAlmostEqual(c.subj_cooc_density, 0.5)
        self.assertEqual(len(c.as_vector()), _VECTOR_LEN)

    def test_rdf_type_class_detection(self):
        # Two entities with distinct types → two classes
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
            "ex:a rdf:type ex:Person .\n"
            "ex:b rdf:type ex:Animal .\n"
        )
        c = BlockC().calculate(self._load_ttl(ttl))
        self.assertEqual(c.num_classes, 2)
        self.assertIn(_EX + "Person", c.class_sizes)
        self.assertIn(_EX + "Animal", c.class_sizes)
        self.assertEqual(c.class_sizes[_EX + "Person"], 1)
        self.assertEqual(c.class_sizes[_EX + "Animal"], 1)

    def test_class_size_counts_distinct_subjects(self):
        # Two entities both typed as ex:Person → class size = 2, not 1
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
            "ex:a rdf:type ex:Person .\n"
            "ex:b rdf:type ex:Person .\n"
        )
        c = BlockC().calculate(self._load_ttl(ttl))
        self.assertEqual(c.num_classes, 1)
        self.assertEqual(c.class_sizes[_EX + "Person"], 2)

    def test_type_relation_conditional_is_normalised(self):
        # ex:a typed as Person with two outgoing non-type edges → probs sum to 1
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
            "ex:a rdf:type ex:Person .\n"
            "ex:a ex:p ex:x .\n"
            "ex:a ex:q ex:y .\n"
        )
        c = BlockC().calculate(self._load_ttl(ttl))
        dist = c.type_relation_conditional.get(_EX + "Person")
        self.assertIsNotNone(dist)
        self.assertAlmostEqual(sum(dist.values()), 1.0)
        # rdf:type + ex:p + ex:q = 3 outgoing edges, each with weight 1/3
        self.assertAlmostEqual(dist.get(_EX + "p", 0.0), 1 / 3)
        self.assertAlmostEqual(dist.get(_EX + "q", 0.0), 1 / 3)

    def test_literal_target_included_in_obj_cooc(self):
        # Object-side co-occurrence is built over all targets, including literals
        c = BlockC().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            'ex:s ex:label "hello" .\n'
        ))
        # The literal "hello" appears as target of ex:label → M_obj is 1×1
        self.assertEqual(len(c.obj_singular_values), 10)
        self.assertEqual(len(c.as_vector()), _VECTOR_LEN)

    def test_class_size_zipf_nan_below_min_samples(self):
        # Only 1 distinct class → below MIN_SAMPLES_FOR_FIT → zipf must be NaN
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
            "ex:a rdf:type ex:T .\n"
        )
        c = BlockC().calculate(self._load_ttl(ttl))
        self.assertEqual(c.num_classes, 1)
        self.assertTrue(math.isnan(c.class_size_zipf_exponent))

    def test_singular_values_always_ten(self):
        # Regardless of |R|, subj/obj singular value arrays have length 10
        for ttl in [
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:a ex:q ex:c .\n",
            "@prefix ex: <http://example.org/> .\n"
            + "".join(f"ex:s{i} ex:p{i} ex:o{i} .\n" for i in range(15)),
        ]:
            with self.subTest(ttl=ttl[:60]):
                c = BlockC().calculate(self._load_ttl(ttl))
                self.assertEqual(len(c.subj_singular_values), 10)
                self.assertEqual(len(c.obj_singular_values), 10)

    def test_vector_length_invariant(self):
        for ttl in [
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:a ex:q ex:c .\n",
            "@prefix ex: <http://example.org/> .\n"
            + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(20)),
        ]:
            with self.subTest(ttl=ttl[:60]):
                self.assertEqual(len(BlockC().calculate(self._load_ttl(ttl)).as_vector()), _VECTOR_LEN)


class TestBlockCSerialize(unittest.TestCase):
    def _make(self) -> BlockC:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(
                "@prefix ex: <http://example.org/> .\n"
                + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(10))
            )
        return BlockC().calculate(load_kg(path))

    def test_feature_names_length(self):
        self.assertEqual(len(BlockC.feature_names()), _VECTOR_LEN)

    def test_as_dict_keys_match_feature_names(self):
        c = self._make()
        self.assertEqual(list(c.as_dict().keys()), BlockC.feature_names())

    def test_as_dict_values_match_as_vector(self):
        c = self._make()
        self.assertEqual(list(c.as_dict().values()), c.as_vector())

    def test_serialization_roundtrip(self):
        c = self._make()
        restored = BlockC.from_serializable(c.to_serializable())
        self.assertEqual(c.as_vector(), restored.as_vector())


if __name__ == "__main__":
    unittest.main()
