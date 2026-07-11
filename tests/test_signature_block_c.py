import math
import os
import tempfile
import unittest

import igraph
from kgsynth.kg_io import load_kg
from kgsynth.signature import BlockC

_VECTOR_LEN = len(BlockC.feature_names())
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
        self.assertTrue(math.isnan(c.class_size_fit.alpha))
        self.assertEqual(len(c.as_vector()), _VECTOR_LEN)

    def test_single_triple_no_type(self):
        # One predicate → 1×1 co-occurrence matrix, no type info
        c = BlockC().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:s ex:p ex:o .\n"
        ))
        self.assertEqual(c.num_classes, 0)
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

    def test_class_count_distinct_subjects(self):
        # Two entities both typed as ex:Person → one class
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
            "ex:a rdf:type ex:Person .\n"
            "ex:b rdf:type ex:Person .\n"
        )
        c = BlockC().calculate(self._load_ttl(ttl))
        self.assertEqual(c.num_classes, 1)

    def test_class_size_zipf_nan_below_min_samples(self):
        # Only 1 distinct class → below MIN_SAMPLES_FOR_FIT → zipf must be NaN
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
            "ex:a rdf:type ex:T .\n"
        )
        c = BlockC().calculate(self._load_ttl(ttl))
        self.assertEqual(c.num_classes, 1)
        self.assertTrue(math.isnan(c.class_size_fit.alpha))

    def test_vector_length_invariant(self):
        for ttl in [
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:a ex:q ex:c .\n",
            "@prefix ex: <http://example.org/> .\n"
            + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(20)),
        ]:
            with self.subTest(ttl=ttl[:60]):
                self.assertEqual(
                    len(BlockC().calculate(self._load_ttl(ttl)).as_vector()), _VECTOR_LEN
                )


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
        import numpy as np
        c = self._make()
        np.testing.assert_array_equal(list(c.as_dict().values()), c.as_vector())

    def test_serialization_roundtrip(self):
        import numpy as np
        c = self._make()
        restored = BlockC.from_serializable(c.to_serializable())
        np.testing.assert_array_equal(c.as_vector(), restored.as_vector())


if __name__ == "__main__":
    unittest.main()
