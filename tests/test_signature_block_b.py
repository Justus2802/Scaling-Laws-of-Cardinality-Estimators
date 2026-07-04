import math
import os
import sys
import tempfile
import unittest

import igraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_io import load_kg
from signature import BlockB
from signature._utils import PowerLawStats

_VECTOR_LEN = len(BlockB.feature_names())


def _isnan_stats(stats: PowerLawStats) -> bool:
    return all(math.isnan(v) for v in stats)


class TestBlockBSmallFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(content)
        return load_kg(path)

    def test_empty_graph(self):
        g = igraph.Graph(directed=True)
        g.vs["is_literal"] = []  # ensure attribute exists for empty vertex set
        b = BlockB().calculate(g)
        self.assertTrue(_isnan_stats(b.out_degree_fit))
        self.assertTrue(_isnan_stats(b.in_degree_fit))
        self.assertEqual(len(b.as_vector()), _VECTOR_LEN)

    def test_single_triple_short_circuits(self):
        b = BlockB().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
        ))
        # 1 sample is far below MIN_SAMPLES_FOR_FIT → aggregate degree fits stay NaN
        self.assertTrue(_isnan_stats(b.out_degree_fit))
        self.assertTrue(_isnan_stats(b.in_degree_fit))
        self.assertEqual(len(b.as_vector()), _VECTOR_LEN)

    def test_vector_length_invariant(self):
        for ttl in [
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:c ex:p ex:d .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:a ex:q ex:c .\n",
        ]:
            with self.subTest(ttl=ttl):
                self.assertEqual(len(BlockB().calculate(self._load_ttl(ttl)).as_vector()), _VECTOR_LEN)


class TestBlockBPowerLawFit(unittest.TestCase):
    """A degree distribution with enough samples should produce non-NaN fit values."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_heavy_tailed_in_degree_fits(self):
        # Construct an in-degree distribution with genuine heavy-tail variance:
        # 15 objects with in-degrees [10, 5, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1].
        ttl = "@prefix ex: <http://example.org/> .\n"
        in_degrees = [10, 5, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        src_id = 0
        for obj_idx, deg in enumerate(in_degrees):
            for _ in range(deg):
                ttl += f"ex:s{src_id} ex:p ex:o{obj_idx} .\n"
                src_id += 1
        path = os.path.join(self.tmp, "heavy.ttl")
        with open(path, "w") as f:
            f.write(ttl)
        b = BlockB().calculate(load_kg(path))
        # 15 positive in-degree samples with real variance → fit should converge
        self.assertFalse(math.isnan(b.in_degree_fit.alpha))
        # alpha for a heavy-tailed distribution should land in a sensible range
        self.assertGreater(b.in_degree_fit.alpha, 1.0)


class TestBlockBSerialize(unittest.TestCase):
    def _make(self) -> BlockB:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(
                "@prefix ex: <http://example.org/> .\n"
                + "".join(f"ex:s ex:p ex:o{i} .\n" for i in range(15))
            )
        return BlockB().calculate(load_kg(path))

    def test_feature_names_length(self):
        self.assertEqual(len(BlockB.feature_names()), _VECTOR_LEN)

    def test_as_dict_keys_match_feature_names(self):
        b = self._make()
        self.assertEqual(list(b.as_dict().keys()), BlockB.feature_names())

    def test_as_dict_values_match_as_vector(self):
        import numpy as np
        b = self._make()
        np.testing.assert_array_equal(list(b.as_dict().values()), b.as_vector())

    def test_serialization_roundtrip(self):
        import numpy as np
        b = self._make()
        restored = BlockB.from_serializable(b.to_serializable())
        np.testing.assert_array_equal(b.as_vector(), restored.as_vector())


if __name__ == "__main__":
    unittest.main()
