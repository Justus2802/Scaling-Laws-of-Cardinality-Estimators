import math
import os
import sys
import tempfile
import unittest

import igraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_io import load_kg
from signature import BlockB, PowerLawStats, block_b


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
        b = block_b(g)
        self.assertTrue(_isnan_stats(b.out_degree_fit))
        self.assertTrue(_isnan_stats(b.in_degree_fit))
        self.assertEqual(b.object_multiplicity, {})
        self.assertEqual(b.subject_multiplicity, {})
        self.assertEqual(b.functionality, {})
        self.assertEqual(b.inverse_functionality, {})
        self.assertEqual(len(b.as_vector()), 68)

    def test_single_triple_short_circuits(self):
        b = block_b(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
        ))
        # 1 sample is far below MIN_SAMPLES_FOR_FIT
        self.assertTrue(_isnan_stats(b.out_degree_fit))
        self.assertTrue(_isnan_stats(b.in_degree_fit))
        # functionality is still computable
        self.assertEqual(b.functionality["http://example.org/p"], 1.0)
        self.assertEqual(b.inverse_functionality["http://example.org/p"], 1.0)
        # multiplicity fit short-circuits to NaN (only 1 subject)
        self.assertTrue(_isnan_stats(b.object_multiplicity["http://example.org/p"]))
        self.assertEqual(len(b.as_vector()), 68)

    def test_functional_relation(self):
        # Each subject has exactly one object → functionality = 1.0
        ttl = "@prefix ex: <http://example.org/> .\n"
        for i in range(5):
            ttl += f"ex:a{i} ex:bornIn ex:c{i} .\n"
        b = block_b(self._load_ttl(ttl))
        self.assertEqual(b.functionality["http://example.org/bornIn"], 1.0)
        self.assertEqual(b.inverse_functionality["http://example.org/bornIn"], 1.0)

    def test_many_to_many_relation(self):
        # ex:a1 ex:knows ex:a2, ex:a3 ; ex:a2 ex:knows ex:a1
        # subjects: a1 has 2 objects, a2 has 1 → functionality = 0.5
        b = block_b(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a1 ex:knows ex:a2 .\n"
            "ex:a1 ex:knows ex:a3 .\n"
            "ex:a2 ex:knows ex:a1 .\n"
        ))
        self.assertLess(b.functionality["http://example.org/knows"], 1.0)
        self.assertEqual(b.functionality["http://example.org/knows"], 0.5)

    def test_vector_length_invariant(self):
        # Any graph → vector length is exactly 68
        for ttl in [
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:c ex:p ex:d .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:a ex:q ex:c .\n",
        ]:
            with self.subTest(ttl=ttl):
                self.assertEqual(len(block_b(self._load_ttl(ttl)).as_vector()), 68)


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
        b = block_b(load_kg(path))
        # 15 positive in-degree samples with real variance → fit should converge
        self.assertFalse(math.isnan(b.in_degree_fit.alpha))
        self.assertFalse(math.isnan(b.in_degree_fit.ks))
        self.assertFalse(math.isnan(b.in_degree_fit.D_lognormal))
        # alpha for a heavy-tailed distribution should land in a sensible range
        self.assertGreater(b.in_degree_fit.alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
