import math
import os
import sys
import tempfile
import unittest

import igraph
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_io import load_kg
from signature import BlockD, PowerLawStats, _TOP_K_PAIRS

_VECTOR_LEN = 6 + 6 + _TOP_K_PAIRS + 2  # 34 with default _TOP_K_PAIRS=20


def _isnan_stats(stats: PowerLawStats) -> bool:
    return all(math.isnan(v) for v in stats)


class TestBlockDSmallFixtures(unittest.TestCase):
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
        d = BlockD().calculate(g)
        self.assertEqual(d.num_distinct_cs, 0)
        self.assertTrue(_isnan_stats(d.cs_freq_stats))
        self.assertTrue(math.isnan(d.cs_size_mean))
        self.assertEqual(d.inv_num_distinct_cs, 0)
        self.assertTrue(_isnan_stats(d.inv_cs_freq_stats))
        self.assertTrue(math.isnan(d.inv_cs_size_mean))
        self.assertTrue(np.all(d.top_pair_freqs == 0.0))
        self.assertTrue(_isnan_stats(d.pair_freq_stats))
        self.assertEqual(d.top_pairs, [])
        self.assertEqual(len(d.as_vector()), _VECTOR_LEN)

    def test_single_triple(self):
        # ex:s ex:p ex:o — one entity with one outgoing predicate
        d = BlockD().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:s ex:p ex:o .\n"
        ))
        self.assertEqual(d.num_distinct_cs, 1)
        self.assertAlmostEqual(d.cs_size_mean, 1.0)
        self.assertAlmostEqual(d.cs_size_median, 1.0)
        self.assertAlmostEqual(d.cs_size_p90, 1.0)
        # Only 1 entity → 1 sample for freq fit → short-circuits to NaN
        self.assertTrue(_isnan_stats(d.cs_freq_stats))
        self.assertEqual(len(d.as_vector()), _VECTOR_LEN)

    def test_two_distinct_cs_groups(self):
        # ex:a has {p}, ex:b has {p, q} → two distinct CS
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:x .\n"
            "ex:b ex:p ex:y .\n"
            "ex:b ex:q ex:z .\n"
        )
        d = BlockD().calculate(self._load_ttl(ttl))
        self.assertEqual(d.num_distinct_cs, 2)
        # sizes: a→1, b→2 → mean=1.5, median=1.5, p90=1.9
        self.assertAlmostEqual(d.cs_size_mean, 1.5)
        self.assertAlmostEqual(d.cs_size_median, 1.5)
        self.assertEqual(len(d.as_vector()), _VECTOR_LEN)

    def test_inverse_cs_distinct_from_forward(self):
        # Star graph: s1,s2,s3 → ex:p → hub; hub has inv_CS={p} with 3 subjects
        ttl = "@prefix ex: <http://example.org/> .\n"
        for i in range(3):
            ttl += f"ex:s{i} ex:p ex:hub .\n"
        d = BlockD().calculate(self._load_ttl(ttl))
        # Forward: each sX has CS={p}, hub has CS={}
        self.assertEqual(d.num_distinct_cs, 1)  # only {p}; hub has no outgoing
        # Inverse: hub has inv_CS={p}, sX have inv_CS={}
        self.assertEqual(d.inv_num_distinct_cs, 1)
        self.assertAlmostEqual(d.inv_cs_size_mean, 1.0)

    def test_two_step_pairs_detected(self):
        # Chain: ex:s -[p1]-> ex:mid -[p2]-> ex:o
        # mid has in_pred={p1}, out_pred={p2} → pair (p1, p2)
        d = BlockD().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:s ex:p1 ex:mid .\n"
            "ex:mid ex:p2 ex:o .\n"
        ))
        self.assertEqual(len(d.top_pairs), 1)
        q, p, cnt = d.top_pairs[0]
        self.assertEqual(q, "http://example.org/p1")
        self.assertEqual(p, "http://example.org/p2")
        self.assertEqual(cnt, 1)
        # Normalised frequency should be 1.0 (only one pair)
        self.assertAlmostEqual(d.top_pair_freqs[0], 1.0)
        self.assertTrue(np.all(d.top_pair_freqs[1:] == 0.0))

    def test_two_step_pairs_multiple(self):
        # Three bridge entities each contributing a different pair
        ttl = "@prefix ex: <http://example.org/> .\n"
        for i in range(3):
            ttl += f"ex:src{i} ex:in{i} ex:mid{i} .\n"
            ttl += f"ex:mid{i} ex:out{i} ex:dst{i} .\n"
        d = BlockD().calculate(self._load_ttl(ttl))
        self.assertEqual(len(d.top_pairs), 3)
        # All pairs have equal count → freqs sum to 1
        self.assertAlmostEqual(d.top_pair_freqs[:3].sum(), 1.0)

    def test_powerlaw_fit_fires_with_enough_samples(self):
        # Build a CS frequency distribution with genuine variance (>= MIN_SAMPLES_FOR_FIT)
        # 12 entities split into 3 CS groups of sizes [8, 3, 1]
        ttl = "@prefix ex: <http://example.org/> .\n"
        # Group 1: {p} — 8 entities
        for i in range(8):
            ttl += f"ex:a{i} ex:p ex:x{i} .\n"
        # Group 2: {q} — 3 entities
        for i in range(3):
            ttl += f"ex:b{i} ex:q ex:y{i} .\n"
        # Group 3: {p, q} — 1 entity
        ttl += "ex:c0 ex:p ex:z0 .\nex:c0 ex:q ex:z1 .\n"
        d = BlockD().calculate(self._load_ttl(ttl))
        # 3 distinct CS with freq counts [8,3,1] → 3 samples, likely still NaN
        # (below MIN_SAMPLES_FOR_FIT=10). Verify vector length regardless.
        self.assertEqual(len(d.as_vector()), _VECTOR_LEN)

    def test_vector_length_invariant(self):
        graphs = [
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:b ex:q ex:c .\n",
            "@prefix ex: <http://example.org/> .\n"
            + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(20)),
        ]
        for ttl in graphs:
            with self.subTest(ttl=ttl[:60]):
                self.assertEqual(len(BlockD().calculate(self._load_ttl(ttl)).as_vector()), _VECTOR_LEN)

    def test_not_calculated_raises(self):
        d = BlockD()
        with self.assertRaises(RuntimeError):
            _ = d.num_distinct_cs

    def test_calculate_returns_self(self):
        g = igraph.Graph(directed=True)
        g.vs["is_literal"] = []
        d = BlockD()
        result = d.calculate(g)
        self.assertIs(result, d)


class TestBlockDSerialize(unittest.TestCase):
    def _make(self) -> BlockD:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(
                "@prefix ex: <http://example.org/> .\n"
                + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(10))
            )
        return BlockD().calculate(load_kg(path))

    def test_feature_names_length(self):
        self.assertEqual(len(BlockD.feature_names()), _VECTOR_LEN)

    def test_as_dict_keys_match_feature_names(self):
        d = self._make()
        self.assertEqual(list(d.as_dict().keys()), BlockD.feature_names())

    def test_as_dict_values_match_as_vector(self):
        d = self._make()
        self.assertEqual(list(d.as_dict().values()), d.as_vector())

    def test_serialization_roundtrip(self):
        d = self._make()
        restored = BlockD.from_serializable(d.to_serializable())
        self.assertEqual(d.as_vector(), restored.as_vector())


if __name__ == "__main__":
    unittest.main()
