import math
import os
import sys
import tempfile
import unittest

import igraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_io import load_kg
from signature._orig_block_f import BlockF

_VECTOR_LEN = 6


def _make_g(n: int, edges: list, literals: list | None = None) -> igraph.Graph:
    """Build a directed igraph.Graph with is_literal set on every vertex."""
    g = igraph.Graph(n=n, edges=edges, directed=True)
    g.vs["is_literal"] = literals if literals is not None else [False] * n
    return g


class TestBlockFEdgeCases(unittest.TestCase):
    def test_empty_graph(self):
        g = igraph.Graph(directed=True)
        g.vs["is_literal"] = []
        f = BlockF().calculate(g)
        self.assertEqual(f.num_components, 0)
        self.assertTrue(math.isnan(f.largest_component_fraction))
        self.assertTrue(math.isnan(f.avg_shortest_path_length))
        self.assertTrue(math.isnan(f.clustering_coefficient))
        self.assertTrue(math.isnan(f.degree_assortativity))
        self.assertEqual(len(f.as_vector()), _VECTOR_LEN)

    def test_single_vertex_no_edges(self):
        # 1 non-literal vertex, no edges → can't form a pair → avg_sp NaN
        g = _make_g(1, [])
        f = BlockF().calculate(g)
        self.assertEqual(f.num_components, 1)
        self.assertAlmostEqual(f.largest_component_fraction, 1.0)
        self.assertTrue(math.isnan(f.avg_shortest_path_length))
        self.assertEqual(len(f.as_vector()), _VECTOR_LEN)

    def test_all_literal_vertices(self):
        # non_lit pool is empty → avg_sp NaN, other stats still computed
        g = _make_g(3, [(0, 1), (1, 2)], literals=[True, True, True])
        f = BlockF().calculate(g)
        self.assertEqual(f.num_components, 1)
        self.assertTrue(math.isnan(f.avg_shortest_path_length))
        self.assertEqual(len(f.as_vector()), _VECTOR_LEN)

    def test_one_literal_one_non_literal(self):
        # Only 1 non-literal → pool size < 2 → avg_sp NaN
        g = _make_g(2, [(0, 1)], literals=[False, True])
        f = BlockF().calculate(g)
        self.assertTrue(math.isnan(f.avg_shortest_path_length))
        self.assertEqual(len(f.as_vector()), _VECTOR_LEN)


class TestBlockFConnectivity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as fh:
            fh.write(content)
        return load_kg(path)

    def test_single_edge(self):
        # Two non-literal vertices, one edge; only non-self pair has distance 1
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:s ex:p ex:o .\n"
        )
        f = BlockF().calculate(g)
        self.assertEqual(f.num_components, 1)
        self.assertAlmostEqual(f.largest_component_fraction, 1.0)
        self.assertAlmostEqual(f.avg_shortest_path_length, 1.0)
        # No vertex has ≥2 neighbours → local clustering = 0
        self.assertAlmostEqual(f.clustering_coefficient, 0.0)
        self.assertEqual(len(f.as_vector()), _VECTOR_LEN)

    def test_triangle_clustering_and_path(self):
        # Directed 3-cycle → undirected triangle
        # All 3 pairwise distances = 1; every vertex has 2 neighbours that are
        # themselves connected → local clustering = 1.0
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:q ex:c .\n"
            "ex:c ex:r ex:a .\n"
        )
        f = BlockF().calculate(g)
        self.assertEqual(f.num_components, 1)
        self.assertAlmostEqual(f.largest_component_fraction, 1.0)
        self.assertAlmostEqual(f.avg_shortest_path_length, 1.0)
        self.assertAlmostEqual(f.clustering_coefficient, 1.0)
        self.assertEqual(len(f.as_vector()), _VECTOR_LEN)

    def test_two_components_count_and_lcc_fraction(self):
        # Component 1: a-b (2 nodes); Component 2: c-d-e chain (3 nodes)
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:c ex:p ex:d .\n"
            "ex:d ex:p ex:e .\n"
        )
        g = self._load_ttl(ttl)
        f = BlockF().calculate(g)
        self.assertEqual(f.num_components, 2)
        self.assertAlmostEqual(f.largest_component_fraction, 3 / 5)
        # avg_sp sampled within LCC (c-d-e chain); distances are 1 or 2
        self.assertGreaterEqual(f.avg_shortest_path_length, 1.0)
        self.assertLessEqual(f.avg_shortest_path_length, 2.0)
        self.assertEqual(len(f.as_vector()), _VECTOR_LEN)

    def test_lcc_fraction_denominator_includes_literals(self):
        # ex:a ex:p ex:b + ex:b ex:label "hello"
        # Vertices: ex:a, ex:b (non-literal) + "hello" (literal) = 3 total
        # All 3 are in one component → lcc_fraction = 3/3 = 1.0
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            'ex:b ex:label "hello" .\n'
        )
        f = BlockF().calculate(g)
        self.assertEqual(f.num_components, 1)
        self.assertAlmostEqual(f.largest_component_fraction, 1.0)
        # Only non-literal vertices (ex:a, ex:b) sampled → avg_sp = 1.0
        self.assertAlmostEqual(f.avg_shortest_path_length, 1.0)
        self.assertEqual(len(f.as_vector()), _VECTOR_LEN)

    def test_negative_assortativity_star(self):
        # Star: hub (deg 4) connected to 4 leaves (deg 1) → hub-leaf pairs
        # → Pearson r(deg_src, deg_tgt) is negative
        g = _make_g(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
        f = BlockF().calculate(g)
        self.assertFalse(math.isnan(f.degree_assortativity))
        self.assertLess(f.degree_assortativity, 0.0)

    def test_chain_no_clustering(self):
        # Linear chain: a→b→c→d — no triangles → clustering = 0
        g = _make_g(4, [(0, 1), (1, 2), (2, 3)])
        f = BlockF().calculate(g)
        self.assertAlmostEqual(f.clustering_coefficient, 0.0)

    def test_avg_sp_on_isolated_pairs_equals_one(self):
        # 10 disconnected s_i-o_i pairs — all edges have distance 1 within
        # each pair; the LCC is one such pair (2 non-lit nodes, distance 1)
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(10))
        )
        g = self._load_ttl(ttl)
        f1 = BlockF().calculate(g, sample_k=1)
        f2 = BlockF().calculate(g, sample_k=2)
        self.assertAlmostEqual(f1.avg_shortest_path_length, 1.0)
        self.assertAlmostEqual(f2.avg_shortest_path_length, 1.0)

    def test_se_nan_when_avg_sp_nan(self):
        # Empty graph → avg_sp NaN → SE must also be NaN
        g = igraph.Graph(directed=True)
        g.vs["is_literal"] = []
        f = BlockF().calculate(g)
        self.assertTrue(math.isnan(f.avg_shortest_path_length_se))

    def test_se_finite_and_non_negative_on_chain(self):
        # Chain a-b-c: distances are 1 and 2 → variance > 0 → SE > 0
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:q ex:c .\n"
        )
        f = BlockF().calculate(g, sample_k=2)
        self.assertFalse(math.isnan(f.avg_shortest_path_length_se))
        self.assertGreater(f.avg_shortest_path_length_se, 0.0)

    def test_se_zero_on_triangle(self):
        # Triangle: all pairwise distances == 1 → no variance → SE == 0
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:q ex:c .\n"
            "ex:c ex:r ex:a .\n"
        )
        f = BlockF().calculate(g, sample_k=2)
        self.assertAlmostEqual(f.avg_shortest_path_length_se, 0.0)

    def test_sample_k_larger_gives_finite_result(self):
        # Increasing k should not break anything; result stays finite
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:q ex:c .\n"
            "ex:c ex:r ex:a .\n"
        )
        f = BlockF().calculate(g, sample_k=3)
        self.assertFalse(math.isnan(f.avg_shortest_path_length))
        self.assertEqual(len(f.as_vector()), _VECTOR_LEN)

    def test_vector_length_invariant(self):
        graphs = [
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:b ex:q ex:c .\n",
            "@prefix ex: <http://example.org/> .\n"
            + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(20)),
        ]
        for ttl in graphs:
            with self.subTest(ttl=ttl[:60]):
                g = self._load_ttl(ttl)
                self.assertEqual(len(BlockF().calculate(g).as_vector()), _VECTOR_LEN)


class TestBlockFSerialize(unittest.TestCase):
    def _make(self) -> BlockF:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(
                "@prefix ex: <http://example.org/> .\n"
                + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(5))
            )
        return BlockF().calculate(load_kg(path))

    def test_feature_names_length(self):
        self.assertEqual(len(BlockF.feature_names()), _VECTOR_LEN)

    def test_as_dict_keys_match_feature_names(self):
        f = self._make()
        self.assertEqual(list(f.as_dict().keys()), BlockF.feature_names())

    def test_as_dict_values_match_as_vector(self):
        f = self._make()
        self.assertEqual(list(f.as_dict().values()), f.as_vector())

    def test_serialization_roundtrip(self):
        f = self._make()
        restored = BlockF.from_serializable(f.to_serializable())
        self.assertEqual(f.as_vector(), restored.as_vector())


if __name__ == "__main__":
    unittest.main()
