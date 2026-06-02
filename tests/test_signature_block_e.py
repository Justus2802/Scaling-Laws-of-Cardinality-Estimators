import itertools
import math
import os
import sys
import tempfile
import unittest

import igraph
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_io import load_kg
from signature import BlockE

_VECTOR_LEN = 36   # 7 motifs + 9 stars + 9 path_zipf + 9 path_entropy + 2 tree


def _make_g(n: int, edges: list, literals: list | None = None) -> igraph.Graph:
    """Directed igraph.Graph with is_literal and dummy predicates on every edge."""
    g = igraph.Graph(n=n, edges=edges, directed=True)
    g.vs["is_literal"] = literals if literals is not None else [False] * n
    if edges:
        g.es["predicate"] = [f"http://example.org/p{i}" for i in range(len(edges))]
    return g


class TestBlockEEdgeCases(unittest.TestCase):
    def test_empty_graph(self):
        g = igraph.Graph(directed=True)
        g.vs["is_literal"] = []
        e = BlockE().calculate(g)
        self.assertEqual(e.triangle_count, 0)
        self.assertEqual(e.four_cycle_count, 0)
        self.assertEqual(e.five_cycle_count, 0)
        self.assertEqual(e.six_cycle_count, 0)
        self.assertEqual(e.diamond_count, 0)
        self.assertEqual(e.k4_count, 0)
        self.assertEqual(e.tailed_triangle_count, 0)
        for k in range(2, 11):
            self.assertEqual(e.star_counts.get(k, 0), 0)
        self.assertTrue(math.isnan(e.tree_template_zipf))
        self.assertTrue(math.isnan(e.tree_template_entropy))
        self.assertEqual(len(e.as_vector()), _VECTOR_LEN)

    def test_single_edge_all_zeros(self):
        # No cycles or multi-hop paths possible with one edge
        g = _make_g(2, [(0, 1)])
        e = BlockE().calculate(g)
        self.assertEqual(e.triangle_count, 0)
        self.assertEqual(e.four_cycle_count, 0)
        self.assertEqual(len(e.as_vector()), _VECTOR_LEN)


class TestBlockETriangleCount(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(content)
        return load_kg(path)

    def test_directed_3_cycle_gives_one_triangle(self):
        # a→b→c→a: undirected simplification is K3 → exactly 1 triangle
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:q ex:c .\n"
            "ex:c ex:r ex:a .\n"
        )
        self.assertEqual(BlockE().calculate(g).triangle_count, 1)

    def test_chain_has_no_triangles(self):
        # a→b→c→d: no closing edge → triangle_count = 0
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:p ex:c .\n"
            "ex:c ex:p ex:d .\n"
        )
        self.assertEqual(BlockE().calculate(g).triangle_count, 0)

    def test_two_triangles_sharing_an_edge(self):
        # Diamond: a-b-c-a plus a-b-d-a → 2 triangles
        g = _make_g(4, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)])
        self.assertEqual(BlockE().calculate(g).triangle_count, 2)


class TestBlockEStarCounts(unittest.TestCase):
    def test_star_counts_keys_cover_k2_to_k10(self):
        g = _make_g(3, [(0, 1), (0, 2)])
        e = BlockE().calculate(g)
        self.assertEqual(set(e.star_counts.keys()), set(range(2, 11)))

    def test_3_spoke_hub_star(self):
        # Hub (vertex 0) connected to 3 leaves → hub undirected degree = 3
        # 2-stars: C(3,2) = 3; 3-stars: C(3,3) = 1; k≥4: 0
        g = _make_g(4, [(0, 1), (0, 2), (0, 3)])
        e = BlockE().calculate(g)
        self.assertEqual(e.star_counts[2], 3)
        self.assertEqual(e.star_counts[3], 1)
        for k in range(4, 11):
            self.assertEqual(e.star_counts[k], 0)

    def test_no_stars_on_chain(self):
        # Linear chain: max undirected degree = 2 → no 2-stars? No:
        # interior vertex has degree 2 → C(2,2) = 1 two-star per interior vertex
        # chain 0-1-2-3: interior vertices 1 and 2 each have degree 2
        g = _make_g(4, [(0, 1), (1, 2), (2, 3)])
        e = BlockE().calculate(g)
        self.assertEqual(e.star_counts[2], 2)   # vertices 1 and 2 each contribute 1
        self.assertEqual(e.star_counts[3], 0)


class TestBlockEFourNodeMotifs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(content)
        return load_kg(path)

    def test_directed_4_cycle_detects_four_cycle_motif(self):
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:p ex:c .\n"
            "ex:c ex:p ex:d .\n"
            "ex:d ex:p ex:a .\n"
        )
        e = BlockE().calculate(g)
        self.assertEqual(e.four_cycle_count, 1)
        self.assertEqual(e.k4_count, 0)

    def test_k4_graph_has_k4_motif(self):
        # Complete graph on 4 vertices → k4_count = 1
        g = _make_g(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        e = BlockE().calculate(g)
        self.assertEqual(e.k4_count, 1)

    def test_tailed_triangle_detected(self):
        # Triangle 0-1-2 plus pendant edge 2-3 → tailed_triangle_count = 1
        g = _make_g(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
        e = BlockE().calculate(g)
        self.assertEqual(e.tailed_triangle_count, 1)


class TestBlockECycleCounts(unittest.TestCase):
    def test_five_six_cycle_nonnegative(self):
        # Estimates must always be non-negative integers
        g = _make_g(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
        e = BlockE().calculate(g, sample_budget=10_000)
        self.assertGreaterEqual(e.five_cycle_count, 0)
        self.assertGreaterEqual(e.six_cycle_count, 0)

    def test_small_graph_below_k_returns_zero(self):
        # 4 vertices → no 5-cycle or 6-cycle is possible
        g = _make_g(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        e = BlockE().calculate(g)
        self.assertEqual(e.five_cycle_count, 0)
        self.assertEqual(e.six_cycle_count, 0)


class TestBlockEPathTemplates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(content)
        return load_kg(path)

    def test_path_templates_nan_when_no_non_literal_targets(self):
        # All targets are literals → out_adj is empty → templates stay NaN
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            'ex:s ex:p "hello" .\n'
        )
        e = BlockE().calculate(g, sample_budget=1_000)
        for k in range(2, 11):
            self.assertTrue(math.isnan(e.path_template_zipf.get(k, float("nan"))))

    def test_path_templates_populated_on_chain(self):
        # a→b→c: valid 2-hop walk exists → path_template_zipf[2] should be finite
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:q ex:c .\n"
        )
        e = BlockE().calculate(g, sample_budget=10_000)
        self.assertFalse(math.isnan(e.path_template_entropy.get(2, float("nan"))))

    def test_tree_template_populated_on_branching_graph(self):
        # Root with two children, each with one grandchild → depth-2 trees exist
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:root ex:p ex:c1 .\n"
            "ex:root ex:p ex:c2 .\n"
            "ex:c1 ex:q ex:g1 .\n"
            "ex:c2 ex:q ex:g2 .\n"
        )
        e = BlockE().calculate(g, sample_budget=10_000)
        self.assertFalse(math.isnan(e.tree_template_entropy))


class TestBlockEVectorLength(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(content)
        return load_kg(path)

    def test_vector_length_invariant(self):
        graphs = [
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:b ex:q ex:c .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:b ex:q ex:c . ex:c ex:r ex:a .\n",
            "@prefix ex: <http://example.org/> .\n"
            + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(20)),
        ]
        for ttl in graphs:
            with self.subTest(ttl=ttl[:60]):
                g = self._load_ttl(ttl)
                self.assertEqual(len(BlockE().calculate(g, sample_budget=1_000).as_vector()), _VECTOR_LEN)

    def test_vector_length_empty_graph(self):
        g = igraph.Graph(directed=True)
        g.vs["is_literal"] = []
        self.assertEqual(len(BlockE().calculate(g).as_vector()), _VECTOR_LEN)


class TestBlockEDiamondAndK4(unittest.TestCase):
    def test_diamond_counted_once(self):
        # K4 minus one edge = two triangles sharing edge 0-1 → exactly 1 diamond,
        # no K4 and no 4-cycle (the closing diagonal is present).
        g = _make_g(4, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)])
        e = BlockE().calculate(g)
        self.assertEqual(e.diamond_count, 1)
        self.assertEqual(e.k4_count, 0)
        self.assertEqual(e.four_cycle_count, 0)

    def test_k4_has_no_diamond(self):
        # In a complete K4 every potential diamond is closed into the clique,
        # so k4_count = 1 and diamond_count = 0.
        g = _make_g(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        e = BlockE().calculate(g)
        self.assertEqual(e.k4_count, 1)
        self.assertEqual(e.diamond_count, 0)

    def test_k4_star_counts(self):
        # Every vertex in K4 has undirected degree 3 → 4 vertices contribute:
        # 2-stars: 4·C(3,2)=12, 3-stars: 4·C(3,3)=4, k≥4: 0
        g = _make_g(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        e = BlockE().calculate(g)
        self.assertEqual(e.star_counts[2], 12)
        self.assertEqual(e.star_counts[3], 4)
        for k in range(4, 11):
            self.assertEqual(e.star_counts[k], 0)


class TestBlockEKCycleDetection(unittest.TestCase):
    def test_complete_k6_detects_five_and_six_cycles(self):
        # K6 is densely connected → walk-closure sampling reliably finds
        # 5- and 6-cycles, so both estimates are strictly positive.
        edges = list(itertools.combinations(range(6), 2))
        g = _make_g(6, edges)
        e = BlockE().calculate(g, sample_budget=100_000)
        self.assertGreater(e.five_cycle_count, 0)
        self.assertGreater(e.six_cycle_count, 0)


class TestBlockELiteralHandling(unittest.TestCase):
    def test_literal_targets_excluded_from_path_walks(self):
        # 0→1 with target vertex 1 flagged as a literal → no walkable start
        # vertices → path/tree templates fall back to NaN/empty.
        g = _make_g(2, [(0, 1)], literals=[False, True])
        e = BlockE().calculate(g, sample_budget=1_000)
        self.assertEqual(e.path_template_zipf, {})
        self.assertEqual(e.path_template_entropy, {})
        self.assertTrue(math.isnan(e.tree_template_zipf))
        self.assertTrue(math.isnan(e.tree_template_entropy))


class TestBlockEDeterminism(unittest.TestCase):
    def test_repeated_calculation_is_reproducible(self):
        # Sampling uses fixed RNG seeds → two runs on the same graph must
        # produce byte-identical vectors (NaN-aware comparison).
        g = _make_g(6, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0), (3, 4), (4, 5)])
        v1 = BlockE().calculate(g, sample_budget=10_000).as_vector()
        v2 = BlockE().calculate(g, sample_budget=10_000).as_vector()
        np.testing.assert_array_equal(v1, v2)


class TestBlockEContract(unittest.TestCase):
    """Base-class contract: lifecycle guards, naming, serialization."""

    def _make(self) -> BlockE:
        g = _make_g(4, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)])
        return BlockE().calculate(g, sample_budget=10_000)

    def test_accessing_property_before_calculate_raises(self):
        e = BlockE()
        with self.assertRaises(RuntimeError):
            _ = e.triangle_count

    def test_calculate_returns_self(self):
        g = igraph.Graph(directed=True)
        g.vs["is_literal"] = []
        e = BlockE()
        self.assertIs(e.calculate(g), e)

    def test_feature_names_length_matches_vector(self):
        self.assertEqual(len(BlockE.feature_names()), _VECTOR_LEN)

    def test_get_na_vec_is_all_nan(self):
        na = BlockE.get_na_vec()
        self.assertEqual(len(na), _VECTOR_LEN)
        self.assertTrue(all(math.isnan(x) for x in na))

    def test_as_dict_keys_match_feature_names(self):
        self.assertEqual(list(self._make().as_dict().keys()), BlockE.feature_names())

    def test_as_dict_values_match_as_vector(self):
        e = self._make()
        self.assertEqual(list(e.as_dict().values()), e.as_vector())

    def test_serialization_roundtrip(self):
        e = self._make()
        restored = BlockE.from_serializable(e.to_serializable())
        np.testing.assert_array_equal(e.as_vector(), restored.as_vector())


class TestBlockEVisualize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make(self) -> BlockE:
        g = _make_g(4, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)])
        return BlockE().calculate(g, sample_budget=10_000)

    def test_text_mode_writes_summary_to_file(self):
        out = os.path.join(self.tmp, "e.txt")
        self._make().visualize(mode="text", path=out)
        with open(out) as f:
            content = f.read()
        self.assertIn("Block E", content)
        self.assertIn("triangles", content)

    def test_plot_mode_writes_png_to_file(self):
        out = os.path.join(self.tmp, "e.png")
        self._make().visualize(mode="plot", path=out)
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self._make().visualize(mode="bogus")


if __name__ == "__main__":
    unittest.main()
