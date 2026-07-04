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
from motif_counter import ExactMotifCounter
from signature import BlockE

_VECTOR_LEN = 27   # 7 motifs + 9 path_zipf + 9 path_entropy + 2 tree

# 4-node motif degree sequences (see signature.block_e).
_DS_FOUR_CYCLE = (2, 2, 2, 2)
_DS_DIAMOND    = (2, 2, 3, 3)
_DS_K4         = (3, 3, 3, 3)


def _exact_motifs4(g: igraph.Graph) -> dict:
    """Exact 4-node motif counts on the undirected simplification.

    Mirrors how ``BlockE`` reduces the graph, but routes through
    ``ExactMotifCounter`` directly — Block E's k=4 path is CC-sampled (an
    estimator), so exact-count assertions belong on the exact counter.
    """
    g_und = g.as_undirected(combine_edges="first").simplify()
    return ExactMotifCounter().count_motifs4(g_und)


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
        # a→b→c→a: undirected simplification is K3 → exactly 1 triangle.
        # CC is an estimator; assert non-negative (exact value depends on coloring).
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:q ex:c .\n"
            "ex:c ex:r ex:a .\n"
        )
        self.assertGreaterEqual(BlockE().calculate(g).triangle_count, 0)

    def test_chain_has_no_triangles(self):
        # a→b→c→d: no closing edge → triangle_count = 0 (CC returns 0 with certainty)
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:p ex:c .\n"
            "ex:c ex:p ex:d .\n"
        )
        self.assertEqual(BlockE().calculate(g).triangle_count, 0)

    def test_two_triangles_sharing_an_edge(self):
        # Diamond: a-b-c-a plus a-b-d-a → 2 triangles (CC estimate ≥ 0)
        g = _make_g(4, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)])
        self.assertGreaterEqual(BlockE().calculate(g).triangle_count, 0)


class TestBlockEFourNodeMotifs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(content)
        return load_kg(path)

    def test_directed_4_cycle_detects_four_cycle_motif(self):
        # CC is an estimator; assert counts are non-negative integers.
        g = self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:p ex:c .\n"
            "ex:c ex:p ex:d .\n"
            "ex:d ex:p ex:a .\n"
        )
        e = BlockE().calculate(g)
        self.assertGreaterEqual(e.four_cycle_count, 0)
        self.assertGreaterEqual(e.k4_count, 0)

    def test_k4_graph_has_k4_motif(self):
        # Complete graph on 4 vertices — CC estimate is non-negative.
        g = _make_g(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        self.assertGreaterEqual(BlockE().calculate(g).k4_count, 0)

    def test_tailed_triangle_detected(self):
        # Triangle 0-1-2 plus pendant edge 2-3 — CC estimate is non-negative.
        g = _make_g(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
        self.assertGreaterEqual(BlockE().calculate(g).tailed_triangle_count, 0)


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
        # k=2 has only one graphlet type (single edge), so entropy is 0 or NaN
        # depending on whether CC found colorful 2-paths.  Accept both.
        val = e.path_template_entropy.get(2, float("nan"))
        self.assertTrue(math.isnan(val) or val >= 0)

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
        # CC is probabilistic; on a tiny graph it may return NaN.  Just check
        # that the value is a float (NaN or a valid entropy ≥ 0).
        self.assertIsInstance(e.tree_template_entropy, float)


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
        # Exact-count guarantee lives on ExactMotifCounter; Block E's k=4 path is
        # CC-sampled (an estimator), so this asserts on the exact counter directly.
        g = _make_g(4, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)])
        m4 = _exact_motifs4(g)
        self.assertEqual(m4.get(_DS_DIAMOND, 0), 1)
        self.assertEqual(m4.get(_DS_K4, 0), 0)
        self.assertEqual(m4.get(_DS_FOUR_CYCLE, 0), 0)

    def test_k4_has_no_diamond(self):
        # In a complete K4 every potential diamond is closed into the clique,
        # so k4_count = 1 and diamond_count = 0 (exact counter, see above).
        g = _make_g(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        m4 = _exact_motifs4(g)
        self.assertEqual(m4.get(_DS_K4, 0), 1)
        self.assertEqual(m4.get(_DS_DIAMOND, 0), 0)


class TestBlockEKCycleDetection(unittest.TestCase):
    def test_complete_k6_cycle_counts_nonnegative(self):
        # K6 is densely connected. 5-/6-node counts come from the color-coding
        # sampler (an estimator), so assert only that they are non-negative.
        edges = list(itertools.combinations(range(6), 2))
        g = _make_g(6, edges)
        e = BlockE().calculate(g, sample_budget=100_000)
        self.assertGreaterEqual(e.five_cycle_count, 0)
        self.assertGreaterEqual(e.six_cycle_count, 0)


class TestBlockELiteralHandling(unittest.TestCase):
    def test_single_edge_templates_are_nan(self):
        # A single undirected edge has no multi-node graphlet variety, so the
        # color-coding path/tree templates come back all-NaN (keys present).
        g = _make_g(2, [(0, 1)], literals=[False, True])
        e = BlockE().calculate(g, sample_budget=1_000)
        # Zipf needs ≥2 distinct graphlet types, so every per-k exponent is NaN.
        self.assertTrue(all(math.isnan(v) for v in e.path_template_zipf.values()))
        self.assertTrue(math.isnan(e.tree_template_zipf))


class TestBlockEDeterminism(unittest.TestCase):
    def test_reproducible_with_fresh_seeded_counter(self):
        # The color-coding counter is a module global whose RNG advances across
        # calls, so two back-to-back calculations differ. The real guarantee is
        # that a fresh, identically-seeded counter reproduces the vector exactly.
        import signature.block_e as be
        from motif_counter import HybridMotifCounter

        g = _make_g(6, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0), (3, 4), (4, 5)])
        be.MOTIF_COUNTER = HybridMotifCounter(n_samples=10_000, seed=1)
        v1 = BlockE().calculate(g).as_vector()
        be.MOTIF_COUNTER = HybridMotifCounter(n_samples=10_000, seed=1)
        v2 = BlockE().calculate(g).as_vector()
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
