import math
import os
import tempfile
import unittest

import igraph
import numpy as np
from kgsynth.kg_io import load_kg
from kgsynth.signature import BlockB
from kgsynth.signature._utils import PowerLawStats

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
                self.assertEqual(
                    len(BlockB().calculate(self._load_ttl(ttl)).as_vector()), _VECTOR_LEN
                )


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


class TestBlockBDegreesExcludeTypes(unittest.TestCase):
    """Degrees are *entity content* degrees: rdf:type edges and class nodes are excluded.

    A class node is not an entity (Stage 2 creates it separately, outside the content-edge
    budget), so counting it puts a class's instance count in the in-degree tail — aids
    measured an in-degree max of 184493, i.e. |class0|, which the generator then imposed
    on entities.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(content)
        return load_kg(path)

    def _typed_graph(self) -> igraph.Graph:
        # a -> b, a -> c content edges; a, b, c all typed ex:C.  So ex:C has in-degree 3
        # (it is the graph's highest-in-degree vertex) and a has out-degree 3 (2 content
        # + 1 rdf:type).
        return self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b . ex:a ex:p ex:c .\n"
            "ex:a a ex:C . ex:b a ex:C . ex:c a ex:C .\n"
        )

    def test_type_edge_not_counted_in_out_degree(self):
        b = BlockB().calculate(self._typed_graph())
        # a's content out-degree is 2, not 3 — its rdf:type edge is wired outside the
        # content budget and must not inflate the target the generator steers toward.
        self.assertEqual(b.out_degree_max, 2)

    def test_class_node_not_an_entity(self):
        b = BlockB().calculate(self._typed_graph())
        # ex:C has in-degree 3 (the graph max) but is a class, not an entity. The content
        # in-degree max is 1 (b and c each receive one ex:p edge).
        self.assertEqual(b.in_degree_max, 1)

    def test_degree_node_set_excludes_class_nodes(self):
        b = BlockB().calculate(self._typed_graph())
        # 3 entities (a, b, c) — the class node ex:C is not one of them.
        self.assertEqual(len(b._out_degrees), 3)
        self.assertEqual(len(b._in_degrees), 3)

    def test_untyped_graph_unaffected(self):
        b = BlockB().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b . ex:a ex:p ex:c .\n"
        ))
        self.assertEqual(b.out_degree_max, 2)
        self.assertEqual(b.in_degree_max, 1)
        self.assertEqual(len(b._out_degrees), 3)


class TestBlockBReciprocity(unittest.TestCase):
    """Per-relation reciprocity: frequency-binned P(symmetric) + symmetric-mode value."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_ttl(self, content: str) -> igraph.Graph:
        path = os.path.join(self.tmp, "g.ttl")
        with open(path, "w") as f:
            f.write(content)
        return load_kg(path)

    def test_all_symmetric_relation_has_frac_one(self):
        # Every edge of ex:p has its reverse also present via ex:p (a<->b, c<->d).
        b = BlockB().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b . ex:b ex:p ex:a .\n"
            "ex:c ex:p ex:d . ex:d ex:p ex:c .\n"
        ))
        self.assertTrue(np.any(b.recip_symmetric_frac > 0.5))
        self.assertGreater(b.recip_symmetric_value, 0.5)

    def test_all_asymmetric_relation_has_frac_zero(self):
        # No reverse edges exist for ex:p anywhere.
        b = BlockB().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b . ex:c ex:p ex:d .\n"
        ))
        frac = b.recip_symmetric_frac
        self.assertTrue(np.all(frac[np.isfinite(frac)] == 0.0))
        self.assertTrue(math.isnan(b.recip_symmetric_value))

    def test_vector_includes_reciprocity(self):
        b = BlockB().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:b ex:p ex:a .\n"
        ))
        self.assertEqual(len(b.as_vector()), _VECTOR_LEN)
        names = BlockB.feature_names()
        self.assertIn("recip_symmetric_value", names)
        self.assertTrue(any(n.startswith("recip_symmetric_frac_bin") for n in names))


if __name__ == "__main__":
    unittest.main()
