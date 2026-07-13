import os
import tempfile
import unittest

import igraph
from kgsynth.kg_io import load_kg
from kgsynth.signature import BlockA

_VECTOR_LEN = 4   # num_entities, num_relations, mean_degree, type_edge_frac


class TestBlockASmallFixtures(unittest.TestCase):
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
        a = BlockA().calculate(g)
        self.assertEqual(a.num_entities, 0)
        self.assertEqual(a.num_relations, 0)
        self.assertEqual(a.mean_degree, 0.0)
        self.assertEqual(len(a.as_vector()), _VECTOR_LEN)

    def test_single_triple_values(self):
        # ex:s ex:p ex:o — 2 non-literal entities, 1 triple, 1 relation
        a = BlockA().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:s ex:p ex:o .\n"
        ))
        self.assertEqual(a.num_entities, 2)
        self.assertEqual(a.num_relations, 1)
        self.assertAlmostEqual(a.mean_degree, 0.5)   # E/V = 1/2
        self.assertEqual(len(a.as_vector()), _VECTOR_LEN)

    def test_literals_excluded_from_entity_count(self):
        # Literal object "hello" must NOT count toward |V|
        a = BlockA().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            'ex:s ex:label "hello" .\n'
        ))
        self.assertEqual(a.num_entities, 1)    # only ex:s; "hello" is a literal
        self.assertEqual(a.num_relations, 1)
        self.assertAlmostEqual(a.mean_degree, 1.0)   # 1 triple / 1 entity

    def test_mean_degree_is_edges_over_entities(self):
        # 3 non-literal entities, 2 triples → mean_degree = 2/3
        a = BlockA().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:b ex:q ex:c .\n"
        ))
        self.assertEqual(a.num_entities, 3)
        self.assertAlmostEqual(a.mean_degree, 2 / 3)

    def test_relation_count_distinct(self):
        # Two triples sharing the same predicate → |R| = 1, not 2
        a = BlockA().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:c ex:p ex:d .\n"
        ))
        self.assertEqual(a.num_relations, 1)

    def test_relation_count_multiple_predicates(self):
        # 3 triples using 2 distinct predicates
        a = BlockA().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:p ex:b .\n"
            "ex:a ex:p ex:c .\n"
            "ex:a ex:q ex:b .\n"
        ))
        self.assertEqual(a.num_relations, 2)

    def test_rdf_type_included_in_relations(self):
        # rdf:type is not special-cased: counts toward |E| and |R|
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
            "ex:s rdf:type ex:Person .\n"
            "ex:s ex:name ex:n .\n"
        )
        a = BlockA().calculate(self._load_ttl(ttl))
        self.assertEqual(a.num_relations, 2)

    def test_multiple_literals_only_one_entity(self):
        # One subject with two literal objects; only the subject is an entity
        a = BlockA().calculate(self._load_ttl(
            "@prefix ex: <http://example.org/> .\n"
            'ex:s ex:label "hello" .\n'
            'ex:s ex:age "42"^^<http://www.w3.org/2001/XMLSchema#integer> .\n'
        ))
        self.assertEqual(a.num_entities, 1)
        self.assertAlmostEqual(a.mean_degree, 2.0)   # 2 triples / 1 entity

    def test_vector_length_invariant(self):
        for ttl in [
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b .\n",
            "@prefix ex: <http://example.org/> .\nex:a ex:p ex:b . ex:b ex:q ex:c .\n",
            "@prefix ex: <http://example.org/> .\n"
            + "".join(f"ex:s{i} ex:p ex:o{i} .\n" for i in range(20)),
        ]:
            with self.subTest(ttl=ttl[:60]):
                self.assertEqual(
                    len(BlockA().calculate(self._load_ttl(ttl)).as_vector()), _VECTOR_LEN
                )


class TestBlockASerialize(unittest.TestCase):
    def _make(self) -> BlockA:
        g = igraph.Graph(directed=True)
        g.add_vertices(2)
        g.vs["name"] = ["http://example.org/s", "http://example.org/o"]
        g.vs["is_literal"] = [False, False]
        g.add_edges([(0, 1)])
        g.es["predicate"] = ["http://example.org/p"]
        return BlockA().calculate(g)

    def test_feature_names_length(self):
        self.assertEqual(len(BlockA.feature_names()), _VECTOR_LEN)

    def test_as_dict_keys_match_feature_names(self):
        a = self._make()
        self.assertEqual(list(a.as_dict().keys()), BlockA.feature_names())

    def test_as_dict_values_match_as_vector(self):
        a = self._make()
        self.assertEqual(list(a.as_dict().values()), a.as_vector())

    def test_serialization_roundtrip(self):
        a = self._make()
        restored = BlockA.from_serializable(a.to_serializable())
        self.assertEqual(a.as_vector(), restored.as_vector())
        self.assertEqual(a.as_dict(), restored.as_dict())


if __name__ == "__main__":
    unittest.main()
