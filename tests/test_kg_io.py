import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_io import load_kg, save_kg

TTL_SAMPLE = """\
@prefix ex: <http://example.org/> .
ex:Alice ex:knows ex:Bob .
ex:Alice ex:age "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
ex:Bob   ex:name "Bob"@en .
"""

NT_SAMPLE = """\
<http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> .
<http://example.org/Alice> <http://example.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
"""


class TestLoadKG(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write(self, filename, content):
        path = os.path.join(self.tmp, filename)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_load_ttl_vertex_and_edge_count(self):
        path = self._write("sample.ttl", TTL_SAMPLE)
        g = load_kg(path)
        self.assertEqual(g.vcount(), 4)  # Alice, Bob, "30"^^xsd:int, "Bob"@en
        self.assertEqual(g.ecount(), 3)

    def test_load_nt_vertex_and_edge_count(self):
        path = self._write("sample.nt", NT_SAMPLE)
        g = load_kg(path)
        self.assertEqual(g.vcount(), 3)  # Alice, Bob, "30"^^xsd:int
        self.assertEqual(g.ecount(), 2)

    def test_graph_is_directed(self):
        path = self._write("sample.ttl", TTL_SAMPLE)
        g = load_kg(path)
        self.assertTrue(g.is_directed())

    def test_vertex_has_name_attribute(self):
        path = self._write("sample.ttl", TTL_SAMPLE)
        g = load_kg(path)
        names = set(g.vs["name"])
        self.assertIn("http://example.org/Alice", names)
        self.assertIn("http://example.org/Bob", names)

    def test_literal_vertex_flags(self):
        path = self._write("sample.ttl", TTL_SAMPLE)
        g = load_kg(path)
        literals = [v for v in g.vs if v["is_literal"]]
        self.assertEqual(len(literals), 2)

        int_lit = next(v for v in literals if v["literal_datatype"] and "integer" in v["literal_datatype"])
        self.assertEqual(int_lit["literal_value"], "30")

        lang_lit = next(v for v in literals if v["literal_lang"] == "en")
        self.assertEqual(lang_lit["literal_value"], "Bob")

    def test_edge_has_predicate_attribute(self):
        path = self._write("sample.ttl", TTL_SAMPLE)
        g = load_kg(path)
        predicates = set(g.es["predicate"])
        self.assertIn("http://example.org/knows", predicates)

    def test_invalid_content_raises(self):
        path = self._write("sample.rdf", "<rdf/>")
        with self.assertRaises(ValueError):
            load_kg(path)

    def test_format_detected_from_content_not_extension(self):
        # N-Triples content under a .ttl name, and Turtle content under a .nt
        # name, both load correctly: detection ignores the extension.
        nt_path = self._write("misnamed.ttl", NT_SAMPLE)
        ttl_path = self._write("misnamed.nt", TTL_SAMPLE)
        self.assertEqual(load_kg(nt_path).ecount(), 2)
        self.assertEqual(load_kg(ttl_path).ecount(), 3)

    def test_extensionless_file_loads(self):
        path = self._write("59622641", NT_SAMPLE)  # raw dumps often have no suffix
        g = load_kg(path)
        self.assertEqual(g.vcount(), 3)
        self.assertEqual(g.ecount(), 2)

    def test_duplicate_triples_collapse_to_one_edge(self):
        ttl = (
            "@prefix ex: <http://example.org/> .\n"
            "ex:a ex:knows ex:b .\n"
            "ex:a ex:knows ex:b .\n"  # duplicate of line above
            "ex:a ex:knows ex:b .\n"  # another duplicate
        )
        path = self._write("dupes.ttl", ttl)
        g = load_kg(path)
        self.assertEqual(g.vcount(), 2)
        self.assertEqual(g.ecount(), 1)


class TestSaveKG(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write(self, filename, content):
        path = os.path.join(self.tmp, filename)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_roundtrip_ttl(self):
        src = self._write("in.ttl", TTL_SAMPLE)
        out = os.path.join(self.tmp, "out.ttl")
        g = load_kg(src)
        save_kg(g, out)
        g2 = load_kg(out)
        self.assertEqual(g.vcount(), g2.vcount())
        self.assertEqual(g.ecount(), g2.ecount())

    def test_roundtrip_nt(self):
        src = self._write("in.nt", NT_SAMPLE)
        out = os.path.join(self.tmp, "out.nt")
        g = load_kg(src)
        save_kg(g, out, fmt="nt")
        g2 = load_kg(out)
        self.assertEqual(g.vcount(), g2.vcount())
        self.assertEqual(g.ecount(), g2.ecount())

    def test_save_nt_produces_file(self):
        src = self._write("in.ttl", TTL_SAMPLE)
        out = os.path.join(self.tmp, "out.nt")
        g = load_kg(src)
        save_kg(g, out, fmt="nt")
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)

    def test_unsupported_format_raises(self):
        src = self._write("in.ttl", TTL_SAMPLE)
        g = load_kg(src)
        with self.assertRaises(ValueError):
            save_kg(g, os.path.join(self.tmp, "out.xml"), fmt="xml")


if __name__ == "__main__":
    unittest.main()
