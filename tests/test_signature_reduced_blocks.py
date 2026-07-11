import os
import tempfile
import unittest

import igraph
import numpy as np
from kgsynth.kg_io import load_kg  # noqa: E402
from kgsynth.signature import (  # noqa: E402
    BlockA, BlockB, BlockC, BlockD, BlockE, BlockF,
    ReducedGraphSignature,
)

_BLOCKS = [BlockA, BlockB, BlockC, BlockD, BlockE, BlockF]
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def _rich_ttl() -> str:
    """A small but non-degenerate KG: types, multiplicities and two-step paths."""
    ex = "http://example.org/"
    lines = [f"@prefix ex: <{ex}> .",
             "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."]
    types = ["Person", "City", "Company"]
    rels = ["knows", "livesIn", "worksAt", "bornIn", "visited"]
    rng = np.random.default_rng(0)
    for i in range(40):
        s = f"ex:e{i}"
        lines.append(f"{s} rdf:type ex:{types[i % 3]} .")
        n_rels = 1 + (i % len(rels))
        for r in rels[:n_rels]:
            # multiplicity varies: some relations fan out to several objects
            for k in range(1 + int(rng.integers(0, 3))):
                o = f"ex:e{(i + k + 1) % 40}"
                lines.append(f"{s} ex:{r} {o} .")
    return "\n".join(lines) + "\n"


def _load_rich() -> igraph.Graph:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "g.ttl")
    with open(path, "w") as f:
        f.write(_rich_ttl())
    return load_kg(path)


class TestReducedBlocksContract(unittest.TestCase):
    """Length / naming / serialization invariants every block must satisfy."""

    @classmethod
    def setUpClass(cls):
        cls.g = _load_rich()
        cls.computed = {blk: blk().calculate(cls.g) for blk in _BLOCKS}

    def test_vector_length_matches_names_and_na(self):
        for blk in _BLOCKS:
            with self.subTest(block=blk.__name__):
                b = self.computed[blk]
                n = len(b.as_vector())
                self.assertEqual(len(blk.feature_names()), n)
                self.assertEqual(len(blk.get_na_vec()), n)

    def test_as_dict_keys_match_feature_names(self):
        for blk in _BLOCKS:
            with self.subTest(block=blk.__name__):
                b = self.computed[blk]
                self.assertEqual(list(b.as_dict().keys()), blk.feature_names())

    def test_as_dict_values_match_vector(self):
        for blk in _BLOCKS:
            with self.subTest(block=blk.__name__):
                b = self.computed[blk]
                np.testing.assert_array_equal(list(b.as_dict().values()), b.as_vector())

    def test_serialization_roundtrip(self):
        # The NamedTuple fits restore as plain tuples; properties must re-wrap so
        # attribute access in as_vector still works after a round-trip.
        for blk in _BLOCKS:
            with self.subTest(block=blk.__name__):
                b = self.computed[blk]
                restored = blk.from_serializable(b.to_serializable())
                np.testing.assert_array_equal(b.as_vector(), restored.as_vector())

    def test_text_visualize_runs(self):
        for blk in _BLOCKS:
            with self.subTest(block=blk.__name__):
                tmp = tempfile.mkdtemp()
                out = os.path.join(tmp, "summary.txt")
                self.computed[blk].visualize(mode="text", path=out)
                self.assertTrue(os.path.getsize(out) > 0)


class TestReducedBlockValues(unittest.TestCase):
    """A few content checks beyond the structural invariants."""

    @classmethod
    def setUpClass(cls):
        cls.g = _load_rich()

    def test_block_a_basics(self):
        a = BlockA().calculate(self.g)
        # 40 entities + 3 rdf:type class nodes (all non-literal vertices count).
        self.assertEqual(a.num_entities, 43)
        self.assertGreater(a.num_relations, 0)
        self.assertGreater(a.mean_degree, 0.0)

    def test_block_c_has_classes(self):
        c = BlockC().calculate(self.g)
        self.assertEqual(c.num_classes, 3)

    def test_block_d_distinct_cs_positive(self):
        d = BlockD().calculate(self.g)
        self.assertGreater(d.num_distinct_cs, 0)

    def test_block_f_lcc_fraction_in_range(self):
        f = BlockF().calculate(self.g)
        self.assertGreaterEqual(f.largest_component_fraction, 0.0)
        self.assertLessEqual(f.largest_component_fraction, 1.0)


class TestReducedSignatureAggregate(unittest.TestCase):
    def test_empty_blocks_are_nan_filled(self):
        sig = ReducedGraphSignature()  # nothing computed
        vec = sig.as_vector()
        self.assertTrue(all(np.isnan(v) for v in vec))
        # length = sum of per-block vector lengths
        expected = sum(len(blk.get_na_vec()) for blk in _BLOCKS)
        self.assertEqual(len(vec), expected)
        self.assertEqual(len(sig.as_dict()), expected)


if __name__ == "__main__":
    unittest.main()
