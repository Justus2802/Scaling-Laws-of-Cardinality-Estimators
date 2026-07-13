"""Round-trip tests for ``Signature.as_features()`` / ``.from_features()``.

The contract these pin: a ``Signature`` rebuilt from nothing but the flat 124-key
feature dict reproduces **every value the generator reads**, and therefore
generates the identical graph. That is what lets the perturbation pipeline work on
the public feature dict instead of reaching into private block state.

The strongest assertion here is :meth:`TestGeneratesIdenticalGraph.test_edge_lists_identical`
— schema equality and edge-list equality at a fixed seed. Everything above it is a
faster localisation of the same property.
"""

import math
import unittest

import numpy as np
from kgsynth import Signature
from kgsynth.corpus import DEFAULT_SEARCH_DIRS, load_target_from_corpus
from kgsynth.generator.stage1 import sample_schema
from kgsynth.generator.stage2 import instantiate
from kgsynth.signature import _BLOCK_CLASSES

# The smallest corpus graph (V=4707), so the end-to-end generation check stays fast.
_GRAPH = "fb237_v4"

# aids has only 5 relations, too few for the per-relation alpha-quantile fit to
# converge: its obj_alpha_q is all-NaN. That is a real measurement outcome, and it
# must survive the round trip as NaN rather than being silently repaired to 0.0.
_NAN_FIT_GRAPH = "aids"

# Every block attribute the generator dereferences, as `sig.<block>.<path>`.
# Derived by tracing the reads in generator/stage{1,2,3}.py — if a stage starts
# reading a new attribute, add it here.
_GENERATOR_READS = [
    "a.num_entities", "a.num_relations", "a.mean_degree",
    "b.out_degree_fit.alpha", "b.in_degree_fit.alpha", "b.relation_zipf.exponent",
    "b.obj_alpha_q", "b.subj_alpha_q", "b.a_obj", "b.a_subj",
    "b.obj_mult_max", "b.subj_mult_max",
    "b.out_degree_max", "b.out_degree_p90", "b.in_degree_max", "b.in_degree_p90",
    "b.recip_symmetric_frac", "b.recip_symmetric_value",
    "c.num_classes", "c.class_size_fit.alpha", "c.edge_multiplicity",
    "c.bidirectional_ratio", "c.subj_cooc_exp", "c.obj_cooc_exp",
    "c.type_rel_spectrum_exp",
    "d.num_distinct_cs", "d.inv_num_distinct_cs", "d.cs_size_q", "d.inv_cs_size_q",
    "d.cs_freq_fit.alpha", "d.cs_freq_fit.v_min", "d.cs_freq_fit.v_max",
    "d.inv_cs_freq_fit.alpha", "d.inv_cs_freq_fit.v_min", "d.inv_cs_freq_fit.v_max",
    "e.triangle_count", "e.four_cycle_count", "e.five_cycle_count",
    "e.six_cycle_count", "e.diamond_count", "e.k4_count", "e.tailed_triangle_count",
    "f.num_components", "f.largest_component_fraction",
    "f.clustering_coefficient", "f.degree_assortativity",
]


def _resolve(sig: Signature, path: str):
    """Follow a dotted attribute path (e.g. ``b.out_degree_fit.alpha``) on *sig*."""
    obj = sig
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _load(name: str = _GRAPH) -> Signature:
    return load_target_from_corpus(name, DEFAULT_SEARCH_DIRS)[0]


def _assert_schema_field_equal(a, b) -> None:
    """Assert two Schema field values are equal, across the field types Schema holds.

    Schema mixes ``None``, ints, str lists (relation/type URIs), float scalars and
    numpy arrays, so a single comparison does not cover it.
    """
    if a is None or b is None:
        assert a is None and b is None, f"{a!r} != {b!r}"
        return
    if isinstance(a, str) or (isinstance(a, list) and a and isinstance(a[0], str)):
        assert a == b, f"{a!r} != {b!r}"
        return
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True
    )


class TestFeatureDict(unittest.TestCase):
    """as_features() shape and key set."""

    @classmethod
    def setUpClass(cls):
        cls.sig = _load()

    def test_has_all_127_features(self):
        feats = self.sig.as_features()
        expected = sum(len(c.feature_names()) for c in _BLOCK_CLASSES.values())
        self.assertEqual(len(feats), expected)
        self.assertEqual(len(feats), 127)

    def test_keys_match_block_feature_names(self):
        feats = self.sig.as_features()
        for cls in _BLOCK_CLASSES.values():
            for name in cls.feature_names():
                self.assertIn(name, feats)

    def test_missing_key_raises(self):
        feats = self.sig.as_features()
        del feats["mean_degree"]
        with self.assertRaises(KeyError):
            Signature.from_features(feats)


class TestRoundTrip(unittest.TestCase):
    """Every generator-consumed value survives as_features() -> from_features()."""

    @classmethod
    def setUpClass(cls):
        cls.sig = _load()
        cls.rebuilt = Signature.from_features(cls.sig.as_features())

    def test_generator_reads_reconstructed_exactly(self):
        for path in _GENERATOR_READS:
            with self.subTest(attr=path):
                original = np.atleast_1d(np.asarray(_resolve(self.sig, path), dtype=float))
                rebuilt = np.atleast_1d(np.asarray(_resolve(self.rebuilt, path), dtype=float))
                np.testing.assert_allclose(rebuilt, original, equal_nan=True)

    def test_features_are_idempotent(self):
        # A second round trip must be a fixed point: from_features(as_features(x))
        # changes nothing that as_features() can see.
        once = self.sig.as_features()
        twice = Signature.from_features(once).as_features()
        for name, value in once.items():
            with self.subTest(feature=name):
                if isinstance(value, float) and math.isnan(value):
                    self.assertTrue(math.isnan(twice[name]))
                else:
                    self.assertAlmostEqual(twice[name], value, places=9)

    def test_counts_are_ints(self):
        # A float count propagates into range() and array shapes downstream.
        self.assertIsInstance(self.rebuilt.a.num_entities, int)
        self.assertIsInstance(self.rebuilt.a.num_relations, int)
        self.assertIsInstance(self.rebuilt.c.num_classes, int)
        self.assertIsInstance(self.rebuilt.d.num_distinct_cs, int)
        self.assertIsInstance(self.rebuilt.e.triangle_count, int)
        self.assertIsInstance(self.rebuilt.f.num_components, int)

    def test_nan_fit_survives_as_nan(self):
        # An unconverged fit is a real measurement outcome, not a missing value: it
        # must come back as NaN, not 0.0, so the generator's NaN fallbacks fire.
        sig = _load(_NAN_FIT_GRAPH)
        rebuilt = Signature.from_features(sig.as_features())
        original = np.asarray(sig.b.obj_alpha_q, dtype=float)
        self.assertTrue(np.isnan(original).all(), "fixture no longer has an all-NaN fit")
        self.assertTrue(np.isnan(np.asarray(rebuilt.b.obj_alpha_q, dtype=float)).all())


class TestGeneratesIdenticalGraph(unittest.TestCase):
    """The load-bearing claim: a rebuilt signature generates the identical graph."""

    @classmethod
    def setUpClass(cls):
        cls.sig = _load()
        cls.rebuilt = Signature.from_features(cls.sig.as_features())

    def _schema(self, sig):
        return sample_schema(sig.a, sig.c, d=sig.d, b=sig.b, f=sig.f, seed=7)

    def test_schemas_identical(self):
        original, rebuilt = self._schema(self.sig), self._schema(self.rebuilt)
        for field in original.__dataclass_fields__:
            with self.subTest(field=field):
                _assert_schema_field_equal(getattr(original, field), getattr(rebuilt, field))

    def test_edge_lists_identical(self):
        g1 = instantiate(self._schema(self.sig), seed=8)
        g2 = instantiate(self._schema(self.rebuilt), seed=8)
        self.assertEqual(g1.vcount(), g2.vcount())
        self.assertEqual(g1.ecount(), g2.ecount())
        self.assertEqual(
            sorted(zip(g1.get_edgelist(), g1.es["predicate"])),
            sorted(zip(g2.get_edgelist(), g2.es["predicate"])),
        )


if __name__ == "__main__":
    unittest.main()
