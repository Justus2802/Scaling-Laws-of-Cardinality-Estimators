"""Tests for the signature transforms (``kgsynth.transform``).

Three properties carry the weight here:

- **Determinism** — same seed, same output; a dataset must be reproducible.
- **Validity** — a perturbed signature must still be a *legal* signature: quantile
  functions non-decreasing, values inside their domains, NaN fits left as NaN.
- **Honesty** — when a domain clamps a perturbation, the transform says so. A
  silently-clamped knob reads as "no effect" in an OFAT sweep when in fact it
  never moved.
"""

import unittest

import numpy as np
from kgsynth import Signature
from kgsynth.corpus import DEFAULT_SEARCH_DIRS, load_target_from_corpus
from kgsynth.signature import _BLOCK_CLASSES
from kgsynth.signature._fits import QUANTILE_SUFFIXES
from kgsynth.transform import (
    CONSTANT,
    COUPLED,
    INERT,
    SURFACE,
    FeatureSpec,
    Identity,
    Perturb,
    PerturbOne,
    group_of,
    validate,
)

_GRAPH = "swdf"
_ALPHA_Q = tuple(f"obj_mult_alpha_{s}" for s in QUANTILE_SUFFIXES)


def _base(name: str = _GRAPH) -> dict[str, float]:
    return load_target_from_corpus(name, DEFAULT_SEARCH_DIRS)[0].as_features()


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestSurface(unittest.TestCase):
    """The surface is the set of features the generator actually reads."""

    def test_size(self):
        # 74 of the 124 signature features reach the generator. If a stage starts
        # (or stops) reading one, this fails rather than silently changing the sweep.
        self.assertEqual(len(SURFACE), 74)

    def test_all_names_are_real_features(self):
        known = {n for cls in _BLOCK_CLASSES.values() for n in cls.feature_names()}
        self.assertEqual(len(known), 124)
        self.assertTrue(SURFACE <= known, sorted(SURFACE - known))

    def test_inert_and_constant_are_on_the_surface(self):
        # Both are *read* by the generator — that is what makes them worth warning
        # about rather than rejecting.
        self.assertTrue(INERT <= SURFACE)
        self.assertTrue(CONSTANT <= SURFACE)

    def test_coupled_groups_are_on_the_surface_or_wholly_off_it(self):
        for group in COUPLED:
            on = [n for n in group if n in SURFACE]
            self.assertIn(len(on), (0, len(group)), f"{group} is half on the surface")

    def test_group_of(self):
        self.assertEqual(group_of("obj_mult_alpha_q50"), _ALPHA_Q)
        self.assertEqual(group_of("mean_degree"), ("mean_degree",))


class TestValidate(unittest.TestCase):
    def test_rejects_unknown_feature(self):
        with self.assertRaises(ValueError) as ctx:
            validate(["not_a_feature"])
        self.assertIn("not_a_feature", str(ctx.exception))

    def test_rejects_off_surface_feature(self):
        # Measured, but never read by the generator: perturbing it cannot change
        # the graph, so accepting it would silently produce a duplicate.
        with self.assertRaises(ValueError) as ctx:
            validate(["shortest_path_mean"])
        self.assertIn("not read by the generator", str(ctx.exception))

    def test_warns_on_inert_and_constant(self):
        self.assertTrue(validate(["subj_cooc_scale"]))
        self.assertTrue(validate(["obj_mult_alpha_q00"]))
        self.assertTrue(validate(["num_entities"]))

    def test_silent_on_a_clean_feature(self):
        self.assertEqual(validate(["mean_degree", "triangle_count"]), [])


class TestPerturb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _base()
        cls.specs = {
            "mean_degree": FeatureSpec("lognormal", sigma=0.15),
            "degree_assortativity": FeatureSpec("normal", sigma=0.05),
            "triangle_count": FeatureSpec("lognormal", sigma=0.2),
        }

    def test_deterministic(self):
        a, _ = Perturb(self.specs).apply(self.base, _rng(7))
        b, _ = Perturb(self.specs).apply(self.base, _rng(7))
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        a, _ = Perturb(self.specs).apply(self.base, _rng(1))
        b, _ = Perturb(self.specs).apply(self.base, _rng(2))
        self.assertNotEqual(a["mean_degree"], b["mean_degree"])

    def test_does_not_mutate_input(self):
        before = dict(self.base)
        Perturb(self.specs).apply(self.base, _rng(0))
        self.assertEqual(self.base, before)

    def test_untouched_features_are_unchanged(self):
        out, _ = Perturb(self.specs).apply(self.base, _rng(0))
        for name in ("num_entities", "edge_multiplicity", "clustering_coefficient"):
            self.assertEqual(out[name], self.base[name], name)

    def test_result_rebuilds_into_a_signature(self):
        out, _ = Perturb(self.specs).apply(self.base, _rng(0))
        sig = Signature.from_features(out)
        self.assertAlmostEqual(sig.a.mean_degree, out["mean_degree"])

    def test_motif_counts_stay_integral_after_rebuild(self):
        out, _ = Perturb(self.specs).apply(self.base, _rng(3))
        self.assertIsInstance(Signature.from_features(out).e.triangle_count, int)

    def test_assortativity_stays_in_domain(self):
        # A wide additive jitter must not push a correlation outside [-1, 1].
        specs = {"degree_assortativity": FeatureSpec("normal", sigma=5.0)}
        for seed in range(20):
            out, _ = Perturb(specs).apply(self.base, _rng(seed))
            self.assertGreaterEqual(out["degree_assortativity"], -1.0)
            self.assertLessEqual(out["degree_assortativity"], 1.0)

    def test_edge_multiplicity_floors_at_one(self):
        specs = {"edge_multiplicity": FeatureSpec("lognormal", sigma=2.0)}
        for seed in range(20):
            out, _ = Perturb(specs).apply(self.base, _rng(seed))
            self.assertGreaterEqual(out["edge_multiplicity"], 1.0)


class TestCoupledGroups(unittest.TestCase):
    """Quantile functions must survive perturbation still invertible."""

    @classmethod
    def setUpClass(cls):
        cls.base = _base()

    def _q(self, feats):
        return [feats[n] for n in _ALPHA_Q]

    def test_whole_group_moves_together(self):
        # Naming one member perturbs all seven: they are one knob, not seven.
        out, _ = Perturb({"obj_mult_alpha_q50": FeatureSpec(sigma=0.1)}).apply(
            self.base, _rng(0)
        )
        moved = [n for n in _ALPHA_Q if out[n] != self.base[n]]
        self.assertGreater(len(moved), 1)

    def test_stays_non_decreasing(self):
        spec = FeatureSpec("lognormal", sigma=0.3)
        for seed in range(25):
            out, _ = Perturb({"obj_mult_alpha_q50": spec}).apply(self.base, _rng(seed))
            q = self._q(out)
            self.assertEqual(q, sorted(q), f"seed={seed}: quantiles inverted")

    def test_cs_size_group_stays_non_decreasing(self):
        spec = FeatureSpec("lognormal", sigma=0.3)
        names = [f"cs_size_{s}" for s in QUANTILE_SUFFIXES]
        for seed in range(15):
            out, _ = Perturb({"cs_size_q50": spec}).apply(self.base, _rng(seed))
            q = [out[n] for n in names]
            self.assertEqual(q, sorted(q), f"seed={seed}")

    def test_nan_group_is_left_alone(self):
        # aids's per-relation alpha fit did not converge (all-NaN). That is a real
        # measurement outcome; perturbing it into a number would invent data.
        base = _base("aids")
        self.assertTrue(all(np.isnan(base[n]) for n in _ALPHA_Q))
        out, _ = Perturb({"obj_mult_alpha_q50": FeatureSpec(sigma=0.2)}).apply(base, _rng(0))
        self.assertTrue(all(np.isnan(out[n]) for n in _ALPHA_Q))


class TestClampReport(unittest.TestCase):
    """A clamped perturbation is a no-op wearing a costume — it must be reported."""

    @classmethod
    def setUpClass(cls):
        cls.base = _base()

    def test_upward_alpha_perturbation_is_flagged_saturated(self):
        # swdf's obj_alpha_q sits at the [1.4, 3.0] window's ceiling for its top
        # three levels, so scaling the group up is mostly absorbed by the clamp.
        # Without this report, an OFAT sweep would read that as "no effect".
        self.assertEqual(self.base["obj_mult_alpha_q100"], 3.0)
        out, report = PerturbOne("obj_mult_alpha_q50", 1.2, FeatureSpec("lognormal")).apply(
            self.base, _rng(0)
        )
        self.assertTrue(report.saturated(), "upward alpha jitter should saturate")
        self.assertGreater(report.absorbed[_ALPHA_Q], 0.5)
        self.assertEqual(out["obj_mult_alpha_q100"], 3.0)  # clamped, not moved

    def test_downward_alpha_perturbation_is_not_saturated(self):
        _, report = PerturbOne("obj_mult_alpha_q50", 0.8, FeatureSpec("lognormal")).apply(
            self.base, _rng(0)
        )
        self.assertFalse(report.saturated())

    def test_clean_perturbation_reports_nothing(self):
        _, report = PerturbOne("mean_degree", 1.1, FeatureSpec("lognormal")).apply(
            self.base, _rng(0)
        )
        self.assertEqual(report.clamped, {})
        self.assertEqual(report.saturated(), [])

    def test_report_is_json_safe(self):
        _, report = PerturbOne("obj_mult_alpha_q50", 1.2, FeatureSpec("lognormal")).apply(
            self.base, _rng(0)
        )
        import json

        json.dumps(report.as_json())  # must not raise


class TestInertFeatures(unittest.TestCase):
    """The three exp-decay ``scale`` features are read but cannot change the output.

    ``_adapters._reconstruct_singular_values`` returns ``scale · exp(−rate · r)``, and
    both consumers normalise it immediately (``svs / svs.sum()`` in stage1), so a
    constant factor cancels exactly. Their only surviving role is the ``isnan(scale)``
    presence check. This test pins that: if someone removes the normalisation, the
    features stop being inert and :data:`INERT` becomes a lie.
    """

    def test_scaling_them_does_not_change_the_schema(self):
        from kgsynth.generator.stage1 import sample_schema

        base = _base("fb237_v4")

        def schema(feats):
            sig = Signature.from_features(feats)
            return sample_schema(sig.a, sig.c, d=sig.d, b=sig.b, f=sig.f, seed=3)

        original = schema(base)
        for feature in sorted(INERT):
            with self.subTest(feature=feature):
                bumped = dict(base)
                bumped[feature] = base[feature] * 100.0
                rebuilt = schema(bumped)
                for field in original.__dataclass_fields__:
                    a, b = getattr(original, field), getattr(rebuilt, field)
                    if a is None or isinstance(a, (str, int)):
                        continue
                    if isinstance(a, list) and a and isinstance(a[0], str):
                        continue
                    np.testing.assert_allclose(
                        np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                        equal_nan=True,
                        err_msg=f"{feature} x100 changed Schema.{field} — it is not inert",
                    )


class TestPerturbOne(unittest.TestCase):
    """OFAT: exactly one knob moves, so any downstream difference is attributable."""

    @classmethod
    def setUpClass(cls):
        cls.base = _base()

    def test_only_the_named_group_moves(self):
        out, _ = PerturbOne("mean_degree", 1.2, FeatureSpec("lognormal")).apply(
            self.base, _rng(0)
        )
        changed = [n for n in self.base if out[n] != self.base[n]
                   and not (np.isnan(self.base[n]) and np.isnan(out[n]))]
        self.assertEqual(changed, ["mean_degree"])

    def test_level_is_a_multiplier_for_lognormal(self):
        out, _ = PerturbOne("mean_degree", 1.5, FeatureSpec("lognormal")).apply(
            self.base, _rng(0)
        )
        self.assertAlmostEqual(out["mean_degree"], self.base["mean_degree"] * 1.5)

    def test_level_is_an_offset_for_normal(self):
        out, _ = PerturbOne("degree_assortativity", 0.1, FeatureSpec("normal")).apply(
            self.base, _rng(0)
        )
        self.assertAlmostEqual(
            out["degree_assortativity"], self.base["degree_assortativity"] + 0.1
        )

    def test_is_deterministic_regardless_of_rng(self):
        a, _ = PerturbOne("mean_degree", 1.2, FeatureSpec()).apply(self.base, _rng(1))
        b, _ = PerturbOne("mean_degree", 1.2, FeatureSpec()).apply(self.base, _rng(999))
        self.assertEqual(a, b)


class TestIdentity(unittest.TestCase):
    def test_changes_nothing(self):
        base = _base()
        out, report = Identity().apply(base, _rng(0))
        self.assertEqual(out, base)
        self.assertEqual(report.clamped, {})

    def test_returns_a_copy(self):
        base = _base()
        out, _ = Identity().apply(base, _rng(0))
        out["mean_degree"] = 999.0
        self.assertNotEqual(base["mean_degree"], 999.0)


class TestFeatureSpec(unittest.TestCase):
    def test_rejects_unknown_dist(self):
        with self.assertRaises(ValueError):
            FeatureSpec("gaussian")

    def test_uniform_needs_bounds(self):
        with self.assertRaises(ValueError):
            FeatureSpec("uniform", lo=1.0)

    def test_needs_positive_sigma(self):
        with self.assertRaises(ValueError):
            FeatureSpec("lognormal", sigma=0.0)

    def test_multiplicative_flag(self):
        self.assertTrue(FeatureSpec("lognormal").multiplicative)
        self.assertTrue(FeatureSpec("loguniform", lo=0.5, hi=2.0).multiplicative)
        self.assertFalse(FeatureSpec("normal").multiplicative)
        self.assertFalse(FeatureSpec("uniform", lo=-1.0, hi=1.0).multiplicative)


if __name__ == "__main__":
    unittest.main()
