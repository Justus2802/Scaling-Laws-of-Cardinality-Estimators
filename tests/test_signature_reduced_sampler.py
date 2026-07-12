import json
import math
import unittest
from pathlib import Path
from kgsynth.signature import BlockE, ReducedGraphSignature  # noqa: E402
from kgsynth.signature_sampler import (  # noqa: E402
    FEATURE_ORDER,
    SignatureSampler,
    UniformRangeSampler,
    _INTEGER_FEATURES,
    _TYPE_PARAM_FEATURES,
)

_REPO = Path(__file__).resolve().parent.parent
_CORPUS_DIR = _REPO / "data" / "graphs"

# Bounded features deliberately set near their domain edge so the ±10 % widening
# overshoots and the post-sampling clamp is exercised.
_BOUNDED = {
    "clustering_coefficient": (0.9, 0.95, 1.0),
    "subj_cooc_density": (0.9, 0.95, 1.0),
    "obj_cooc_density": (0.9, 0.95, 1.0),
    "largest_component_fraction": (0.9, 0.95, 1.0),
    "degree_assortativity": (-1.0, -0.9, -0.8),
}
# Truly-constant features (range 0) must be reproduced exactly.
_CONST = {"obj_mult_alpha_q00": 1.4, "obj_mult_alpha_q100": 3.0}


def _synthetic_corpus() -> dict[str, dict[str, float]]:
    """Three synthetic graphs spanning a non-zero range on every feature."""
    graphs = {"g1": {}, "g2": {}, "g3": {}}
    for i, feat in enumerate(FEATURE_ORDER):
        base = float(i + 1)
        graphs["g1"][feat], graphs["g2"][feat], graphs["g3"][feat] = base, 2 * base, 3 * base
    for feat, vals in _BOUNDED.items():
        for g, v in zip(graphs.values(), vals):
            g[feat] = v
    for feat, v in _CONST.items():
        for g in graphs.values():
            g[feat] = v
    return graphs


def _same(d1: dict, d2: dict) -> bool:
    """Dict equality treating NaN == NaN (so reproducibility can be asserted)."""
    if d1.keys() != d2.keys():
        return False
    return all(
        (math.isnan(d1[k]) and math.isnan(d2[k])) or d1[k] == d2[k] for k in d1
    )


class TestFeatureOrder(unittest.TestCase):
    def test_keys_match_signature_schema_minus_block_e(self):
        # The sampler deliberately excludes Block E (motifs / G5 are out of scope),
        # so FEATURE_ORDER must equal the measured schema with Block E's feature
        # names removed, in the same order.
        self.assertEqual(len(FEATURE_ORDER), 97)
        self.assertEqual(len(set(FEATURE_ORDER)), 97)  # no duplicates
        block_e_names = set(BlockE.feature_names())
        schema_minus_e = [
            k for k in ReducedGraphSignature().as_dict() if k not in block_e_names
        ]
        self.assertEqual(schema_minus_e, FEATURE_ORDER)


class TestUniformRangeSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = UniformRangeSampler(_synthetic_corpus())

    def test_sample_returns_exact_69_keys(self):
        out = self.sampler.sample(seed=0)
        self.assertEqual(list(out.keys()), FEATURE_ORDER)

    def test_reproducible_and_seed_sensitive(self):
        a = self.sampler.sample(seed=7)
        b = self.sampler.sample(seed=7)
        c = self.sampler.sample(seed=8)
        self.assertTrue(_same(a, b))
        self.assertNotAlmostEqual(a["mean_degree"], c["mean_degree"])  # free feature

    def test_values_within_widened_range(self):
        out = self.sampler.sample(seed=3)
        for feat in FEATURE_ORDER:
            if feat == "num_classes" or feat in _TYPE_PARAM_FEATURES:
                continue
            finite = self.sampler._finite[feat]
            if finite.size < 2:
                continue
            lo, hi = float(finite.min()), float(finite.max())
            pad = UniformRangeSampler.WIDEN * (hi - lo)
            # Integer rounding can nudge up to 0.5 outside the raw draw interval.
            tol = 0.5 if feat in _INTEGER_FEATURES else 1e-9
            self.assertGreaterEqual(out[feat], lo - pad - tol, feat)
            self.assertLessEqual(out[feat], hi + pad + tol, feat)

    def test_integer_features_are_whole(self):
        out = self.sampler.sample(seed=1)
        # _INTEGER_FEATURES spans the whole 124-feature signature; the sampler
        # covers only A/B/C/D/F (Block E motifs are out of its scope), so intersect.
        for feat in set(_INTEGER_FEATURES) & set(FEATURE_ORDER):
            if math.isnan(out[feat]):
                continue
            self.assertEqual(out[feat], round(out[feat]), feat)

    def test_type_block_is_untyped(self):
        out = self.sampler.sample(seed=2)
        self.assertEqual(out["num_classes"], 0.0)
        for feat in _TYPE_PARAM_FEATURES:
            self.assertTrue(math.isnan(out[feat]), feat)

    def test_constant_feature_reproduced_exactly(self):
        out = self.sampler.sample(seed=5)
        for feat, v in _CONST.items():
            self.assertEqual(out[feat], v, feat)

    def test_domain_clamps_respected(self):
        # Sample several seeds; bounded features must never leave their domain.
        for seed in range(20):
            out = self.sampler.sample(seed=seed)
            for feat in ("clustering_coefficient", "subj_cooc_density",
                         "obj_cooc_density", "largest_component_fraction"):
                self.assertGreaterEqual(out[feat], 0.0, feat)
                self.assertLessEqual(out[feat], 1.0, feat)
            self.assertGreaterEqual(out["degree_assortativity"], -1.0)
            self.assertLessEqual(out["degree_assortativity"], 1.0)

    def test_insufficient_support_yields_nan(self):
        # A non-type feature finite in only one graph → NaN (need ≥2 to form a range).
        corpus = _synthetic_corpus()
        corpus["g2"]["mean_degree"] = float("nan")
        corpus["g3"]["mean_degree"] = float("nan")
        out = UniformRangeSampler(corpus).sample(seed=0)
        self.assertTrue(math.isnan(out["mean_degree"]))

    def test_json_output_contract(self):
        out = self.sampler.sample(seed=0)
        wrapped = self.sampler.to_json(out)
        self.assertEqual(set(wrapped), {"source", "features"})
        self.assertEqual(wrapped["source"], "sampled:UniformRangeSampler")
        # Round-trips through JSON (NaN tokens included) with keys preserved.
        restored = json.loads(json.dumps(wrapped))
        self.assertEqual(list(restored["features"].keys()), FEATURE_ORDER)


class TestAbstractBase(unittest.TestCase):
    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            SignatureSampler(_synthetic_corpus())  # _sample_one is abstract

    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            UniformRangeSampler({})


@unittest.skipUnless(
    sorted(_CORPUS_DIR.glob("*/signature/signature.json")),
    "no measured corpus under data/graphs/",
)
class TestLoadRealCorpus(unittest.TestCase):
    def test_load_corpus_and_sample(self):
        sampler = UniformRangeSampler.load_corpus(_CORPUS_DIR)
        self.assertGreaterEqual(len(sampler.corpus), 1)
        out = sampler.sample(seed=0)
        self.assertEqual(list(out.keys()), FEATURE_ORDER)
        self.assertEqual(out["num_classes"], 0.0)  # untyped output


if __name__ == "__main__":
    unittest.main()
