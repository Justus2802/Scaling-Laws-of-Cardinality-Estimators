"""Tests for the dataset runner (``kgsynth.dataset``).

Everything here is fast: config parsing and plan enumeration are pure, and the
worker's validation runs before the (expensive) generation step. Generating a real
graph takes tens of minutes, so it is exercised end-to-end by hand, not here.

The properties that matter:

- **Fail early.** A run is tens of minutes per graph. A bad feature name must be
  caught at parse time, not on graph 40.
- **Reproducible.** A unit's seeds depend only on its index — never on worker count
  or completion order.
- **Realizable.** A perturbed signature is checked *before* generating, so an
  impossible target fails fast instead of crashing deep inside a stage.
"""

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kgsynth.dataset import DatasetConfig, build_units
from kgsynth.dataset.worker import InvalidSignature, validate_features
from kgsynth.corpus import DEFAULT_SEARCH_DIRS, load_target_from_corpus

_BASE = "fb237_v4"

_MINIMAL = """
base: fb237_v4
design: joint
num_graphs: 3
seed: 42
out_dir: {out}
features:
  mean_degree: {{dist: lognormal, sigma: 0.15, levels: [0.9, 1.1]}}
  triangle_count: {{dist: lognormal, sigma: 0.2, levels: [0.8, 1.2]}}
"""


def _write(tmp: str, body: str) -> Path:
    path = Path(tmp) / "run.yaml"
    path.write_text(textwrap.dedent(body).format(out=Path(tmp) / "out"))
    return path


class TestConfig(unittest.TestCase):
    def test_parses_minimal(self):
        with TemporaryDirectory() as tmp:
            cfg = DatasetConfig.from_yaml(_write(tmp, _MINIMAL))
            self.assertEqual(cfg.base, _BASE)
            self.assertEqual(cfg.design, "joint")
            self.assertEqual(cfg.num_graphs, 3)
            self.assertEqual(set(cfg.specs), {"mean_degree", "triangle_count"})
            self.assertFalse(cfg.measure)

    def test_rejects_off_surface_feature(self):
        # shortest_path_mean is measured but never read by the generator: perturbing
        # it would silently produce a duplicate of the baseline graph.
        with TemporaryDirectory() as tmp:
            body = _MINIMAL.replace("mean_degree:", "shortest_path_mean:")
            with self.assertRaises(ValueError) as ctx:
                DatasetConfig.from_yaml(_write(tmp, body))
            self.assertIn("not read by the generator", str(ctx.exception))

    def test_rejects_unknown_design(self):
        with TemporaryDirectory() as tmp:
            body = _MINIMAL.replace("design: joint", "design: latin_hypercube")
            with self.assertRaises(ValueError):
                DatasetConfig.from_yaml(_write(tmp, body))

    def test_joint_needs_num_graphs(self):
        with TemporaryDirectory() as tmp:
            body = _MINIMAL.replace("num_graphs: 3", "num_graphs: 0")
            with self.assertRaises(ValueError) as ctx:
                DatasetConfig.from_yaml(_write(tmp, body))
            self.assertIn("num_graphs", str(ctx.exception))

    def test_ofat_needs_levels_on_every_feature(self):
        with TemporaryDirectory() as tmp:
            body = _MINIMAL.replace("design: joint", "design: ofat").replace(
                "triangle_count: {{dist: lognormal, sigma: 0.2, levels: [0.8, 1.2]}}",
                "triangle_count: {{dist: lognormal, sigma: 0.2}}",
            )
            with self.assertRaises(ValueError) as ctx:
                DatasetConfig.from_yaml(_write(tmp, body))
            self.assertIn("levels", str(ctx.exception))

    def test_rejects_unknown_base(self):
        with TemporaryDirectory() as tmp:
            body = _MINIMAL.replace("base: fb237_v4", "base: not_a_graph")
            with self.assertRaises(ValueError) as ctx:
                DatasetConfig.from_yaml(_write(tmp, body))
            self.assertIn("not found", str(ctx.exception))

    def test_rejects_unknown_spec_key(self):
        with TemporaryDirectory() as tmp:
            body = _MINIMAL.replace("sigma: 0.15", "sigma: 0.15, stdev: 3")
            with self.assertRaises(ValueError) as ctx:
                DatasetConfig.from_yaml(_write(tmp, body))
            self.assertIn("stdev", str(ctx.exception))

    def test_rejects_empty_features(self):
        with TemporaryDirectory() as tmp:
            body = "base: fb237_v4\ndesign: joint\nnum_graphs: 1\nout_dir: {out}\n"
            with self.assertRaises(ValueError):
                DatasetConfig.from_yaml(_write(tmp, body))

    def test_example_config_is_valid(self):
        # The shipped example must parse — it is the first thing a reader runs.
        repo = Path(__file__).resolve().parent.parent
        cfg = DatasetConfig.from_yaml(repo / "examples" / "perturb_dataset.yaml")
        self.assertTrue(cfg.specs)


class TestPlan(unittest.TestCase):
    def _cfg(self, tmp, body=_MINIMAL):
        return DatasetConfig.from_yaml(_write(tmp, body))

    def test_joint_yields_num_graphs_units(self):
        with TemporaryDirectory() as tmp:
            units = build_units(self._cfg(tmp))
            self.assertEqual(len(units), 3)
            self.assertEqual([u.label for u in units], ["joint"] * 3)

    def test_ofat_yields_baseline_plus_one_per_level(self):
        with TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, _MINIMAL.replace("design: joint", "design: ofat"))
            units = build_units(cfg)
            # 1 baseline + 2 knobs x 2 levels
            self.assertEqual(len(units), 5)
            self.assertEqual(units[0].label, "baseline")
            self.assertIn("mean_degree×0.9", [u.label for u in units])

    def test_ofat_sweeps_a_coupled_group_once(self):
        # Naming two members of one quantile group is one knob, not two: moving
        # either moves all seven.
        with TemporaryDirectory() as tmp:
            body = """
            base: fb237_v4
            design: ofat
            out_dir: {out}
            features:
              obj_mult_alpha_q25: {{dist: lognormal, levels: [1.1]}}
              obj_mult_alpha_q75: {{dist: lognormal, levels: [1.1]}}
            """
            units = build_units(DatasetConfig.from_yaml(_write(tmp, body)))
            self.assertEqual(len(units), 2)  # baseline + ONE knob x 1 level

    def test_seeds_are_distinct_and_stable(self):
        with TemporaryDirectory() as tmp:
            a = build_units(self._cfg(tmp))
            b = build_units(self._cfg(tmp))
            self.assertEqual([u.perturb_seed for u in a], [u.perturb_seed for u in b])
            seeds = [u.perturb_seed for u in a] + [u.generate_seed for u in a]
            self.assertEqual(len(set(seeds)), len(seeds))

    def test_seed_depends_only_on_index(self):
        # The point of SeedSequence.spawn: unit 2's result must not depend on how
        # many workers ran, or on which units finished first.
        with TemporaryDirectory() as tmp:
            few = build_units(self._cfg(tmp))
            many = build_units(self._cfg(tmp, _MINIMAL.replace("num_graphs: 3", "num_graphs: 9")))
            for i in range(3):
                self.assertEqual(few[i].perturb_seed, many[i].perturb_seed)
                self.assertEqual(few[i].generate_seed, many[i].generate_seed)

    def test_master_seed_changes_everything(self):
        with TemporaryDirectory() as tmp:
            a = build_units(self._cfg(tmp))
            b = build_units(self._cfg(tmp, _MINIMAL.replace("seed: 42", "seed: 43")))
            self.assertNotEqual(a[0].perturb_seed, b[0].perturb_seed)

    def test_units_are_picklable(self):
        # They cross a process boundary; a transform that is not picklable would
        # only surface when the pool starts.
        import pickle

        with TemporaryDirectory() as tmp:
            for unit in build_units(self._cfg(tmp)):
                self.assertEqual(pickle.loads(pickle.dumps(unit)).index, unit.index)


class TestValidateFeatures(unittest.TestCase):
    """A perturbed signature is checked before the (expensive) generation step."""

    @classmethod
    def setUpClass(cls):
        cls.base = load_target_from_corpus(_BASE, DEFAULT_SEARCH_DIRS)[0].as_features()

    def test_baseline_is_valid(self):
        validate_features(dict(self.base))  # must not raise

    def test_rejects_zero_edge_budget(self):
        feats = dict(self.base)
        feats["mean_degree"] = 0.0
        with self.assertRaises(InvalidSignature):
            validate_features(feats)

    def test_rejects_more_cs_than_entities(self):
        feats = dict(self.base)
        feats["num_distinct_cs"] = feats["num_entities"] * 2
        with self.assertRaises(InvalidSignature):
            validate_features(feats)

    def test_rejects_inverted_quantiles(self):
        feats = dict(self.base)
        feats["cs_size_q10"], feats["cs_size_q90"] = 9.0, 1.0
        with self.assertRaises(InvalidSignature) as ctx:
            validate_features(feats)
        self.assertIn("non-decreasing", str(ctx.exception))

    def test_rejects_negative_motif_count(self):
        feats = dict(self.base)
        feats["triangle_count"] = -1.0
        with self.assertRaises(InvalidSignature):
            validate_features(feats)

    def test_rejects_impossible_component_fraction(self):
        feats = dict(self.base)
        feats["largest_component_fraction"] = 1.5
        with self.assertRaises(InvalidSignature):
            validate_features(feats)


if __name__ == "__main__":
    unittest.main()
