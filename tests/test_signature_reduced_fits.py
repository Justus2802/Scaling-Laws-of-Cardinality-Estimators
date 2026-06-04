import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from signature_reduced._fits import (  # noqa: E402
    fit_skewnorm,
    fit_exp_decay_rank,
    fit_truncated_powerlaw,
    fit_zipf,
    fit_cs_size_offset,
    nan_skewnorm,
)


class TestFitSkewNorm(unittest.TestCase):
    def test_recovers_bounds_and_finite_params(self):
        rng = np.random.default_rng(0)
        # Right-skewed sample (shape > 0).
        data = np.abs(rng.standard_normal(500)) + 1.0
        fit = fit_skewnorm(data)
        self.assertTrue(all(math.isfinite(v) for v in fit))
        # Cutoffs default to the observed range.
        self.assertAlmostEqual(fit.lo, float(data.min()))
        self.assertAlmostEqual(fit.hi, float(data.max()))

    def test_explicit_cutoffs_override(self):
        rng = np.random.default_rng(1)
        data = rng.normal(2.0, 0.3, size=200)
        fit = fit_skewnorm(data, lo=1.4, hi=3.0)
        self.assertEqual(fit.lo, 1.4)
        self.assertEqual(fit.hi, 3.0)

    def test_small_sample_is_nan(self):
        fit = fit_skewnorm([1.0, 2.0, 3.0])
        self.assertTrue(all(math.isnan(v) for v in fit))
        self.assertEqual(fit, nan_skewnorm())


class TestFitExpDecayRank(unittest.TestCase):
    def test_recovers_known_rate(self):
        true_rate, true_scale = 0.4, 10.0
        k = np.arange(12)
        values = true_scale * np.exp(-true_rate * k)
        fit = fit_exp_decay_rank(values)
        self.assertAlmostEqual(fit.rate, true_rate, places=6)
        self.assertAlmostEqual(fit.scale, true_scale, places=4)

    def test_unsorted_input_is_sorted_internally(self):
        k = np.arange(12)
        values = 5.0 * np.exp(-0.3 * k)
        shuffled = values.copy()
        np.random.default_rng(2).shuffle(shuffled)
        self.assertAlmostEqual(
            fit_exp_decay_rank(values).rate, fit_exp_decay_rank(shuffled).rate, places=6
        )

    def test_too_few_points_is_nan(self):
        fit = fit_exp_decay_rank([5.0, 2.0])
        self.assertTrue(math.isnan(fit.rate))


class TestFitTruncatedPowerLaw(unittest.TestCase):
    def test_returns_bounds_and_alpha(self):
        rng = np.random.default_rng(3)
        # Discrete heavy-tailed sample within a bounded range.
        data = (rng.pareto(1.5, size=2000) + 1).astype(int)
        data = data[data <= 50]
        fit = fit_truncated_powerlaw(data)
        self.assertEqual(fit.v_min, float(data.min()))
        self.assertEqual(fit.v_max, float(data.max()))
        self.assertTrue(math.isfinite(fit.alpha))
        self.assertGreater(fit.alpha, 1.0)

    def test_degenerate_range_is_nan(self):
        fit = fit_truncated_powerlaw([5] * 20)
        self.assertTrue(math.isnan(fit.alpha))

    def test_small_sample_is_nan(self):
        self.assertTrue(math.isnan(fit_truncated_powerlaw([1, 2, 3]).alpha))


class TestFitZipf(unittest.TestCase):
    def test_finite_exponent_on_heavy_tail(self):
        rng = np.random.default_rng(4)
        counts = (rng.zipf(2.0, size=500)).astype(float)
        fit = fit_zipf(counts)
        self.assertTrue(math.isfinite(fit.exponent))
        self.assertGreater(fit.exponent, 1.0)


class TestFitCsSizeOffset(unittest.TestCase):
    def test_recovers_known_slope(self):
        rng = np.random.default_rng(5)
        cs_size = rng.integers(1, 30, size=400).astype(float)
        true_a = 0.6
        mult = np.exp(true_a * np.log(cs_size) + rng.normal(0, 0.01, size=cs_size.size))
        a = fit_cs_size_offset(cs_size, mult)
        self.assertAlmostEqual(a, true_a, places=1)

    def test_no_variation_is_nan(self):
        a = fit_cs_size_offset([4.0] * 20, [2.0] * 20)
        self.assertTrue(math.isnan(a))

    def test_small_sample_is_nan(self):
        self.assertTrue(math.isnan(fit_cs_size_offset([1.0, 2.0], [1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
