import math
import unittest

import numpy as np
import scipy.stats
from kgsynth.signature import _distance, fit_quantiles  # noqa: E402
from kgsynth.signature._fits import ZipfFit, ExpDecayFit, TruncPowerLawFit  # noqa: E402
from kgsynth.signature._utils import PowerLawStats  # noqa: E402
from kgsynth.generator._adapters import sample_quantiles_trunc  # noqa: E402


class TestSampleQuantilesTrunc(unittest.TestCase):
    def test_recovers_known_distribution(self):
        rng = np.random.default_rng(0)
        data = rng.normal(5.0, 2.0, size=5000)
        fit = fit_quantiles(data)
        draws = sample_quantiles_trunc(fit, 20000, np.random.default_rng(1))
        # Inverse-transform sampling reproduces the source mean/median closely
        # (the coarse 7-knot grid linearly interpolates, biasing the mean a touch).
        self.assertAlmostEqual(float(np.mean(draws)), float(np.mean(data)), delta=0.2)
        self.assertAlmostEqual(float(np.median(draws)), float(np.median(data)), delta=0.1)
        # Truncated to the stored [q0, q1] range.
        self.assertGreaterEqual(draws.min(), fit.q0)
        self.assertLessEqual(draws.max(), fit.q100)

    def test_nan_fit_returns_none(self):
        self.assertIsNone(sample_quantiles_trunc((float("nan"),) * 7, 5, np.random.default_rng(0)))


class TestWasserstein1(unittest.TestCase):
    def test_zero_on_identical_quantile_fit(self):
        fit = fit_quantiles(np.arange(1, 101.0))
        self.assertEqual(_distance.wasserstein1(fit, fit, _distance.QUANTILE), 0.0)

    def test_pure_shift_equals_offset(self):
        a = fit_quantiles(np.arange(1, 101.0))
        b = fit_quantiles(np.arange(1, 101.0) + 5.0)
        # W1 between a distribution and its rigid +5 shift is exactly 5.
        self.assertAlmostEqual(_distance.wasserstein1(a, b, _distance.QUANTILE), 5.0, places=3)

    def test_matches_scipy_on_reconstructed_samples(self):
        a = fit_quantiles(np.arange(1, 101.0))
        b = fit_quantiles(np.arange(1, 101.0) * 2.0)
        u = np.random.default_rng(_distance._SEED).random(_distance._N_SAMPLE)
        sa = np.interp(u, _distance.QUANTILE_LEVELS, np.asarray(a, dtype=float))
        sb = np.interp(u, _distance.QUANTILE_LEVELS, np.asarray(b, dtype=float))
        self.assertAlmostEqual(
            _distance.wasserstein1(a, b, _distance.QUANTILE),
            float(scipy.stats.wasserstein_distance(sa, sb)),
            places=6,
        )

    def test_zero_on_identical_powerlaw_and_zipf_and_expdecay(self):
        pl = PowerLawStats(2.5, 1.0, *([float("nan")] * 4))
        zf = ZipfFit(2.5, 1.0)
        ed = ExpDecayFit(0.4, 10.0)
        tp = TruncPowerLawFit(2.0, 1.0, 50.0)
        self.assertEqual(_distance.wasserstein1(pl, pl, _distance.POWERLAW), 0.0)
        self.assertEqual(_distance.wasserstein1(zf, zf, _distance.ZIPF), 0.0)
        self.assertEqual(_distance.wasserstein1(ed, ed, _distance.EXP_DECAY), 0.0)
        self.assertEqual(_distance.wasserstein1(tp, tp, _distance.TRUNC_POWERLAW), 0.0)

    def test_nan_fit_gives_nan(self):
        a = fit_quantiles(np.arange(1, 101.0))
        nan_fit = fit_quantiles([1.0])  # too few samples → all NaN
        self.assertTrue(math.isnan(_distance.wasserstein1(a, nan_fit, _distance.QUANTILE)))


if __name__ == "__main__":
    unittest.main()
