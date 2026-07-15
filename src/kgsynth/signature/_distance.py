"""Wasserstein-1 distance between two stored distribution fits.

The roundtrip compares an original signature against the signature re-measured
from a synthetic graph. Comparing fitted *parameters* directly is misleading (an
unstable shape parameter explodes the relative error even when the distributions
agree), so this module compares the *distributions* the fits describe.

Each supported fit ``kind`` is turned into a representative sample (quantile fits
by inverse-transform of their stored quantile function; the parametric families
by their closed-form inverse CDF / reconstructed spectrum), and the distance is
the library Wasserstein-1 (`scipy.stats.wasserstein_distance`) between the two
samples. W1 is in the variable's own units and equals the L1 area between the
quantile functions, so it is a faithful, stable distribution distance.
"""

import numpy as np
import scipy.stats

from ._fits import QUANTILE_LEVELS

# Sample size used to reconstruct each distribution before measuring W1, and a
# fixed seed so the distance is deterministic for a given pair of fits.
_N_SAMPLE = 4000
_SEED = 0
# Number of ranks reconstructed for an exp-decay spectrum (it has no sample size
# of its own); matches the handful of singular values the fit summarises.
_EXP_DECAY_RANKS = 10

# Supported fit kinds (the second element of each ``distribution_fits`` entry).
QUANTILE = "quantile"
POWERLAW = "powerlaw"        # PowerLawStats (alpha, xmin, …)
TRUNC_POWERLAW = "trunc_powerlaw"  # TruncPowerLawFit (alpha, v_min, v_max)
EXP_DECAY = "exp_decay"      # ExpDecayFit (rate, scale)


def _powerlaw_sample(alpha: float, x_min: float, u: np.ndarray,
                     x_max: float | None = None) -> np.ndarray:
    """Inverse-CDF sample of a (possibly truncated) continuous power-law.

    Two regimes, keyed on whether an upper bound is supplied:

    * **Bounded** (``x_max`` finite) — sample the *truncated* power-law on
      ``[x_min, x_max]`` by its own inverse CDF
      ``x = [x_min^(1−α) + u·(x_max^(1−α) − x_min^(1−α))]^(1/(1−α))`` (log-uniform
      ``x_min·(x_max/x_min)^u`` in the ``α → 1`` limit). This is **not** the same
      as sampling the unbounded law and clipping at ``x_max``: clipping piles all
      the tail mass that would exceed ``x_max`` into a point mass *at* ``x_max``,
      so an ``α`` near 1 collapses to a spike at the bound (that clipping bug once
      inflated ``cs_freq``'s W1 ~9×). Because the bounded law is normalisable for
      any finite α, the ``α ≤ 1`` rejection below applies only to the unbounded
      regime.
    * **Unbounded** (``x_max`` ``None``/non-finite) — the Pareto inverse CDF
      ``x = x_min·(1 − u)^(−1/(α − 1))``, which needs ``α > 1`` to be normalisable;
      non-finite / extreme draws are capped at the finite 99.9th percentile so the
      reconstructed sample — and hence W1 — stays finite.

    Returns an empty array when the parameters are unusable (NaN, ``x_min ≤ 0``,
    or ``α ≤ 1`` in the unbounded regime).
    """
    if not np.isfinite([alpha, x_min]).all() or x_min <= 0:
        return np.array([], dtype=float)
    if x_max is not None and np.isfinite(x_max):
        if x_max <= x_min:
            return np.full_like(u, float(x_min))
        if abs(alpha - 1.0) < 1e-6:
            return x_min * (x_max / x_min) ** u          # log-uniform (α → 1)
        lo, hi = x_min ** (1.0 - alpha), x_max ** (1.0 - alpha)
        return (lo + u * (hi - lo)) ** (1.0 / (1.0 - alpha))
    if alpha <= 1.0:
        return np.array([], dtype=float)
    with np.errstate(over="ignore"):
        vals = x_min * (1.0 - u) ** (-1.0 / (alpha - 1.0))
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.array([], dtype=float)
    cap = float(np.percentile(finite, 99.9))
    return np.minimum(np.where(np.isfinite(vals), vals, cap), cap)


def _reconstruct_sample(fit, kind: str, u: np.ndarray) -> np.ndarray:
    """Return a representative sample for ``fit`` of the given ``kind``.

    ``u`` are shared uniform deviates (common random numbers), so two identical
    fits produce identical samples and hence W1 = 0. Returns an empty array when
    the fit is unavailable (NaN parameters), which :func:`wasserstein1` treats as
    "distance undefined".
    """
    if kind == QUANTILE:
        qs = np.asarray(fit, dtype=float)
        if not np.isfinite(qs).all():
            return np.array([], dtype=float)
        return np.interp(u, QUANTILE_LEVELS, qs)
    if kind == POWERLAW:
        return _powerlaw_sample(fit.alpha, fit.xmin, u)
    if kind == TRUNC_POWERLAW:
        return _powerlaw_sample(fit.alpha, fit.v_min, u, x_max=fit.v_max)
    if kind == EXP_DECAY:
        if not np.isfinite([fit.rate, fit.scale]).all():
            return np.array([], dtype=float)
        ranks = np.arange(_EXP_DECAY_RANKS, dtype=float)
        return fit.scale * np.exp(-fit.rate * ranks)
    raise ValueError(f"Unknown distribution kind {kind!r}")


def wasserstein1(fit_a, fit_b, kind: str) -> float:
    """Wasserstein-1 distance between two fits of the same ``kind``.

    Reconstructs a deterministic sample from each fit (using shared uniform draws
    so identical fits give exactly 0) and returns
    ``scipy.stats.wasserstein_distance`` between them. Returns NaN when either fit
    is unavailable (so its reconstructed sample is empty).

    Args:
        fit_a: the reference (target) fit.
        fit_b: the comparison (synthetic) fit.
        kind: one of the module-level kind constants.
    """
    u = np.random.default_rng(_SEED).random(_N_SAMPLE)
    sa = _reconstruct_sample(fit_a, kind, u)
    sb = _reconstruct_sample(fit_b, kind, u)
    if sa.size == 0 or sb.size == 0:
        return float("nan")
    return float(scipy.stats.wasserstein_distance(sa, sb))


def reconstructed_iqr(fit, kind: str) -> float:
    """Interquartile range of the sample reconstructed from ``fit``.

    Used to scale-normalise a Wasserstein-1 distance (``W1 / IQR``) so distances
    from distributions on very different scales are comparable. Returns NaN when
    the fit is unavailable.
    """
    u = np.random.default_rng(_SEED).random(_N_SAMPLE)
    s = _reconstruct_sample(fit, kind, u)
    if s.size == 0:
        return float("nan")
    q75, q25 = np.percentile(s, [75, 25])
    return float(q75 - q25)
