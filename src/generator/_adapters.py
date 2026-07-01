"""Reduced-signature adapters.

The reduced signature blocks (``signature``) store the *parameters of a
distribution family* rather than the raw moments the generator originally read
from the full blocks. These helpers reconstruct the few quantities Stage 1
needs from those parameters, so the Stage-1/2/3 logic stays unchanged.
"""

import math

import numpy as np
from scipy.special import zeta

from signature import QUANTILE_LEVELS

# Index of the median (0.5) level within QUANTILE_LEVELS, for scalar summaries.
_MEDIAN_IDX = QUANTILE_LEVELS.index(0.5)


def _quantile_mean(fit) -> float:
    """Mean of a reduced-signature quantile fit (NaN when the fit is absent).

    ``fit`` is a ``QuantileFit``-shaped tuple of sample quantiles at
    :data:`QUANTILE_LEVELS`; the mean is the trapezoid integral of the quantile
    function over the levels (``∫₀¹ Q(u) du``), close enough for sizing the CS
    budget.
    """
    if fit is None:
        return float("nan")
    qs = np.asarray(fit, dtype=float)
    if not np.isfinite(qs).all():
        return float("nan")
    return float(np.trapezoid(qs, QUANTILE_LEVELS))


def _functionality_from_alpha(fit, floor: float = 0.1) -> float:
    """Estimate mean relation (inverse-)functionality from a multiplicity-α fit.

    Reduced Block B drops the per-relation ``functionality`` dict and instead
    stores the spread of per-relation multiplicity power-law exponents as a
    quantile function. For a discrete power-law ``p(m) ∝ m^(−α)`` on ``m ≥ 1`` the
    fraction of single-object slots is ``P(m=1) = 1/ζ(α)``; the median quantile is
    the typical per-relation α. Falls back to 1.0 (fully functional) when α is
    unavailable or ≤ 1 (where ζ diverges). Clamped to ``[floor, 1.0]`` to match
    the old clip bounds.
    """
    alpha = float(fit[_MEDIAN_IDX]) if fit is not None else float("nan")
    if math.isnan(alpha) or alpha <= 1.0:
        return 1.0
    return float(np.clip(1.0 / zeta(alpha), floor, 1.0))


def sample_quantiles_trunc(fit, n: int, rng: np.random.Generator):
    """Sample ``n`` values from a quantile-function fit, or ``None``.

    ``fit`` is a ``QuantileFit``-shaped tuple of sample quantiles at
    :data:`QUANTILE_LEVELS` (works for both the NamedTuple and a plain decoded
    tuple). Draws by inverse-transform sampling — interpolating the stored
    quantile function at uniform deviates via ``np.interp`` — which naturally
    truncates to the stored ``[q@0, q@1]`` range. Returns ``None`` when the fit
    is unavailable (NaN params), so callers fall back to a budget-derived /
    neutral default.
    """
    qs = np.asarray(fit, dtype=float)
    if not np.isfinite(qs).all():
        return None
    return np.interp(rng.random(n), QUANTILE_LEVELS, qs)


def sample_powerlaw(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` continuous power-law(α) draws on ``[1, ∞)`` via inverse-CDF.

    For ``p(x) ∝ x^(−α)`` on ``x ≥ 1`` the inverse CDF is
    ``x = (1 − u)^(−1/(α−1))``. Returns uniform ones when ``α`` is NaN or ``≤ 1``
    (no usable tail shape → callers get equal weights = the neutral fallback).
    """
    if n <= 0:
        return np.array([], dtype=float)
    if math.isnan(alpha) or alpha <= 1.0:
        return np.ones(n, dtype=float)
    u = rng.random(n)
    return (1.0 - u) ** (-1.0 / (alpha - 1.0))


def _reconstruct_singular_values(exp_fit, k: int = 10) -> np.ndarray:
    """Rebuild a singular-value spectrum from an exp-decay fit ``(rate, scale)``.

    Reduced Block C stores the co-occurrence spectrum as
    ``value(rank r) = scale·exp(−rate·r)`` instead of the raw singular values.
    Only the relative magnitudes matter to ``_sample_type_relation_probs`` (it
    normalises them), so a ``k``-point reconstruction is sufficient. Returns an
    empty array when the fit is unavailable, which the caller treats as "no
    co-occurrence signal".
    """
    if exp_fit is None or math.isnan(exp_fit.rate) or math.isnan(exp_fit.scale):
        return np.array([], dtype=float)
    ranks = np.arange(k, dtype=float)
    return exp_fit.scale * np.exp(-exp_fit.rate * ranks)
