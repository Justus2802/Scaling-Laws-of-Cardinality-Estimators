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


# Default tail exponent when the degree power-law fit is degenerate but the
# p90/max quantile targets are available (typical KG degree exponent).
_DEGSEQ_FALLBACK_ALPHA = 2.5
# Fraction of nodes drawn from the [p90, max] tail — by definition of the 90th
# percentile, 10% of measured degrees lie at or above it.
_DEGSEQ_TAIL_FRACTION = 0.1


def sample_degree_sequence(alpha: float, p90: float, d_max: float, mean_deg: float,
                           n: int, rng: np.random.Generator):
    """Sample an ``n``-node target degree sequence from signature-vector components.

    Uses only quantities that are part of the comparison vector — the degree
    power-law exponent ``alpha``, the ``p90`` / ``max`` degree scalars and the
    graph mean degree — never the raw measured degree arrays (which Stage 2
    must not depend on). Construction:

    * the top 10% of nodes draw from a power law truncated to ``[p90, max]``
      whose exponent is **extreme-value matched**: ``α_tail = 1 +
      ln(n_tail)/ln(max/p90)``, so the expected maximum of ``n_tail`` draws
      above p90 lands on the target max. The global fit ``alpha`` is too
      shallow for this range (the empirical tail has a finite-size cutoff) and
      would overshoot mid-tail mass (p99, star counts);
    * the remaining 90% draw from the same power law truncated to ``[1, p90]``
      (the fit's own body range), then a random subset is zero-inflated so the
      overall sequence mean matches ``mean_deg`` (edge conservation) without
      distorting the body shape.

    Returns ``None`` when ``p90``/``max`` are unavailable (NaN or < 1) — Stage 2
    then wires without degree steering.

    :param alpha: degree power-law exponent (falls back to 2.5 when unusable).
    :param p90: 90th-percentile degree from the signature.
    :param d_max: maximum degree from the signature.
    :param mean_deg: target mean degree (E/V from Block A).
    :param n: number of target values to draw.
    :param rng: RNG for the sampling.
    :returns: int64 array of length ``n``, or ``None``.
    """
    if n <= 0 or not np.isfinite([p90, d_max]).all() or d_max < 1:
        return None
    if math.isnan(alpha) or alpha <= 1.0:
        alpha = _DEGSEQ_FALLBACK_ALPHA
    lo = max(float(p90), 1.0)
    hi = max(float(d_max), lo)

    n_tail = max(1, int(round(n * _DEGSEQ_TAIL_FRACTION)))
    n_body = n - n_tail

    def _trunc_powerlaw(lo_t: float, hi_t: float, size: int, exp_t: float) -> np.ndarray:
        """Inverse-CDF draws from a power law(exp_t) truncated to [lo_t, hi_t]."""
        if size <= 0:
            return np.array([], dtype=np.int64)
        if hi_t <= lo_t:
            return np.full(size, int(round(lo_t)), dtype=np.int64)
        a1 = 1.0 - exp_t
        u = rng.random(size)
        vals = (lo_t ** a1 + u * (hi_t ** a1 - lo_t ** a1)) ** (1.0 / a1)
        return np.round(vals).astype(np.int64)

    # Extreme-value-matched tail exponent: expected max of n_tail draws = hi.
    alpha_tail = 1.0 + math.log(n_tail) / math.log(hi / lo) if hi > lo and n_tail > 1 else alpha
    tail = _trunc_powerlaw(lo, hi, n_tail, alpha_tail)
    if hi > lo:
        tail[np.argmax(tail)] = int(round(hi))   # ensure the sampled max hits the target max

    # Body: same power law on its own [1, p90] range, then zero-inflate a random
    # subset so the overall mean matches mean_deg (edge conservation) without
    # distorting the body shape. A body lighter than the budget is left as-is —
    # Stage 2's edge-budget top-up covers the shortfall.
    body = _trunc_powerlaw(1.0, lo, n_body, alpha)
    if n_body > 0 and np.isfinite(mean_deg):
        excess = float(body.sum() + tail.sum()) - n * float(mean_deg)
        nz = np.where(body > 0)[0]
        if excess > 0 and nz.size:
            mean_nz = float(body[nz].mean())
            k = min(nz.size, int(round(excess / max(mean_nz, 1e-9))))
            if k > 0:
                body[rng.choice(nz, size=k, replace=False)] = 0
    seq = np.concatenate([body, tail]) if n_body > 0 else tail
    return rng.permutation(seq)


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
