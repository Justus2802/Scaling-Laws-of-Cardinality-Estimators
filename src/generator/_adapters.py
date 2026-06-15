"""Reduced-signature adapters.

The reduced signature blocks (``signature_reduced``) store the *parameters of a
distribution family* rather than the raw moments the generator originally read
from the full blocks. These helpers reconstruct the few quantities Stage 1
needs from those parameters, so the Stage-1/2/3 logic stays unchanged.
"""

import math

import numpy as np
import scipy.stats
from scipy.special import zeta


def _skewnorm_mean(fit) -> float:
    """Mean of a reduced-signature skew-normal fit (NaN when the fit is absent).

    ``fit`` is a ``SkewNormFit`` ``(loc, scale, shape, lo, hi)``; the mean uses
    the scipy parameterisation and ignores the truncation cutoffs (close enough
    for sizing the CS budget).
    """
    if fit is None or math.isnan(fit.loc) or math.isnan(fit.scale) or math.isnan(fit.shape):
        return float("nan")
    return float(scipy.stats.skewnorm.mean(fit.shape, loc=fit.loc, scale=fit.scale))


def _functionality_from_alpha(fit, floor: float = 0.1) -> float:
    """Estimate mean relation (inverse-)functionality from a multiplicity-α fit.

    Reduced Block B drops the per-relation ``functionality`` dict and instead
    stores the spread of per-relation multiplicity power-law exponents as a
    skew-normal. For a discrete power-law ``p(m) ∝ m^(−α)`` on ``m ≥ 1`` the
    fraction of single-object slots is ``P(m=1) = 1/ζ(α)``; the skew-normal
    ``loc`` is the typical per-relation α. Falls back to 1.0 (fully functional)
    when α is unavailable or ≤ 1 (where ζ diverges). Clamped to ``[floor, 1.0]``
    to match the old clip bounds.
    """
    alpha = fit.loc if fit is not None else float("nan")
    if math.isnan(alpha) or alpha <= 1.0:
        return 1.0
    return float(np.clip(1.0 / zeta(alpha), floor, 1.0))


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
