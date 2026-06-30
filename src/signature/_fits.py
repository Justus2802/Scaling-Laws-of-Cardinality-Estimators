"""Distribution fitters for the reduced (non-over-determined) signature.

Each fitter returns a small fixed-length ``NamedTuple`` so the blocks'
``as_vector`` stays fixed-length, and short-circuits to all-NaN when there are
too few samples to fit — matching the existing ``_fit_powerlaw`` contract in
``signature._utils``. The implementations delegate to library code wherever
possible: ``scipy.stats.skewnorm`` for the skew-normal fits,
``scipy.stats.linregress`` for the log-linear (exponential-decay / offset) fits,
and the ``powerlaw`` package (via the shared ``_fit_powerlaw``) for the
power-law fits.

The reduced signature stores the *parameters of the distribution family* each
quantity follows (see ``docs/signature_redesign.md``), not raw moments, so the
shape can be regenerated at sampling time.
"""

import contextlib
import io
import warnings
from typing import NamedTuple, Optional

import numpy as np
import scipy.stats

# Reuse the existing power-law fitter and the shared minimum-sample threshold so
# the two signatures agree on when a fit is trustworthy.
from ._utils import MIN_SAMPLES_FOR_FIT, _fit_powerlaw

# Rank curves (singular-value spectra, per-type entropy) are inherently short —
# only a handful of ranks exist — so they use a smaller minimum than the
# sample-distribution fitters.
_MIN_RANK_POINTS = 3

_NAN = float("nan")

# Probability levels at which sample distributions are summarised as a quantile
# function (the non-parametric replacement for the skew-normal fits). The first
# and last levels (0.0 / 1.0) are the lower/upper clip bounds, so the quantile
# vector folds in the old truncation. Kept as a module constant — not stored per
# fit — so the feature-vector / JSON length stays fixed at ``len(QUANTILE_LEVELS)``.
QUANTILE_LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


# ── return types ───────────────────────────────────────────────────────────────


class QuantileFit(NamedTuple):
    """Empirical quantile function evaluated at :data:`QUANTILE_LEVELS`.

    Each field is the sample quantile at the corresponding level; ``q0``/``q100``
    are the min/max (the truncation cutoffs). The values are non-decreasing by
    construction, so they double as an invertible CDF for inverse-transform
    sampling and their L1 difference is the Wasserstein-1 distance.
    """
    q0: float
    q10: float
    q25: float
    q50: float
    q75: float
    q90: float
    q100: float


class ExpDecayFit(NamedTuple):
    """Exponential-decay rank curve ``value(rank k) ≈ scale · exp(−rate · k)``.

    ``rate`` (λ) is how fast values fall with rank; ``scale`` (A) is the
    magnitude of the top-ranked value.
    """
    rate: float
    scale: float


class TruncPowerLawFit(NamedTuple):
    """Truncated power-law ``p(v) ∝ v^(−alpha)`` on ``[v_min, v_max]``."""
    alpha: float
    v_min: float
    v_max: float


class ZipfFit(NamedTuple):
    """Zipf / power-law frequency law: ``exponent`` plus the fit's ``x_min``."""
    exponent: float
    x_min: float


# The NamedTuple field count must track the level grid (they are splatted together).
assert len(QuantileFit._fields) == len(QUANTILE_LEVELS)


def nan_quantiles() -> QuantileFit:
    """Return the canonical 'fit unavailable' quantile value (all NaN)."""
    return QuantileFit(*([_NAN] * len(QUANTILE_LEVELS)))


def nan_exp_decay() -> ExpDecayFit:
    """Return the canonical 'fit unavailable' exponential-decay value."""
    return ExpDecayFit(_NAN, _NAN)


def nan_trunc_powerlaw() -> TruncPowerLawFit:
    """Return the canonical 'fit unavailable' truncated-power-law value."""
    return TruncPowerLawFit(_NAN, _NAN, _NAN)


def nan_zipf() -> ZipfFit:
    """Return the canonical 'fit unavailable' Zipf value."""
    return ZipfFit(_NAN, _NAN)


# ── fitters ────────────────────────────────────────────────────────────────────


def fit_skewnorm(
    values,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> SkewNormFit:
    """Fit a (truncated) skew-normal to a 1-D sample.

    Uses ``scipy.stats.skewnorm.fit`` (method-of-moments-seeded MLE). The
    cutoffs default to the observed min/max; pass ``lo``/``hi`` to pin them to
    fixed bounds (e.g. the ≈[1.4, 3.0] range for per-relation multiplicity α).
    Returns all-NaN when fewer than ``MIN_SAMPLES_FOR_FIT`` finite samples are
    available or the fit raises.

    Args:
        values: iterable of real-valued samples.
        lo: lower truncation cutoff (default: observed minimum).
        hi: upper truncation cutoff (default: observed maximum).
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_SAMPLES_FOR_FIT:
        return nan_skewnorm()
    try:
        with warnings.catch_warnings(), \
             np.errstate(divide="ignore", invalid="ignore"), \
             contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            shape, loc, scale = scipy.stats.skewnorm.fit(arr)
        obs_lo = float(arr.min()) if lo is None else float(lo)
        obs_hi = float(arr.max()) if hi is None else float(hi)
        return SkewNormFit(float(loc), float(scale), float(shape), obs_lo, obs_hi)
    except Exception:
        return nan_skewnorm()


def fit_exp_decay_rank(values) -> ExpDecayFit:
    """Fit an exponential-decay curve to a rank-ordered set of magnitudes.

    The values are sorted descending, restricted to strictly positive entries,
    and a line is fit to ``ln(value)`` against 0-based rank via
    ``scipy.stats.linregress``. With ``ln v_k = ln A − λ·k`` the slope gives
    ``rate = −slope`` and ``scale = exp(intercept)``. Used for the quantities
    whose rank order matters (co-occurrence and type-relation singular values,
    per-type relation entropy). Returns all-NaN below ``_MIN_RANK_POINTS``.

    Args:
        values: iterable of magnitudes (any order; sorted internally).
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < _MIN_RANK_POINTS:
        return nan_exp_decay()
    arr = np.sort(arr)[::-1]
    ranks = np.arange(arr.size, dtype=float)
    try:
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")
            result = scipy.stats.linregress(ranks, np.log(arr))
        rate = float(-result.slope)
        scale = float(np.exp(result.intercept))
        return ExpDecayFit(rate, scale)
    except Exception:
        return nan_exp_decay()


def fit_truncated_powerlaw(values) -> TruncPowerLawFit:
    """Fit a truncated power-law ``p(v) ∝ v^(−alpha)`` bounded to the data range.

    Delegates to ``powerlaw.Fit`` with ``xmin``/``xmax`` pinned to the observed
    range (so the fit describes the *value set* without an open tail), reading
    the exponent from ``fit.power_law.alpha``. Used for the two-step pair
    path-count distribution, which is inherently bounded. Returns all-NaN below
    ``MIN_SAMPLES_FOR_FIT`` or when the range is degenerate (``v_min == v_max``).

    Args:
        values: iterable of non-negative values (zeros/negatives dropped).
    """
    import powerlaw

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < MIN_SAMPLES_FOR_FIT:
        return nan_trunc_powerlaw()
    v_min = float(arr.min())
    v_max = float(arr.max())
    if v_min >= v_max:
        return nan_trunc_powerlaw()
    try:
        with warnings.catch_warnings(), \
             np.errstate(divide="ignore", invalid="ignore"), \
             contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            fit = powerlaw.Fit(arr, xmin=v_min, xmax=v_max, discrete=True, verbose=False)
            alpha = float(fit.power_law.alpha)
        return TruncPowerLawFit(alpha, v_min, v_max)
    except Exception:
        return nan_trunc_powerlaw()


def fit_zipf(counts) -> ZipfFit:
    """Fit a Zipf / power-law frequency law to a set of usage counts.

    Reuses the shared ``_fit_powerlaw`` (the ``powerlaw`` package) and exposes
    its exponent and ``x_min``. Used for relation-usage frequency.

    Args:
        counts: iterable of per-item occurrence counts.
    """
    arr = np.asarray(list(counts), dtype=float)
    stats = _fit_powerlaw(arr)
    return ZipfFit(float(stats.alpha), float(stats.xmin))


def fit_cs_size_offset(cs_size, mult) -> float:
    """Estimate the CS-size→multiplicity offset slope ``a`` (G2b).

    Ordinary least squares of ``ln(mult)`` on ``ln(cs_size)`` over per-edge
    pairs, via ``scipy.stats.linregress``. ``a`` is the global coefficient in
    the multiplicative factor ``cs_size^a`` applied to the multiplicity location
    at generation time. Returns NaN when there are too few pairs or no variation
    in ``cs_size`` (slope undefined).

    Args:
        cs_size: per-edge characteristic-set size of the subject (positive).
        mult: per-edge object-multiplicity of the subject on that relation.
    """
    x = np.asarray(list(cs_size), dtype=float)
    y = np.asarray(list(mult), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if x.size < MIN_SAMPLES_FOR_FIT:
        return _NAN
    log_x = np.log(x)
    if np.ptp(log_x) == 0.0:  # no spread in cs_size → slope undefined
        return _NAN
    try:
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")
            return float(scipy.stats.linregress(log_x, np.log(y)).slope)
    except Exception:
        return _NAN
