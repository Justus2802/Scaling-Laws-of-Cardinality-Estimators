"""Shared plotting helpers that overlay a fitted curve on the raw data.

Every reduced block keeps the *unsummarised* samples it fit (per-relation
exponents, singular values, row entropies, path lengths, …) on the object, and
its ``visualize`` draws those samples with the fitted distribution overlaid, so
the fit can be eyeballed against the data it came from. The curve evaluations
reuse the same library parameterisations the fits use (the stored quantile
function for sample distributions, the exp-decay/power-law closed forms).
"""

import numpy as np

from ._fits import QUANTILE_LEVELS


def overlay_quantiles(ax, values: np.ndarray, fit, *, bins: int = 20,
                      label: str = "data", color: str = "steelblue") -> bool:
    """Histogram ``values`` and overlay the stored quantile markers.

    Draws a vertical line at each stored quantile (the median dashed, the rest
    dotted), so the non-parametric quantile fit can be eyeballed against the
    sample it summarises.

    Args:
        ax: matplotlib axis.
        values: the raw sample the quantiles were computed from.
        fit: a ``QuantileFit`` (quantiles at :data:`QUANTILE_LEVELS`).
        bins: histogram bin count.
        label: legend label for the data histogram.
        color: histogram colour.

    Returns:
        True if anything was drawn, False if there was no data.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    ax.hist(values, bins=bins, alpha=0.6, label=label, color=color)
    qs = np.asarray(fit, dtype=float)
    if np.isfinite(qs).all():
        for level, q in zip(QUANTILE_LEVELS, qs):
            is_median = level == 0.5
            ax.axvline(q, color="r", linewidth=1.5 if is_median else 1.0,
                       linestyle="--" if is_median else ":",
                       label="quantile fit" if is_median else None)
        ax.legend(fontsize=8)
    return True


def overlay_exp_decay_rank(ax, values: np.ndarray, fit, *,
                           label: str = "data", color: str = "mediumpurple") -> bool:
    """Plot rank-ordered ``values`` and overlay the fitted exp-decay curve.

    Args:
        ax: matplotlib axis.
        values: the raw magnitudes (sorted descending internally).
        fit: an ``ExpDecayFit`` (rate, scale).

    Returns:
        True if anything was drawn, False if there was no positive data.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return False
    ranked = np.sort(values)[::-1]
    k = np.arange(ranked.size)
    ax.plot(k, ranked, "o", markersize=4, color=color, label=label)
    if np.isfinite([fit.rate, fit.scale]).all():
        ax.plot(k, fit.scale * np.exp(-fit.rate * k), "r-", linewidth=1.5,
                label=f"exp-decay (λ={fit.rate:.2f})")
        ax.legend(fontsize=8)
    return True


def _overlay_open_powerlaw(ax, values: np.ndarray, exponent: float, x_min: float, *,
                           label: str, color: str, fit_label: str) -> bool:
    """Empirical CCDF of ``values`` with the fitted open-tailed power-law overlaid.

    Plots ``P(X ≥ x)`` (empirical complementary CDF) on a log-log axis, which
    avoids histogram binning artefacts and normalisation ambiguity. The fitted
    line is ``P(X ≥ xmin) · (x / xmin)^(1 − exponent)`` for ``x ≥ xmin``, the
    theoretical CCDF of a continuous power-law — a straight line on log-log when
    the fit is good. Follows Clauset et al. (2009) §3.3.

    Args:
        ax: matplotlib axis.
        values: the raw value set the law was fit to.
        exponent: the power-law / Zipf exponent (α, must be > 1 for a valid CCDF).
        x_min: lower cutoff of the fitted tail.
        label: legend label for the empirical CCDF markers.
        color: data marker colour.
        fit_label: legend label for the fitted line.

    Returns:
        True if anything was drawn, False if there was no positive data.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return False
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    # P(X >= x_(i)) for the i-th order statistic (1-indexed from the top)
    ccdf = np.arange(n, 0, -1) / n
    ax.loglog(sorted_vals, ccdf, "o", markersize=3, color=color, label=label, alpha=0.7)
    if np.isfinite([exponent, x_min]).all() and exponent > 1:
        p_xmin = float((values >= x_min).sum()) / n
        xs = np.logspace(np.log10(x_min), np.log10(values.max()), 200)
        ys = p_xmin * (xs / x_min) ** (1.0 - exponent)
        ax.loglog(xs, ys, "r-", linewidth=1.5, label=fit_label)
        ax.legend(fontsize=8)
    return True


def overlay_powerlaw(ax, values: np.ndarray, fit, *,
                     label: str = "data", color: str = "seagreen") -> bool:
    """Log-log histogram of ``values`` with the fitted power-law tail overlaid.

    Companion to :func:`overlay_truncated_powerlaw` for the open-tailed
    ``PowerLawStats`` fits (``P(x) ∝ x^(−alpha)`` above a KS-optimised
    ``xmin``).

    Args:
        ax: matplotlib axis.
        values: the raw value set the power-law was fit to.
        fit: a ``PowerLawStats`` (uses ``alpha`` and ``xmin``).

    Returns:
        True if anything was drawn, False if there was no positive data.
    """
    return _overlay_open_powerlaw(ax, values, fit.alpha, fit.xmin, label=label,
                                  color=color, fit_label=f"power-law (α={fit.alpha:.2f})")


def overlay_zipf(ax, values: np.ndarray, fit, *,
                 label: str = "data", color: str = "teal") -> bool:
    """Log-log histogram of ``values`` with the fitted Zipf tail overlaid.

    Zipf variant of :func:`overlay_powerlaw` for ``ZipfFit`` (``exponent`` above
    ``x_min``); used for relation-usage frequency counts.

    Args:
        ax: matplotlib axis.
        values: the raw usage counts the Zipf law was fit to.
        fit: a ``ZipfFit`` (uses ``exponent`` and ``x_min``).

    Returns:
        True if anything was drawn, False if there was no positive data.
    """
    return _overlay_open_powerlaw(ax, values, fit.exponent, fit.x_min, label=label,
                                  color=color, fit_label=f"Zipf (α={fit.exponent:.2f})")


def overlay_truncated_powerlaw(ax, values: np.ndarray, fit, *,
                               label: str = "data", color: str = "darkorange") -> bool:
    """Empirical CCDF of ``values`` with the fitted truncated power-law overlaid.

    The theoretical CCDF of a truncated power-law ``p(x) ∝ x^(−α)`` on
    ``[v_min, v_max]`` is:
    ``P(X ≥ x) = (v_max^(1−α) − x^(1−α)) / (v_max^(1−α) − v_min^(1−α))``,
    normalised to the empirical fraction of data with value ≥ v_min.

    Args:
        ax: matplotlib axis.
        values: the raw value set the power-law was fit to.
        fit: a ``TruncPowerLawFit`` (alpha, v_min, v_max).

    Returns:
        True if anything was drawn, False if there was no positive data.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return False
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    ccdf = np.arange(n, 0, -1) / n
    ax.loglog(sorted_vals, ccdf, "o", markersize=3, color=color, label=label, alpha=0.7)
    if np.isfinite([fit.alpha, fit.v_min, fit.v_max]).all() and fit.alpha > 0 and fit.v_min < fit.v_max:
        alpha, v_min, v_max = fit.alpha, fit.v_min, fit.v_max
        p_vmin = float((values >= v_min).sum()) / n
        xs = np.logspace(np.log10(v_min), np.log10(v_max), 200)
        if abs(alpha - 1.0) < 1e-6:
            ys_raw = np.log(v_max / xs) / np.log(v_max / v_min)
        else:
            denom = v_max ** (1 - alpha) - v_min ** (1 - alpha)
            ys_raw = (v_max ** (1 - alpha) - xs ** (1 - alpha)) / denom
        ys = np.clip(ys_raw, 0, 1) * p_vmin
        ax.loglog(xs, ys, "r-", linewidth=1.5, label=f"trunc. power-law (α={alpha:.2f})")
        ax.legend(fontsize=8)
    return True
