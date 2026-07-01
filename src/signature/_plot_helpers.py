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


def overlay_truncated_powerlaw(ax, values: np.ndarray, fit, *,
                               label: str = "data", color: str = "darkorange") -> bool:
    """Log-log histogram of ``values`` with the fitted truncated power-law line.

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
    bins = np.logspace(np.log10(values.min()), np.log10(values.max() + 1), 25)
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.loglog(centers[counts > 0], counts[counts > 0], "o", markersize=4,
              color=color, label=label)
    if np.isfinite([fit.alpha, fit.v_min, fit.v_max]).all() and fit.alpha > 0:
        xs = np.linspace(fit.v_min, fit.v_max, 200)
        ys = xs ** (-fit.alpha)
        mask = (centers >= fit.v_min) & (counts > 0)
        if mask.any():
            ys = ys / ys.sum() * counts[mask].sum()
        ax.loglog(xs, ys, "r-", linewidth=1.5, label=f"trunc. power-law (α={fit.alpha:.2f})")
        ax.legend(fontsize=8)
    return True
