"""Shared utilities used across signature blocks."""

import contextlib
import io
import warnings
from dataclasses import dataclass
from typing import Iterable, NamedTuple

import numpy as np
import powerlaw

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
MIN_SAMPLES_FOR_FIT = 10  # below this, powerlaw.Fit results are dominated by noise


class PowerLawStats(NamedTuple):
    """Six-number summary of a power-law fit via `powerlaw.Fit`.

    Used uniformly for the two aggregate degree distributions and for every
    per-relation multiplicity distribution. An all-NaN instance means the fit
    was skipped (too few samples) or raised internally.
    """
    alpha: float          # power-law exponent α from P(x) ∝ x^(-α)
    xmin: float           # lower-bound of the tail (KS-optimized by powerlaw)
    ks: float             # KS distance of the power-law fit itself
    D_lognormal: float    # KS distance for the alternative lognormal fit
    D_exponential: float  # KS distance for the alternative exponential fit
    D_truncated: float    # KS distance for the alternative truncated_power_law fit


def _nan_power_law_stats() -> PowerLawStats:
    """Return an all-NaN PowerLawStats — the canonical 'fit unavailable' value."""
    return PowerLawStats(*([float("nan")] * 6))


def _fit_powerlaw(data: np.ndarray) -> PowerLawStats:
    """Fit a power-law to a 1-D non-negative integer array and report KS distances.

    Filters to strictly positive samples (the `powerlaw` package rejects zeros).
    If fewer than MIN_SAMPLES_FOR_FIT positive samples remain, short-circuits
    to all-NaN — Clauset/Shalizi/Newman (2009, Sec. 3.3) show that fitted α
    has prohibitively wide confidence intervals on small samples, so the fit
    would produce noise. Skipping also avoids the package's stdout chatter,
    division-by-zero warnings, and per-call overhead on long-tail relations.

    The fit is pinned at ``xmin=1`` (see the call site), so the reported
    ``xmin`` is always 1.0 and ``alpha`` is the MLE over the full positive
    range rather than an auto-searched tail.

    Returns a PowerLawStats with:
      - alpha, xmin, ks from `fit.power_law` (the power-law fit itself)
      - D_lognormal, D_exponential, D_truncated from each alternative's own KS
        distance (`fit.<dist>.D`). Smaller D ⇒ that distribution fits better.

    Any exception inside the fitter is swallowed and yields all-NaN.
    """
    positive = data[data > 0]
    if positive.size < MIN_SAMPLES_FOR_FIT:
        return _nan_power_law_stats()
    try:
        with warnings.catch_warnings(), \
             np.errstate(divide="ignore", invalid="ignore"), \
             contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            # Pin xmin=1 (the domain minimum after the `data > 0` filter) so the
            # fitted alpha describes the whole range its consumers sample from,
            # not an auto-searched Clauset-Shalizi-Newman tail that then gets
            # extrapolated down to the body. Mirrors fit_truncated_powerlaw's
            # pinning for Block D. See docs/signature.md (deviations) / plan 2.13.
            fit = powerlaw.Fit(positive, discrete=True, xmin=1, verbose=False)
            return PowerLawStats(
                alpha=float(fit.power_law.alpha),
                xmin=float(fit.power_law.xmin),
                ks=float(fit.power_law.D),
                D_lognormal=float(fit.lognormal.D),
                D_exponential=float(fit.exponential.D),
                D_truncated=float(fit.truncated_power_law.D),
            )
    except Exception:
        return _nan_power_law_stats()


@dataclass
class ValueSummary:
    mean: float
    std: float
    min: float
    max: float
    median: float


def _nan_value_summary() -> ValueSummary:
    nan = float("nan")
    return ValueSummary(nan, nan, nan, nan, nan)


def _summarize_values(values: Iterable[float]) -> ValueSummary:
    """Return NaN-safe (mean, std, min, max, median) over an iterable of floats.

    Returns all-NaN when the iterable is empty or all values are NaN
    (guards the `nanmin`/`nanmax` "all-NaN slice" warning).
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return _nan_value_summary()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ValueSummary(
            mean=float(np.nanmean(arr)),
            std=float(np.nanstd(arr)),
            min=float(np.nanmin(arr)),
            max=float(np.nanmax(arr)),
            median=float(np.nanmedian(arr)),
        )
