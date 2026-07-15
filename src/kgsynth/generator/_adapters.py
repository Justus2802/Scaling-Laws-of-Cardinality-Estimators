"""Reduced-signature adapters.

The reduced signature blocks (``signature``) store the *parameters of a
distribution family* rather than the raw moments the generator originally read
from the full blocks. These helpers reconstruct the few quantities Stage 1
needs from those parameters, so the Stage-1/2/3 logic stays unchanged.
"""

import math

import numpy as np

from ..signature import QUANTILE_LEVELS


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


def sample_powerlaw_trunc(alpha: float, lo: float, hi: float, n: int,
                          rng: np.random.Generator) -> np.ndarray:
    """``n`` continuous power-law(α) draws truncated to ``[lo, hi]`` via inverse-CDF.

    For ``p(x) ∝ x^(−α)`` on ``lo ≤ x ≤ hi`` the inverse CDF is
    ``x = (lo^a + u·(hi^a − lo^a))^(1/a)`` with ``a = 1 − α``. Every power law in
    the generator is drawn this way: each quantity it samples (CS-template reuse,
    per-relation multiplicity, degree) is bounded, and the fits are truncated MLEs
    over those bounds. Drawing unbounded and clamping instead would deposit an atom
    of probability mass on ``hi`` rather than redistributing it over the support.

    Degenerate inputs collapse to a constant ``lo``, which normalises to equal
    weights — the neutral fallback: an unusable exponent (NaN, or ``α ≤ 1``, which
    includes the ``a = 0`` singularity) or an empty range (``hi ≤ lo``, or a NaN
    bound from a fit that did not converge).

    :param alpha: power-law exponent (NaN or ``≤ 1`` → flat).
    :param lo: lower bound of the support (floored at 1).
    :param hi: upper bound of the support.
    :param n: number of draws.
    :param rng: RNG for the sampling.
    :returns: float array of length ``n``, all values in ``[lo, hi]``.
    """
    if n <= 0:
        return np.array([], dtype=float)
    lo = float(lo) if math.isfinite(lo) and lo > 1.0 else 1.0
    hi = float(hi)
    if math.isnan(alpha) or alpha <= 1.0 or not math.isfinite(hi) or hi <= lo:
        return np.full(n, lo, dtype=float)
    a1 = 1.0 - alpha
    u = rng.random(n)
    return (lo ** a1 + u * (hi ** a1 - lo ** a1)) ** (1.0 / a1)


# Default tail exponent when the degree power-law fit is degenerate but the
# p90/max quantile targets are available (typical KG degree exponent).
_DEGSEQ_FALLBACK_ALPHA = 2.5
# Fraction of nodes drawn from the [p90, max] tail — by definition of the 90th
# percentile, 10% of measured degrees lie at or above it.
_DEGSEQ_TAIL_FRACTION = 0.1


def repair_degree_sum(
    seq: np.ndarray,
    target_sum: int,
    rng: np.random.Generator,
    *,
    floor: np.ndarray | None = None,
    adjustable: np.ndarray | None = None,
) -> np.ndarray:
    """Adjust ``seq`` so it sums to exactly ``target_sum``, two-sided and shape-safe.

    A degree sequence has to satisfy two things at once: its *shape* (the α / p90 /
    max targets) and its *sum* (the edge budget — Σ out = Σ in = E is what makes a
    directed wiring possible at all). They conflict, and the sum is the one that
    cannot bend: everything downstream is built against the edge count.

    So the sum is enforced here, and the distortion is pushed into the part of the
    distribution that carries the least information:

    * **Deficit** — spread the missing units uniformly over the adjustable entries.
      Uniform, *not* ∝ seq: a proportional top-up is a rescale, which multiplies the
      hubs by the same factor and destroys the p90/max targets the tail was built to
      hit (it inflated max by up to 2.3× before this existed).
    * **Surplus** — remove units from adjustable entries weighted by their headroom
      above ``floor``, so the trim comes off the entries that have room to give.

    ``adjustable`` (default: all) marks entries the repair may touch — callers pass a
    mask that **excludes the tail**, so the extreme-value-matched max survives intact.
    ``floor`` (default: 0) is a per-entry lower bound the trim will not cross; when the
    floor makes ``target_sum`` unreachable the sequence is left at its floor and the
    residual is returned unrepaired (the caller's budget is then simply infeasible).

    :param seq: integer degree sequence to repair (not modified in place).
    :param target_sum: the exact sum to hit.
    :param rng: RNG for the spread/trim draws.
    :param floor: per-entry lower bound for trimming.
    :param adjustable: boolean mask of entries the repair may change.
    :returns: int64 copy of ``seq`` summing to ``target_sum`` (floor permitting).
    """
    out = np.asarray(seq, dtype=np.int64).copy()
    n = out.size
    if n == 0:
        return out
    lower = (np.zeros(n, dtype=np.int64) if floor is None
             else np.asarray(floor, dtype=np.int64))
    idx = (np.arange(n) if adjustable is None
           else np.where(np.asarray(adjustable, dtype=bool))[0])
    if idx.size == 0:
        return out

    delta = int(target_sum) - int(out.sum())
    if delta > 0:
        out[idx] += rng.multinomial(delta, np.full(idx.size, 1.0 / idx.size))
    elif delta < 0:
        need = -delta
        while need > 0:
            headroom = np.maximum(out[idx] - lower[idx], 0)
            total = int(headroom.sum())
            if total == 0:
                break                      # floor-bound: cannot trim any further
            take = min(need, total)
            cut = rng.multinomial(take, headroom / total)
            cut = np.minimum(cut, headroom)   # a multinomial cell can overshoot its entry
            out[idx] -= cut
            need -= int(cut.sum())
    return out


def sample_degree_sequence(alpha: float, p90: float, d_max: float, mean_deg: float,
                           n: int, rng: np.random.Generator, active_frac: float = 1.0):
    """Sample an ``n``-node target degree sequence from signature-vector components.

    Uses only quantities that are part of the comparison vector — the degree
    power-law exponent ``alpha``, the ``p90`` / ``max`` degree scalars, the graph
    mean degree and the ``active_frac`` (share of entities with a nonzero degree on
    this side) — never the raw measured degree arrays (which Stage 2 must not depend
    on). Construction:

    * ``(1 − active_frac)·n`` nodes get degree **0**. Not every entity is a subject
      (or an object): on swdf only 30% of entities emit an edge. Spreading the whole
      out-budget over all ``n`` entities instead flattens the distribution *and* — via
      the ≥1-edge-per-CS-relation floor in Stage 2 — asks for ``Σ|CS|`` far above the
      edge budget (swdf: 606 500 vs 242 256), which forces that floor to be dropped and
      collapses the realised CS size from 6 to 1. The zeros fix both.
    * of the ``active_frac·n`` nonzero nodes, the top ``10%`` of *all* ``n`` draw from a
      power law truncated to ``[p90, max]`` whose exponent is **extreme-value matched**
      (``α_tail = 1 + ln(n_tail)/ln(max/p90)``) so the expected maximum of ``n_tail``
      draws above p90 lands on the target max. The tail fraction is ``0.1·n`` (of *all*
      entities), because p90/max are measured over all entities including the zeros;
    * the remaining active nodes draw from the same power law truncated to ``[1, p90]``
      and are repaired — up *or* down — so the whole sequence sums to exactly
      ``n · mean_deg`` (edge conservation). The repair is confined to that body: the
      tail carries p90/max and the zeros carry ``active_frac``, so neither is touched.

    Returns ``None`` when ``p90``/``max`` are unavailable (NaN or < 1) — Stage 2 then
    wires without degree steering.

    :param alpha: degree power-law exponent (falls back to 2.5 when unusable).
    :param p90: 90th-percentile degree from the signature (over all entities).
    :param d_max: maximum degree from the signature.
    :param mean_deg: target mean degree — the *content* mean over all entities
        (rdf:type edges are wired outside this budget and are excluded from Block B's
        degree fits).
    :param n: number of target values to draw.
    :param rng: RNG for the sampling.
    :param active_frac: fraction of entities with a nonzero degree on this side
        (Block B ``subject_frac`` / ``object_frac``); ``1.0`` → no zeros (legacy).
    :returns: int64 array of length ``n`` summing to ``round(n · mean_deg)``, or ``None``.
    """
    if n <= 0 or not np.isfinite([p90, d_max]).all() or d_max < 1:
        return None
    if math.isnan(alpha) or alpha <= 1.0:
        alpha = _DEGSEQ_FALLBACK_ALPHA
    lo = max(float(p90), 1.0)
    hi = max(float(d_max), lo)

    af = 1.0 if not np.isfinite(active_frac) else min(1.0, max(0.0, float(active_frac)))
    n_active = max(1, int(round(af * n)))
    n_zero = n - n_active
    # The tail is the top 10% of *all* entities (p90/max are measured over all of them).
    # It is always drawn from the active pool, since active_frac ≥ 0.1 on any real graph.
    n_tail = max(1, min(n_active, int(round(n * _DEGSEQ_TAIL_FRACTION))))
    n_body = n_active - n_tail

    def _degrees(lo_t: float, hi_t: float, size: int, exp_t: float) -> np.ndarray:
        """Integer degrees from a power law(exp_t) truncated to [lo_t, hi_t]."""
        vals = sample_powerlaw_trunc(exp_t, lo_t, hi_t, size, rng)
        return np.round(vals).astype(np.int64)

    # Extreme-value-matched tail exponent: expected max of n_tail draws = hi.
    alpha_tail = 1.0 + math.log(n_tail) / math.log(hi / lo) if hi > lo and n_tail > 1 else alpha
    tail = _degrees(lo, hi, n_tail, alpha_tail)
    if hi > lo:
        tail[np.argmax(tail)] = int(round(hi))   # ensure the sampled max hits the target max

    # Body: same power law on its own [1, p90] range. Zeros: the non-subjects/non-objects.
    body = _degrees(1.0, lo, n_body, alpha) if n_body > 0 else np.array([], dtype=np.int64)
    zeros = np.zeros(n_zero, dtype=np.int64)
    seq = np.concatenate([zeros, body, tail])

    if np.isfinite(mean_deg):
        target = int(round(n * float(mean_deg)))
        is_body = np.zeros(seq.size, dtype=bool)
        is_body[n_zero:n_zero + n_body] = True
        # The active nodes must keep degree ≥1 — that is what makes ``active_frac`` mean
        # what it says. A heavy in-hub (swdf: an in-degree of 9148 over 7671 tail nodes)
        # otherwise makes the tail so heavy that repairing the sum down zeroes out much of
        # the body, producing 35% zeros where the signature says 0.2%. So repair the body
        # with a floor of 1, and only if the tail is *still* too heavy to fit the budget
        # let the trim reach the tail (bending max/p90 — the last thing to give, but ahead
        # of silently inventing non-objects).
        floor1 = is_body.astype(np.int64)
        seq = repair_degree_sum(seq, target, rng, floor=floor1, adjustable=is_body)
        if int(seq.sum()) != target:
            active = is_body.copy()
            active[n_zero + n_body:] = True                     # body + tail, never the zeros
            seq = repair_degree_sum(seq, target, rng, floor=active.astype(np.int64),
                                    adjustable=active)
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
