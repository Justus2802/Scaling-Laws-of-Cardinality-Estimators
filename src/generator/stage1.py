"""Stage 1 — Schema sampler.

Builds the abstract schema (relations, types, type-relation probability table)
from a measured reduced-signature BlockA + BlockC target, optionally refined by
BlockB (edge multiplicity, degree shape) and BlockD (characteristic-set reuse).

Design decisions:
  - Relation frequency weights are sampled from a Zipf distribution whose
    exponent is a tunable parameter; the spec requires it but it is not
    directly available from Blocks A or C (Block B would supply it).
  - The type-relation probability table P(r|t) is constructed via a low-rank
    random factorisation whose singular values match the Block C target, so
    the co-occurrence structure of the generated schema resembles the real KG.
  - All randomness goes through a single np.random.Generator seeded at call
    time, making every output fully reproducible.
"""

import math

import numpy as np

from signature import BlockA, BlockB, BlockC, BlockD, BlockF

from ._adapters import (
    _functionality_from_alpha,
    _reconstruct_singular_values,
    _quantile_mean,
)
from ._logging import get_logger
from .schema import Schema, _NAN_Q

log = get_logger(__name__)

# ── Tuning constants (Stage-1 schema) — adjust here ─────────────────────────────
DEFAULT_ZIPF_EXPONENT = 2.0        # fallback for relation- / CS-frequency Zipf exponents
FUNCTIONALITY_FLOOR = 0.1          # clamp floor for mean_functionality (out-side)
# Connectivity fallbacks when Block F is absent (fully-connected behaviour).
DEFAULT_NUM_COMPONENTS = 1
DEFAULT_LCC = 1.0
# Number of co-occurrence groups synthesised from the Block C M_subj / M_obj spectra.
# Fixed at the Block C measurement cap (10 SVs) rather than derived from spectral
# entropy: spectral entropy conflates group *count* with weight *uniformity* — a KG
# with 5 groups where one dominates gives k_eff ≈ 1.4 instead of 5.  The exp-decay
# weights already encode skewness; k just needs to be "large enough not to miss
# structure", which the measurement cap satisfies.  See docs/generator.md §"Co-occurrence groups".
COOC_NUM_GROUPS = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zipf_weights(n: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n normalized frequency weights from a Zipf distribution.

    Weights are shuffled so that relation indices carry no implicit rank
    ordering — the Zipf shape is preserved in the *distribution* of weights,
    not in their index order.
    """
    if n == 0:
        return np.array([], dtype=float)
    ranks = np.arange(1, n + 1, dtype=float)
    weights = ranks ** (-exponent)
    weights /= weights.sum()
    rng.shuffle(weights)
    return weights


def _sample_type_relation_probs(
    num_types: int,
    num_relations: int,
    relation_weights: np.ndarray,
    target_singular_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a P(r|t) matrix whose singular spectrum matches the Block C target.

    Construction (low-rank random factorisation):
      1. Determine rank k = number of nonzero target singular values,
         capped by min(|T|, |R|).
      2. Draw random orthonormal U (|T|×k) and V (|R|×k) via QR.
      3. Form logits = U @ diag(sigma_normalised) @ V^T.
      4. Multiply each row element-wise by the global relation_weights so
         frequently-used relations are more likely to appear across all types.
      5. Row-normalise with softmax to produce valid probability rows.

    Falls back to tiling relation_weights uniformly across types when the
    singular value information is insufficient for a low-rank construction.
    """
    if num_types == 0 or num_relations == 0:
        return np.zeros((num_types, num_relations), dtype=float)

    nonzero_svs = target_singular_values[target_singular_values > 0]
    rank = min(len(nonzero_svs), num_types, num_relations)

    if rank == 0:
        # No co-occurrence signal: every type gets the same global relation weights
        return np.tile(relation_weights, (num_types, 1))

    # Random orthonormal factors
    U = np.linalg.qr(rng.standard_normal((num_types, rank)))[0]      # (T, k)
    V = np.linalg.qr(rng.standard_normal((num_relations, rank)))[0]  # (R, k)
    sigma = nonzero_svs[:rank] / nonzero_svs[:rank].sum()            # normalised

    logits = U @ np.diag(sigma) @ V.T   # (T, R)

    # Shift for numerical stability before exponentiation
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)

    # Bias each row toward globally frequent relations
    P *= relation_weights[np.newaxis, :]

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P /= row_sums
    return P


def _build_type_rel_probs_from_measured(
    type_relation_conditional: dict[str, dict[str, float]],
    num_types: int,
    num_relations: int,
    relation_weights: np.ndarray,
) -> np.ndarray:
    """Build P(r|t) from measured type_relation_conditional via rank mapping.

    Sorts real types by activity (descending) and real relations by aggregate
    frequency (descending), then maps them positionally onto schema indices.
    This preserves co-occurrence structure without requiring URI alignment.

    The reduced Block C does not expose a measured ``type_relation_conditional``
    dict, so for reduced-signature inputs this path is unused and Stage 1 always
    falls back to the low-rank synthesis above.
    """
    if not type_relation_conditional or num_types == 0 or num_relations == 0:
        return np.tile(relation_weights, (num_types, 1))

    real_types = sorted(
        type_relation_conditional,
        key=lambda t: sum(type_relation_conditional[t].values()),
        reverse=True,
    )
    rel_agg: dict[str, float] = {}
    for probs in type_relation_conditional.values():
        for rel, p in probs.items():
            rel_agg[rel] = rel_agg.get(rel, 0.0) + p
    real_rels = sorted(rel_agg, key=lambda r: rel_agg[r], reverse=True)

    P = np.zeros((num_types, num_relations), dtype=float)
    for t_idx in range(num_types):
        real_t = real_types[t_idx % len(real_types)]
        src = type_relation_conditional[real_t]
        for r_idx in range(num_relations):
            real_r = real_rels[r_idx % len(real_rels)]
            P[t_idx, r_idx] = src.get(real_r, 0.0)

    row_sums = P.sum(axis=1, keepdims=True)
    zero_mask = (row_sums.ravel() == 0)
    P[zero_mask] = relation_weights
    row_sums[row_sums == 0] = 1.0
    P /= row_sums
    return P


def _sample_degree_sequence(
    rng: np.random.Generator,
    n: int,
    total_edges: int,
    alpha: float,
    p90: float,
    d_max: int,
) -> "np.ndarray | None":
    """Sample a per-entity target degree sequence from Block B power-law scalars.

    Top 10% of nodes draw from a truncated power law in [p90, d_max]; the
    remaining 90% draw from a Poisson body whose mean is calibrated so the
    overall mean matches total_edges / n.  Returns None when the fit scalars
    are unusable (NaN, non-positive, or n == 0).
    """
    if n <= 0 or total_edges <= 0:
        return None
    try:
        alpha_f = float(alpha)
        p90_f = float(p90)
        dmax_i = int(d_max)
    except (TypeError, ValueError):
        return None
    if math.isnan(alpha_f) or math.isnan(p90_f) or alpha_f <= 1.0 or p90_f <= 0 or dmax_i <= 0:
        return None

    mean_deg = total_edges / n
    n_tail = max(1, int(round(0.1 * n)))
    n_body = n - n_tail

    # Tail: inverse-CDF sampling from truncated power law on [p90_f, dmax_i]
    lo, hi = max(1.0, p90_f), float(dmax_i)
    if hi <= lo:
        hi = lo + 1.0
    u = rng.random(n_tail)
    exp = 1.0 - alpha_f
    tail_samples = (lo ** exp + u * (hi ** exp - lo ** exp)) ** (1.0 / exp)
    tail_samples = np.clip(np.round(tail_samples), lo, hi).astype(np.int64)

    # Body: Poisson with mean calibrated so E[total] ≈ total_edges
    tail_sum = float(tail_samples.sum())
    body_mean = max(1.0, (total_edges - tail_sum) / max(n_body, 1))
    body_mean = min(body_mean, p90_f)   # body shouldn't exceed the tail threshold
    body_samples = rng.poisson(body_mean, size=n_body).astype(np.int64)
    body_samples = np.clip(body_samples, 1, int(p90_f))

    seq = np.concatenate([tail_samples, body_samples])
    rng.shuffle(seq)
    return seq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_schema(
    a: BlockA,
    c: BlockC,
    *,
    d: BlockD = None,
    b: BlockB = None,
    f: BlockF = None,
    relation_zipf_exponent: float = DEFAULT_ZIPF_EXPONENT,
    seed: int = 0,
) -> Schema:
    """Stage 1: derive an abstract schema from a target BlockA + BlockC.

    Parameters
    ----------
    a : BlockA
        Measured (reduced) size/density signature of the target KG.
        |V|, mean degree (→ |E|) and |R| are used directly.
    c : BlockC
        Measured (reduced) schema/correlation signature of the target KG.
        ``num_classes``, ``class_size_fit.alpha`` and the P(r|t) type-relation
        exp-decay spectrum (``type_rel_spectrum_exp``) guide the type structure
        and the low-rank P(r|t) reconstruction.
    d : BlockD, optional
        Characteristic-set statistics.  When provided, Stage 2 will use the mean
        of ``cs_size_q`` and ``num_distinct_cs`` to build a realistic pool of
        reusable CS templates instead of sampling every entity independently.
        This fixes the co-occurrence density and num_distinct_cs deviations.
    b : BlockB, optional
        Degree-structure statistics.  When provided, the mean relation
        functionality is derived from the object-multiplicity α quantiles to
        sample more than one object per (s,p) pair, matching the target's edge
        multiplicity.
    f : BlockF, optional
        Connectivity / path statistics.  When provided, the measured
        ``num_components`` and ``largest_component_fraction`` are forwarded to
        the Schema so Stage 2 can leave the correct number of satellite
        components disconnected instead of fully connecting the graph.
    relation_zipf_exponent : float
        Zipf exponent for relation frequency weights.  Controls how skewed
        relation usage is; real KGs typically fall in [1.5, 2.5].
    seed : int
        RNG seed; the same seed + inputs always produce the same schema.

    Returns
    -------
    Schema
        Abstract schema ready to be handed to Stage 2 (instantiate).
    """
    rng = np.random.default_rng(seed)

    num_relations = max(1, a.num_relations)
    num_types = max(0, c.num_classes)
    # Reduced Block A stores mean degree (E/V) rather than |E|; recover the
    # edge budget as round(V × mean_degree).
    num_triples = int(round(a.num_entities * a.mean_degree))
    log.info(
        "Stage 1: sampling schema (seed=%d) for V=%d, R=%d, T=%d, target E=%d",
        seed, a.num_entities, num_relations, num_types, num_triples,
    )

    # --- Relations ---
    # Prefer the measured relation-usage Zipf exponent (Block B) over the
    # hard-coded parameter default, matching the brief's "Zipf(s)" with s = target.
    rel_zipf = relation_zipf_exponent
    if b is not None:
        measured = b.relation_zipf.exponent
        if not math.isnan(measured) and measured > 0:
            rel_zipf = float(measured)
            log.info("Stage 1: using measured relation Zipf exponent %.3f", rel_zipf)
    relations = [f"http://kgsynth.org/rel/{i}" for i in range(num_relations)]
    relation_weights = _zipf_weights(num_relations, rel_zipf, rng)

    # --- Types ---
    types = [f"http://kgsynth.org/type/{i}" for i in range(num_types)]

    if num_types > 0:
        type_zipf = c.class_size_fit.alpha
        if not np.isnan(type_zipf) and type_zipf > 0:
            type_weights = _zipf_weights(num_types, type_zipf, rng)
        else:
            # Block C could not fit a power-law (too few classes): fall back to uniform
            log.info("Stage 1: class-size α unavailable — using uniform type weights")
            type_weights = np.full(num_types, 1.0 / num_types)
    else:
        type_weights = np.array([], dtype=float)

    # --- Type-relation probability table ---
    # Use measured P(r|t) directly when available (full-block inputs); reduced
    # Block C provides none, so synthesise from the P(r|t) type-relation
    # spectrum — its *own* T×R singular spectrum, not the R×R co-occurrence
    # spectrum the generator used to conflate it with.
    trc = getattr(c, "type_relation_conditional", None) or {}
    if trc:
        type_relation_probs = _build_type_rel_probs_from_measured(
            trc, num_types, num_relations, relation_weights,
        )
    else:
        target_svs = _reconstruct_singular_values(c.type_rel_spectrum_exp)
        if num_types > 0 and target_svs.size == 0:
            log.info("Stage 1: no P(r|t) spectrum — uniform per-type relation weights")
        type_relation_probs = _sample_type_relation_probs(
            num_types, num_relations, relation_weights, target_svs, rng,
        )

    # --- Co-occurrence group prototypes from Block C spectra ---
    # subj_cooc_exp / obj_cooc_exp are the exp-decay fits of the V-normalised
    # singular spectra of the R×R entity co-occurrence matrices.  We reconstruct
    # COOC_NUM_GROUPS singular values and build one group-prototype P(r|group) row
    # per group using the same low-rank random factorisation as _sample_type_relation_probs.
    # Stage 2 will draw entity CSes from these prototypes and assign types post-hoc.
    def _build_group_probs(cooc_exp):
        """Build (probs, weights) group prototypes from an exp-decay cooc fit, or (None, None)."""
        svs = _reconstruct_singular_values(cooc_exp, k=COOC_NUM_GROUPS)
        if svs.size == 0 or num_relations == 0:
            return None, None
        probs = _sample_type_relation_probs(len(svs), num_relations, relation_weights, svs, rng)
        return probs, svs / svs.sum()

    subj_group_probs, subj_group_weights = _build_group_probs(c.subj_cooc_exp)
    obj_group_probs,  obj_group_weights  = _build_group_probs(c.obj_cooc_exp)
    log.info(
        "Stage 1: cooc groups — subj %s (%s), obj %s (%s)",
        "built" if subj_group_probs is not None else "unavailable",
        f"k={len(subj_group_weights)}" if subj_group_weights is not None else "NaN fit",
        "built" if obj_group_probs is not None else "unavailable",
        f"k={len(obj_group_weights)}" if obj_group_weights is not None else "NaN fit",
    )

    # --- CS structure from Block D ---
    cs_size_mean_val = _quantile_mean(d.cs_size_q) if d is not None else float("nan")
    if d is not None and not math.isnan(cs_size_mean_val) and cs_size_mean_val > 0:
        cs_size_mean = float(cs_size_mean_val)
        cs_num_templates = max(1, int(d.num_distinct_cs))
        cs_template_zipf = (
            float(d.cs_freq_fit.alpha)
            if not math.isnan(d.cs_freq_fit.alpha) else DEFAULT_ZIPF_EXPONENT
        )
    else:
        cs_size_mean = 0.0   # signal instantiate to derive from E/V budget
        cs_num_templates = 0
        cs_template_zipf = DEFAULT_ZIPF_EXPONENT

    # --- Edge multiplicity and target degree sequences from Block B ---
    if b is not None:
        mean_functionality = _functionality_from_alpha(b.obj_alpha_q, floor=FUNCTIONALITY_FLOOR)
    else:
        mean_functionality = 1.0

    # Sample per-entity target degree sequences from the Block B power-law fits.
    # The top 10% of nodes draw from a truncated power law pinned to [p90, max];
    # the remaining 90% draw from a Poisson body whose mean is solved so the
    # overall mean matches E/V.  None → Stage 2 falls back to PA-free wiring.
    target_out_degrees: "np.ndarray | None" = None
    target_in_degrees: "np.ndarray | None" = None
    if b is not None:
        n_ent = a.num_entities
        target_out_degrees = _sample_degree_sequence(
            rng, n_ent, num_triples,
            b.out_degree_fit.alpha, b.out_degree_p90, b.out_degree_max,
        )
        target_in_degrees = _sample_degree_sequence(
            rng, n_ent, num_triples,
            b.in_degree_fit.alpha, b.in_degree_p90, b.in_degree_max,
        )

    # --- Per-relation multiplicity shape (G2) + CS-size offset (G2b) + CS-size shape ---
    # Stored as plain quantile tuples; Stage 2 samples a per-relation α from obj_alpha_q
    # and applies the cs_size^a_obj offset. NaN fits → neutral fallback in Stage 2.
    obj_alpha_q = tuple(b.obj_alpha_q) if b is not None else _NAN_Q
    subj_alpha_q = tuple(b.subj_alpha_q) if b is not None else _NAN_Q
    a_obj = float(b.a_obj) if (b is not None and not math.isnan(b.a_obj)) else 0.0
    a_subj = float(b.a_subj) if (b is not None and not math.isnan(b.a_subj)) else 0.0
    cs_size_q = tuple(d.cs_size_q) if d is not None else _NAN_Q
    # Inverse CS (object side), symmetric to forward CS structure (Block D).
    inv_cs_size_q = tuple(d.inv_cs_size_q) if d is not None else _NAN_Q
    inv_cs_num_templates = (
        max(1, int(d.inv_num_distinct_cs)) if (d is not None and d.inv_num_distinct_cs > 0) else 0
    )
    inv_cs_template_zipf = (
        float(d.inv_cs_freq_fit.alpha)
        if (d is not None and not math.isnan(d.inv_cs_freq_fit.alpha)) else DEFAULT_ZIPF_EXPONENT
    )
    # CS-frequency reuse truncation (Block D truncated power-law v_max).
    cs_template_vmax = (
        float(d.cs_freq_fit.v_max)
        if (d is not None and not math.isnan(d.cs_freq_fit.v_max)) else float("nan")
    )
    inv_cs_template_vmax = (
        float(d.inv_cs_freq_fit.v_max)
        if (d is not None and not math.isnan(d.inv_cs_freq_fit.v_max)) else float("nan")
    )

    log.info(
        "Stage 1: schema ready — mean_functionality=%.3f, cs_num_templates=%d, "
        "a_obj=%.3f, obj_alpha_qmin=%.3f, target_degrees=%s",
        mean_functionality, cs_num_templates, a_obj, obj_alpha_q[0],
        "sampled" if target_out_degrees is not None else "None",
    )
    target_num_components = int(f.num_components) if f is not None else DEFAULT_NUM_COMPONENTS
    target_lcc = float(f.largest_component_fraction) if f is not None else DEFAULT_LCC

    # --- Path-length targets from Block F (max, mean, var) ---
    path_mean_target = float("nan")
    path_hi_target = 0
    if f is not None:
        path_mean_target = f.shortest_path_mean   # NaN when paths not sampled
        max_val = f.shortest_path_max
        if not math.isnan(max_val) and max_val > 0:
            path_hi_target = int(max_val)

    return Schema(
        relations=relations,
        relation_weights=relation_weights,
        types=types,
        type_weights=type_weights,
        type_relation_probs=type_relation_probs,
        num_entities=a.num_entities,
        num_triples=num_triples,
        cs_size_mean=cs_size_mean,
        cs_num_templates=cs_num_templates,
        cs_template_zipf=cs_template_zipf,
        cs_template_vmax=cs_template_vmax,
        mean_functionality=mean_functionality,
        target_out_degrees=target_out_degrees,
        target_in_degrees=target_in_degrees,
        obj_alpha_q=obj_alpha_q,
        a_obj=a_obj,
        subj_alpha_q=subj_alpha_q,
        a_subj=a_subj,
        cs_size_q=cs_size_q,
        inv_cs_size_q=inv_cs_size_q,
        inv_cs_num_templates=inv_cs_num_templates,
        inv_cs_template_zipf=inv_cs_template_zipf,
        inv_cs_template_vmax=inv_cs_template_vmax,
        subj_group_probs=subj_group_probs,
        subj_group_weights=subj_group_weights,
        obj_group_probs=obj_group_probs,
        obj_group_weights=obj_group_weights,
        target_num_components=target_num_components,
        target_lcc=target_lcc,
        path_mean_target=path_mean_target,
        path_hi_target=path_hi_target,
    )
