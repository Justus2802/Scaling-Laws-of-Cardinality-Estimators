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

from ..signature import BlockA, BlockB, BlockC, BlockD, BlockF, QUANTILE_LEVELS

from ._adapters import (
    _functionality_from_alpha,
    _reconstruct_singular_values,
    _quantile_mean,
    sample_degree_sequence,
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
        # Truncate Stage-2 reuse draws at the measured max recurrence: the fitted
        # α covers the full bounded range, so unbounded draws would over-skew.
        cs_template_vmax = float(d.cs_freq_fit.v_max)
    else:
        cs_size_mean = 0.0   # signal instantiate to derive from E/V budget
        cs_num_templates = 0
        cs_template_zipf = DEFAULT_ZIPF_EXPONENT
        cs_template_vmax = float("nan")

    # --- Edge multiplicity, degree targets, inverse functionality from Block B ---
    if b is not None:
        mean_functionality = _functionality_from_alpha(b.obj_alpha_q, floor=FUNCTIONALITY_FLOOR)
    else:
        mean_functionality = 1.0

    if b is not None:
        # Target degree sequences (replace the old extreme-value max-degree caps):
        # one target degree per entity, sampled purely from signature-vector
        # components — the degree power-law α, the p90/max degree scalars and the
        # mean degree — never Block B's raw retained arrays.  Stage 2 steers
        # wiring toward these, so the whole distribution (body, p90, max) is
        # targeted rather than a single hard cap.  NaN p90/max (old signatures)
        # → None → no degree steering in Stage 2.
        n_ent = a.num_entities
        mean_deg = num_triples / n_ent if n_ent > 0 else float("nan")

        def _safe(fn) -> float:
            """Value of a Block B property, NaN when absent from stale data."""
            try:
                return float(fn())
            except RuntimeError:
                return float("nan")

        target_out_degrees = sample_degree_sequence(
            b.out_degree_fit.alpha, _safe(lambda: b.out_degree_p90),
            _safe(lambda: b.out_degree_max), mean_deg, n_ent, rng)
        target_in_degrees = sample_degree_sequence(
            b.in_degree_fit.alpha, _safe(lambda: b.in_degree_p90),
            _safe(lambda: b.in_degree_max), mean_deg, n_ent, rng)
    else:
        target_out_degrees = None
        target_in_degrees = None

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
    inv_cs_template_vmax = float(d.inv_cs_freq_fit.v_max) if d is not None else float("nan")

    log.info(
        "Stage 1: schema ready — mean_functionality=%.3f, "
        "degree targets out(max=%s, p90=%s) in(max=%s, p90=%s), "
        "cs_num_templates=%d, a_obj=%.3f, obj_alpha_qmin=%.3f",
        mean_functionality,
        int(target_out_degrees.max()) if target_out_degrees is not None else "—",
        int(np.percentile(target_out_degrees, 90)) if target_out_degrees is not None else "—",
        int(target_in_degrees.max()) if target_in_degrees is not None else "—",
        int(np.percentile(target_in_degrees, 90)) if target_in_degrees is not None else "—",
        cs_num_templates, a_obj, obj_alpha_q[0],
    )
    target_num_components = int(f.num_components) if f is not None else DEFAULT_NUM_COMPONENTS
    target_lcc = float(f.largest_component_fraction) if f is not None else DEFAULT_LCC

    # Block C pair-level edge multiplicity (overlap) targets; NaN / not-measured
    # (stale signatures) → 1.0, the neutral legacy near-simple graph.
    def _ratio(getter) -> float:
        try:
            v = float(getter())
        except Exception:  # noqa: BLE001 — NotCalculated on stale caches → neutral
            return 1.0
        return v if (v == v and v >= 1.0) else 1.0
    edge_multiplicity = _ratio(lambda: c.edge_multiplicity)
    bidirectional_ratio = _ratio(lambda: c.bidirectional_ratio)

    # Per-relation reciprocity (Block B): assign each synthetic relation symmetric
    # (~recip_symmetric_value) or asymmetric (0) by a Bernoulli draw on
    # frac_symmetric[bin], where `bin` is the relation's OWN cumulative edge-fraction
    # rank under `relation_weights` — i.e. a frequency-rank lookup, not an
    # independent marginal draw. This preserves the frequency↔reciprocity pairing
    # (which relation is symmetric matters, not just how many are): assigning
    # reciprocity independently of frequency was found to put it on the wrong
    # relations (e.g. the biggest relation getting ρ=0 despite being symmetric in the
    # original) — see docs/notes/relation_reciprocity_and_bidirectionality.md.
    # An empty bin (no relations landed there when Block B was measured — common
    # when R is small, e.g. only 5 relations over 6 fixed bins) is a data gap, not
    # evidence the bin is asymmetric: falling back to 0 there silently overrides
    # graphs that are symmetric almost everywhere (aids: every relation reciprocity
    # 1.0, yet naive zero-fill would still assign 0 to a relation whose bin happens
    # to be empty). So an empty bin borrows the value of its nearest non-empty bin
    # (ties broken toward the higher-frequency side) rather than defaulting to 0.
    relation_reciprocity = None
    if b is not None:
        try:
            frac_symmetric = np.asarray(b.recip_symmetric_frac, dtype=float)
            symmetric_value = float(b.recip_symmetric_value)
        except Exception:  # noqa: BLE001 — NotCalculated on stale caches → asymmetric
            frac_symmetric = None
        if frac_symmetric is not None and np.isfinite(frac_symmetric).any():
            n_bins = frac_symmetric.size
            valid = np.where(np.isfinite(frac_symmetric))[0]
            filled = frac_symmetric.copy()
            for i in range(n_bins):
                if not np.isfinite(filled[i]):
                    nearest = valid[np.argmin(np.abs(valid - i))]
                    filled[i] = frac_symmetric[nearest]

            order = np.argsort(-relation_weights)               # frequency-rank order
            cum_frac = np.cumsum(relation_weights[order])
            bin_edges = np.asarray(QUANTILE_LEVELS[1:], dtype=float)  # fixed thresholds
            bin_idx = np.clip(np.searchsorted(bin_edges, cum_frac, side="left"),
                              0, n_bins - 1)
            p_sym = filled[bin_idx]
            is_sym = rng.random(num_relations) < p_sym
            recip_ordered = np.where(is_sym, symmetric_value if np.isfinite(symmetric_value) else 0.9, 0.0)
            relation_reciprocity = np.empty(num_relations, dtype=float)
            relation_reciprocity[order] = recip_ordered

    return Schema(
        relations=relations,
        relation_weights=relation_weights,
        types=types,
        type_weights=type_weights,
        type_relation_probs=type_relation_probs,
        num_entities=a.num_entities,
        num_triples=num_triples,
        edge_multiplicity=edge_multiplicity,
        bidirectional_ratio=bidirectional_ratio,
        relation_reciprocity=relation_reciprocity,
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
    )
