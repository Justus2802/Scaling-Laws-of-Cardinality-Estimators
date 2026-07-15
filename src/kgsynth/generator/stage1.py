"""Stage 1 — Schema sampler.

Builds the abstract schema (relations, types, type-relation probability table)
from a measured reduced-signature target: BlockA + BlockC for size/schema shape,
BlockB for edge multiplicity and degree shape, BlockD for characteristic-set reuse,
and BlockF for connectivity. All five blocks are mandatory (see
user_docs/generator.md §"Target signature must be complete") — a real graph always
measures them, so there is no degraded-mode path here.

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
    _reconstruct_singular_values,
    sample_degree_sequence,
)
from .._logging import get_logger
from .schema import Schema

log = get_logger(__name__)

# ── Tuning constants (Stage-1 schema) — adjust here ─────────────────────────────
DEFAULT_ZIPF_EXPONENT = 2.0        # fallback for relation- / CS-frequency Zipf exponents (small-R)
# Number of co-occurrence groups synthesised from the Block C M_subj / M_obj spectra.
# Fixed at the Block C measurement cap (10 SVs) rather than derived from spectral
# entropy: spectral entropy conflates group *count* with weight *uniformity* — a KG
# with 5 groups where one dominates gives k_eff ≈ 1.4 instead of 5.  The exp-decay
# weights already encode skewness; k just needs to be "large enough not to miss
# structure", which the measurement cap satisfies.  See user_docs/generator.md §"Co-occurrence groups".
COOC_NUM_GROUPS = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_frac(getter) -> float:
    """Read a signature fraction, defaulting to 1.0 when it is absent or unusable.

    A pre-``subject_frac`` signature (measured before this feature existed) has no such
    field; 1.0 reproduces the old "every entity is a subject/object" behaviour, so old
    corpora still generate rather than crash. A NaN or out-of-range value is treated the
    same way.
    """
    try:
        v = float(getter())
    except (KeyError, AttributeError, TypeError, ValueError, RuntimeError):
        return 1.0            # RuntimeError: a block that never had this field measured
    return v if np.isfinite(v) and 0.0 < v <= 1.0 else 1.0


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


def _relation_weights(n: int, logq, zipf_fallback: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Rebuild ``n`` relation frequency weights from Block B's log-share quantile fit.

    Reconstructs the *rank curve* directly: evaluate the stored quantile function of
    ``log(E_r / Σ E_r)`` at ``n`` evenly-spaced levels, exponentiate, renormalise. This
    replaces the old ``Zipf(exponent)`` weights, which lost on every corpus graph — the
    curves are frequently not Zipf-shaped, and the exponent could not be fitted at all
    below ~10 relations (see ``block_b.py``'s G1 note and the plan).

    The evaluation is deterministic, not an iid draw from the quantile function: with R
    small the rank curve *is* the signal, so sampling would only add variance to a
    quantity that has almost no degrees of freedom left. Weights are still shuffled, so
    relation *indices* carry no implicit rank ordering.

    :param n: number of relations.
    :param logq: Block B's ``rel_freq_logq`` quantile fit (all-NaN when unavailable).
    :param zipf_fallback: Zipf exponent used only when the fit is unavailable.
    :param rng: RNG for the index shuffle.
    :returns: float array of length ``n`` summing to 1.
    """
    if n == 0:
        return np.array([], dtype=float)
    qs = np.asarray(logq, dtype=float)
    if not np.isfinite(qs).all():
        return _zipf_weights(n, zipf_fallback, rng)
    weights = np.exp(np.interp(np.linspace(0.0, 1.0, n), QUANTILE_LEVELS, qs))
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _validate_target(a: BlockA, b: BlockB, c: BlockC, d: BlockD, f: BlockF) -> None:
    """Raise ``ValueError`` naming the first non-finite feature Stage 1/2 require.

    These quantities are measurable on any real graph (see user_docs/generator.md
    §"Target signature must be complete" for the evidence, gathered by
    enumerating every NaN across the 9-KG corpus). A NaN here means the target
    signature is incomplete or corrupted, not a legitimate small-KG edge case —
    so this fails loudly at the Stage-1 boundary instead of Stage 1/2 silently
    degrading three steps downstream. Features that *can* legitimately be NaN
    (small-R Zipf/CS fits, untyped-KG class stats, "no symmetric relation")
    are read directly by ``sample_schema`` / ``instantiate`` with their own
    documented fallback and are deliberately not checked here.
    """
    def _fin(name: str, value: float) -> None:
        if value != value:
            raise ValueError(f"Target signature is missing required feature: {name!r}")

    _fin("num_entities", float(a.num_entities))
    _fin("mean_degree", a.mean_degree)
    _fin("num_relations", float(a.num_relations))
    for i, q in enumerate(d.cs_size_q):
        _fin(f"cs_size_q[{i}]", q)
    for i, q in enumerate(d.inv_cs_size_q):
        _fin(f"inv_cs_size_q[{i}]", q)
    _fin("num_distinct_cs", float(d.num_distinct_cs))
    _fin("inv_num_distinct_cs", float(d.inv_num_distinct_cs))
    _fin("out_degree_fit.alpha", b.out_degree_fit.alpha)
    _fin("in_degree_fit.alpha", b.in_degree_fit.alpha)
    _fin("out_degree_p90", b.out_degree_p90)
    _fin("out_degree_max", float(b.out_degree_max))
    _fin("in_degree_p90", b.in_degree_p90)
    _fin("in_degree_max", float(b.in_degree_max))
    _fin("a_obj", b.a_obj)
    _fin("a_subj", b.a_subj)
    _fin("subj_cooc_exp.rate", c.subj_cooc_exp.rate)
    _fin("subj_cooc_exp.scale", c.subj_cooc_exp.scale)
    _fin("obj_cooc_exp.rate", c.obj_cooc_exp.rate)
    _fin("obj_cooc_exp.scale", c.obj_cooc_exp.scale)
    _fin("num_components", float(f.num_components))
    _fin("largest_component_fraction", f.largest_component_fraction)


def sample_schema(
    a: BlockA,
    c: BlockC,
    *,
    d: BlockD,
    b: BlockB,
    f: BlockF,
    relation_zipf_exponent: float = DEFAULT_ZIPF_EXPONENT,
    seed: int = 0,
) -> Schema:
    """Stage 1: derive an abstract schema from a target signature.

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
    d : BlockD
        Characteristic-set statistics. Stage 2 uses the ``cs_size_q`` quantiles
        and ``num_distinct_cs`` to build a realistic pool of reusable CS
        templates instead of sampling every entity independently — this fixes
        the co-occurrence density and num_distinct_cs deviations.
    b : BlockB
        Degree-structure statistics: relation-usage Zipf exponent, out/in
        degree fits, and per-relation multiplicity/reciprocity shape.
    f : BlockF
        Connectivity / path statistics. ``num_components`` and
        ``largest_component_fraction`` are forwarded to the Schema so Stage 2
        can leave the correct number of satellite components disconnected
        instead of fully connecting the graph.
    relation_zipf_exponent : float
        Zipf exponent for relation frequency weights.  Controls how skewed
        relation usage is; real KGs typically fall in [1.5, 2.5]. Only used
        when Block B's measured exponent is unavailable (small R).
    seed : int
        RNG seed; the same seed + inputs always produce the same schema.

    Returns
    -------
    Schema
        Abstract schema ready to be handed to Stage 2 (instantiate).

    Raises
    ------
    ValueError
        If the target signature is missing a feature that a real graph always
        measures — see :func:`_validate_target`.
    """
    _validate_target(a, b, c, d, f)
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
    # Rebuild the rank curve from Block B's log-share quantile function: evaluate it at
    # num_relations evenly-spaced levels, exponentiate, renormalise. Deterministic rather
    # than iid-sampled — with R small the rank curve *is* the signal, and iid draws would
    # add variance to a quantity with almost no degrees of freedom left. Falls back to a
    # Zipf only when the fit is unavailable (a graph with no relations at all).
    relations = [f"http://kgsynth.org/rel/{i}" for i in range(num_relations)]
    relation_weights = _relation_weights(
        num_relations, b.rel_freq_logq, relation_zipf_exponent, rng
    )

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
    # The reduced Block C exposes no measured P(r|t) dict, so synthesise it from
    # the P(r|t) type-relation spectrum — its *own* T×R singular spectrum, not the
    # R×R co-occurrence spectrum the generator used to conflate it with.
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
    # Stage 2 draws entity CSes from these prototypes and assigns types post-hoc.
    # _validate_target guarantees both fits are finite and num_relations ≥ 1, so
    # _reconstruct_singular_values always returns COOC_NUM_GROUPS values here.
    def _build_group_probs(cooc_exp):
        """Build (probs, weights) group prototypes from an exp-decay cooc fit."""
        svs = _reconstruct_singular_values(cooc_exp, k=COOC_NUM_GROUPS)
        probs = _sample_type_relation_probs(len(svs), num_relations, relation_weights, svs, rng)
        return probs, svs / svs.sum()

    subj_group_probs, subj_group_weights = _build_group_probs(c.subj_cooc_exp)
    obj_group_probs,  obj_group_weights  = _build_group_probs(c.obj_cooc_exp)
    log.info(
        "Stage 1: cooc groups — subj k=%d, obj k=%d",
        len(subj_group_weights), len(obj_group_weights),
    )

    # --- CS structure from Block D ---
    cs_num_templates = int(d.num_distinct_cs)
    cs_template_zipf = (
        float(d.cs_freq_fit.alpha)
        if not math.isnan(d.cs_freq_fit.alpha) else DEFAULT_ZIPF_EXPONENT
    )
    # Support of the Stage-2 reuse draw: cs_freq's α is a truncated MLE over
    # [v_min, v_max], so Stage 2 draws from that same bounded law.
    cs_template_vmin = float(d.cs_freq_fit.v_min)
    cs_template_vmax = float(d.cs_freq_fit.v_max)

    # --- Degree targets from Block B ---
    # One target degree per entity, sampled purely from signature-vector components
    # — the degree power-law α, the p90/max degree scalars and the mean degree —
    # never Block B's raw retained arrays.  Stage 2 steers wiring toward these, so
    # the whole distribution (body, p90, max) is targeted rather than a single cap.
    #
    # The mean handed over is the *content* mean, not Block A's E/V.  Block B measures
    # entity content degrees (rdf:type edges and class nodes excluded), and Stage 2
    # wires rdf:type edges outside the content budget, so E/V would describe a
    # different population than the fits do — an over-budget on both sides, and on the
    # in-side (where entities never receive a type edge at all) it has no counterpart
    # correction downstream.  Both sides now sum to exactly content_E, which is what
    # a directed wiring needs (Σ out = Σ in = E).
    # subject_frac / object_frac: the share of entities that emit / receive a content edge.
    # Not every entity is a subject (swdf: 30%), and spreading the budget over all of them
    # flattens the degree distribution and inflates Σ|CS| past the edge budget (see
    # sample_degree_sequence). Stale signatures without the fields → 1.0 (legacy, no zeros).
    n_ent = a.num_entities
    content_E = num_triples * (1.0 - float(a.type_edge_frac))
    mean_deg = content_E / n_ent if n_ent > 0 else float("nan")
    subject_frac = _safe_frac(lambda: b.subject_frac)
    object_frac = _safe_frac(lambda: b.object_frac)
    target_out_degrees = sample_degree_sequence(
        b.out_degree_fit.alpha, b.out_degree_p90, b.out_degree_max, mean_deg, n_ent, rng,
        active_frac=subject_frac)
    target_in_degrees = sample_degree_sequence(
        b.in_degree_fit.alpha, b.in_degree_p90, b.in_degree_max, mean_deg, n_ent, rng,
        active_frac=object_frac)

    # --- Per-relation multiplicity shape (G2) + CS-size offset (G2b) + CS-size shape ---
    # Stored as plain quantile tuples; Stage 2 samples a per-relation α from obj_alpha_q
    # and applies the cs_size^a_obj offset. NaN fits → neutral fallback in Stage 2
    # (a legitimate small-R outcome — obj_alpha_q/subj_alpha_q are per-relation fits).
    obj_alpha_q = tuple(b.obj_alpha_q)
    subj_alpha_q = tuple(b.subj_alpha_q)
    # Upper bound of each multiplicity law — Stage 2 draws the per-relation tail on
    # [1, max] rather than unbounded (α was fitted over exactly that range).
    obj_mult_max = float(b.obj_mult_max)
    subj_mult_max = float(b.subj_mult_max)
    a_obj = float(b.a_obj)
    a_subj = float(b.a_subj)
    cs_size_q = tuple(d.cs_size_q)
    # Inverse CS (object side), symmetric to forward CS structure (Block D).
    inv_cs_size_q = tuple(d.inv_cs_size_q)
    inv_cs_num_templates = int(d.inv_num_distinct_cs)
    inv_cs_template_zipf = (
        float(d.inv_cs_freq_fit.alpha)
        if not math.isnan(d.inv_cs_freq_fit.alpha) else DEFAULT_ZIPF_EXPONENT
    )
    inv_cs_template_vmin = float(d.inv_cs_freq_fit.v_min)
    inv_cs_template_vmax = float(d.inv_cs_freq_fit.v_max)

    log.info(
        "Stage 1: schema ready — "
        "degree targets out(max=%d, p90=%.1f) in(max=%d, p90=%.1f), "
        "cs_num_templates=%d, a_obj=%.3f, obj_alpha_qmin=%.3f",
        int(target_out_degrees.max()), np.percentile(target_out_degrees, 90),
        int(target_in_degrees.max()), np.percentile(target_in_degrees, 90),
        cs_num_templates, a_obj, obj_alpha_q[0],
    )
    target_num_components = int(f.num_components)
    target_lcc = float(f.largest_component_fraction)

    # Block C pair-level edge multiplicity (overlap) targets. A current-format
    # Block C always carries these; read them directly so a missing value raises
    # loudly rather than being masked. A NaN value is a real measurement outcome
    # (a graph with zero content/undirected edges — see BlockC.calculate) and
    # clamps to 1.0, the neutral near-simple target, as does any value < 1.0.
    def _clamp_ratio(v: float) -> float:
        return v if (v == v and v >= 1.0) else 1.0
    edge_multiplicity = _clamp_ratio(float(c.edge_multiplicity))
    bidirectional_ratio = _clamp_ratio(float(c.bidirectional_ratio))

    # Per-relation reciprocity (Block B): assign each synthetic relation symmetric
    # (~recip_symmetric_value) or asymmetric (0) by a Bernoulli draw on
    # frac_symmetric[bin], where `bin` is the relation's OWN cumulative edge-fraction
    # rank under `relation_weights` — i.e. a frequency-rank lookup, not an
    # independent marginal draw. This preserves the frequency↔reciprocity pairing
    # (which relation is symmetric matters, not just how many are): assigning
    # reciprocity independently of frequency was found to put it on the wrong
    # relations (e.g. the biggest relation getting ρ=0 despite being symmetric in the
    # original) — see developer_docs/notes/relation_reciprocity_and_bidirectionality.md.
    # An empty bin (no relations landed there when Block B was measured — common
    # when R is small, e.g. only 5 relations over 6 fixed bins) is a data gap, not
    # evidence the bin is asymmetric: falling back to 0 there silently overrides
    # graphs that are symmetric almost everywhere (aids: every relation reciprocity
    # 1.0, yet naive zero-fill would still assign 0 to a relation whose bin happens
    # to be empty). So an empty bin borrows the value of its nearest non-empty bin
    # (ties broken toward the higher-frequency side) rather than defaulting to 0.
    # All-NaN frac_symmetric (no bin has any data at all) leaves every relation
    # asymmetric — this can only happen with R=0, which num_relations =
    # max(1, a.num_relations) already excludes.
    relation_reciprocity = None
    frac_symmetric = np.asarray(b.recip_symmetric_frac, dtype=float)
    symmetric_value = float(b.recip_symmetric_value)
    if np.isfinite(frac_symmetric).any():
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
        recip_ordered = np.where(
            is_sym,
            symmetric_value if np.isfinite(symmetric_value) else 0.9,
            0.0,
        )
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
        type_edge_frac=float(a.type_edge_frac),
        edge_multiplicity=edge_multiplicity,
        bidirectional_ratio=bidirectional_ratio,
        relation_reciprocity=relation_reciprocity,
        cs_num_templates=cs_num_templates,
        cs_template_zipf=cs_template_zipf,
        cs_template_vmin=cs_template_vmin,
        cs_template_vmax=cs_template_vmax,
        target_out_degrees=target_out_degrees,
        target_in_degrees=target_in_degrees,
        obj_alpha_q=obj_alpha_q,
        a_obj=a_obj,
        subj_alpha_q=subj_alpha_q,
        a_subj=a_subj,
        obj_mult_max=obj_mult_max,
        subj_mult_max=subj_mult_max,
        cs_size_q=cs_size_q,
        inv_cs_size_q=inv_cs_size_q,
        inv_cs_num_templates=inv_cs_num_templates,
        inv_cs_template_zipf=inv_cs_template_zipf,
        inv_cs_template_vmin=inv_cs_template_vmin,
        inv_cs_template_vmax=inv_cs_template_vmax,
        subj_group_probs=subj_group_probs,
        subj_group_weights=subj_group_weights,
        obj_group_probs=obj_group_probs,
        obj_group_weights=obj_group_weights,
        target_num_components=target_num_components,
        target_lcc=target_lcc,
    )
