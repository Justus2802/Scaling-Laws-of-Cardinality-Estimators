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

from signature_reduced import BlockA, BlockB, BlockC, BlockD

from ._adapters import (
    _functionality_from_alpha,
    _reconstruct_singular_values,
    _skewnorm_mean,
)
from .schema import Schema


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
    relation_zipf_exponent: float = 2.0,
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
        of ``cs_size_skew`` and ``num_distinct_cs`` to build a realistic pool of
        reusable CS templates instead of sampling every entity independently.
        This fixes the co-occurrence density and num_distinct_cs deviations.
    b : BlockB, optional
        Degree-structure statistics.  When provided, the mean relation
        functionality is derived from the object-multiplicity α skew-normal to
        sample more than one object per (s,p) pair, matching the target's edge
        multiplicity.
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

    # --- Relations ---
    relations = [f"http://kgsynth.org/rel/{i}" for i in range(num_relations)]
    relation_weights = _zipf_weights(num_relations, relation_zipf_exponent, rng)

    # --- Types ---
    types = [f"http://kgsynth.org/type/{i}" for i in range(num_types)]

    if num_types > 0:
        type_zipf = c.class_size_fit.alpha
        if not np.isnan(type_zipf) and type_zipf > 0:
            type_weights = _zipf_weights(num_types, type_zipf, rng)
        else:
            # Block C could not fit a power-law (too few classes): fall back to uniform
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
        type_relation_probs = _sample_type_relation_probs(
            num_types, num_relations, relation_weights, target_svs, rng,
        )

    # --- CS structure from Block D ---
    cs_size_mean_val = _skewnorm_mean(d.cs_size_skew) if d is not None else float("nan")
    if d is not None and not math.isnan(cs_size_mean_val) and cs_size_mean_val > 0:
        cs_size_mean = float(cs_size_mean_val)
        cs_num_templates = max(1, int(d.num_distinct_cs))
        cs_template_zipf = (
            float(d.cs_freq_fit.alpha)
            if not math.isnan(d.cs_freq_fit.alpha) else 2.0
        )
    else:
        cs_size_mean = 0.0   # signal instantiate to derive from E/V budget
        cs_num_templates = 0
        cs_template_zipf = 2.0

    # --- Edge multiplicity, PA exponent, inverse functionality from Block B ---
    if b is not None:
        mean_functionality = _functionality_from_alpha(b.obj_alpha_skew, floor=0.1)
    else:
        mean_functionality = 1.0

    if b is not None:
        alpha_in = b.in_degree_fit.alpha
        # Dorogovtsev-Mendes relation: α = 2 + 1/β → β = 1/(α−2)
        # α must be > 2 for a finite-mean power law; clamp β to [0.1, 2.0].
        if not math.isnan(alpha_in) and alpha_in > 2.0:
            in_pa_exponent = float(np.clip(1.0 / (alpha_in - 2.0), 0.1, 2.0))
        else:
            in_pa_exponent = 0.5
        # Expected maximum in-degree: n^(1/(α−1)) (extreme-value statistic)
        n_ent = a.num_entities
        if not math.isnan(alpha_in) and alpha_in > 1.1 and n_ent > 0:
            max_in_degree = max(10, int(round(n_ent ** (1.0 / (alpha_in - 1.0)))))
        else:
            max_in_degree = 0
        mean_inv_functionality = _functionality_from_alpha(b.subj_alpha_skew, floor=0.01)
    else:
        in_pa_exponent = 0.5
        mean_inv_functionality = 1.0
        max_in_degree = 0

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
        mean_functionality=mean_functionality,
        in_pa_exponent=in_pa_exponent,
        mean_inv_functionality=mean_inv_functionality,
        max_in_degree=max_in_degree,
    )
