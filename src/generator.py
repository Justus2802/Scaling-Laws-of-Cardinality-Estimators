"""kgsynth Stage 1 — Schema sampler.

Builds the abstract schema (relations, types, type-relation probability table)
from a measured BlockA + BlockC target signature.  The schema is the
deterministic seed for Stage 2 (CS-aware graph instantiation).

Design decisions:
  - Relation frequency weights are sampled from a Zipf distribution whose
    exponent is a tunable parameter; the spec requires it but it is not
    directly available from Blocks A or C (Block B would supply it).
  - The type-relation probability table P(r|t) is constructed via a low-rank
    random factorisation whose singular values match the Block C target, so
    the co-occurrence structure of the generated schema resembles the real KG.
  - All randomness goes through a single np.random.Generator seeded at call
    time, making the schema fully reproducible.
"""

from dataclasses import dataclass

import numpy as np

from signature import BlockA, BlockC


# ---------------------------------------------------------------------------
# Schema dataclass
# ---------------------------------------------------------------------------


@dataclass
class Schema:
    """Stage 1 output: abstract schema for a synthetic KG.

    Passed directly to Stage 2 (instantiate) to build the actual graph.

    Attributes
    ----------
    relations : list[str]
        |R| synthetic relation URIs, e.g. "http://kgsynth.org/rel/0".
    relation_weights : np.ndarray, shape (|R|,)
        Normalized frequency weights (sum to 1); controls how often each
        relation appears relative to the others.
    types : list[str]
        |T| synthetic type URIs.  Empty when Block C reports no classes.
    type_weights : np.ndarray, shape (|T|,)
        Normalized type-size weights (sum to 1); governs how many entities
        each type receives in Stage 2.
    type_relation_probs : np.ndarray, shape (|T|, |R|)
        P(r | t) table — for each type, the probability distribution over
        outgoing relations.  Rows sum to 1.  Shape is (0, |R|) when |T| = 0.
    num_entities : int
        Target |V| copied from Block A; used by Stage 2 to size the graph.
    num_triples : int
        Target |E| copied from Block A; used by Stage 2 to size the graph.
    """

    relations: list
    relation_weights: np.ndarray
    types: list
    type_weights: np.ndarray
    type_relation_probs: np.ndarray
    num_entities: int
    num_triples: int


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_schema(
    a: BlockA,
    c: BlockC,
    *,
    relation_zipf_exponent: float = 2.0,
    seed: int = 0,
) -> Schema:
    """Stage 1: derive an abstract schema from a target BlockA + BlockC.

    Parameters
    ----------
    a : BlockA
        Measured size/density signature of the target KG.
        |V|, |E|, |R| are used directly.
    c : BlockC
        Measured schema/correlation signature of the target KG.
        num_classes, class_size_zipf_exponent, and subj_singular_values
        guide the type structure and co-occurrence reconstruction.
    relation_zipf_exponent : float
        Zipf exponent for relation frequency weights.  Controls how skewed
        relation usage is; real KGs typically fall in [1.5, 2.5].  Block B
        (per-relation multiplicity fits) would supply this in a full pipeline;
        here it is an explicit tuning knob.
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

    # --- Relations ---
    relations = [f"http://kgsynth.org/rel/{i}" for i in range(num_relations)]
    relation_weights = _zipf_weights(num_relations, relation_zipf_exponent, rng)

    # --- Types ---
    types = [f"http://kgsynth.org/type/{i}" for i in range(num_types)]

    if num_types > 0:
        type_zipf = c.class_size_zipf_exponent
        if not np.isnan(type_zipf) and type_zipf > 0:
            type_weights = _zipf_weights(num_types, type_zipf, rng)
        else:
            # Block C could not fit a Zipf (too few classes): fall back to uniform
            type_weights = np.full(num_types, 1.0 / num_types)
    else:
        type_weights = np.array([], dtype=float)

    # --- Type-relation probability table ---
    type_relation_probs = _sample_type_relation_probs(
        num_types,
        num_relations,
        relation_weights,
        c.subj_singular_values,
        rng,
    )

    return Schema(
        relations=relations,
        relation_weights=relation_weights,
        types=types,
        type_weights=type_weights,
        type_relation_probs=type_relation_probs,
        num_entities=a.num_entities,
        num_triples=a.num_triples,
    )
