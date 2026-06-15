"""Stage 1 output: the abstract Schema dataclass handed to Stage 2."""

from dataclasses import dataclass

import numpy as np


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
        Target |E| (from Block A's mean degree × |V|); used by Stage 2 to size
        the graph.
    """

    relations: list
    relation_weights: np.ndarray
    types: list
    type_weights: np.ndarray
    type_relation_probs: np.ndarray
    num_entities: int
    num_triples: int
    # Block D-derived CS structure (defaults = legacy behaviour)
    cs_size_mean: float = 0.0       # 0 → derive from E/V budget at instantiate time
    cs_num_templates: int = 0       # 0 → per-entity independent sampling
    cs_template_zipf: float = 2.0   # Zipf exponent for template frequency
    # Block B-derived edge multiplicity and degree distribution
    mean_functionality: float = 1.0      # 1.0 → single object per (s,p) pair
    in_pa_exponent: float = 0.5          # PA exponent for object selection → in-degree shape
    mean_inv_functionality: float = 1.0  # 1.0 → no cap on subjects per (predicate, object)
    max_in_degree: int = 0               # 0 → uncapped; limits hub formation
