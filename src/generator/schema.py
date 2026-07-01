"""Stage 1 output: the abstract Schema dataclass handed to Stage 2."""

from dataclasses import dataclass, field

import numpy as np

from signature import QUANTILE_LEVELS

# Canonical "fit unavailable" quantile function (one NaN per QUANTILE_LEVELS
# level); the generator treats it as "no usable shape" and falls back to neutral
# behavior.
_NAN_Q = (float("nan"),) * len(QUANTILE_LEVELS)


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
    mean_functionality: float = 1.0      # out-side fallback only (CS-size mean when no Block D)
    in_pa_exponent: float = 0.5          # PA exponent → aggregate in-degree hub preference
    max_in_degree: int = 0               # 0 → uncapped; limits in-degree hub formation
    max_out_degree: int = 0              # 0 → uncapped; limits out-degree hub formation
    # Per-relation multiplicity shape + G2b offset (Block B); CS-size shape (Block D).
    # Defaults are NEUTRAL (no tail shape / no offset / budget-derived CS size), not the
    # old wiring — Stage 2 falls back to uniform per-subject weights when these are NaN.
    obj_alpha_q: tuple = field(default_factory=lambda: _NAN_Q)   # per-relation obj-mult α quantiles
    a_obj: float = 0.0                   # G2b cs_size^a out-degree offset (0 → no effect)
    subj_alpha_q: tuple = field(default_factory=lambda: _NAN_Q)  # per-relation subj-mult α quantiles
    a_subj: float = 0.0                  # G2b inv_cs_size^a in-degree offset (0 → no effect)
    cs_size_q: tuple = field(default_factory=lambda: _NAN_Q)     # forward CS-size quantiles
    # Inverse CS (object side), symmetric to forward CS. 0 templates → every object
    # eligible for every relation (today's behaviour) and the a_subj factor is inert.
    inv_cs_size_q: tuple = field(default_factory=lambda: _NAN_Q)
    inv_cs_num_templates: int = 0        # 0 → no inverse-CS restriction
    inv_cs_template_zipf: float = 2.0    # inverse-CS reuse skew (inv_cs_freq α)
    # Block F-derived connectivity targets.  Defaults reproduce current fully-connected behaviour.
    target_num_components: int = 1    # target weakly-connected component count
    target_lcc: float = 1.0           # target largest-component fraction of entity nodes
    # Block F-derived path-length targets.  NaN / 0 = skip the corresponding steering step.
    path_mean_target: float = float("nan")  # target mean shortest path (skew-normal mean)
    path_hi_target: int = 0                 # target diameter cap (skew-normal hi); 0 = uncapped
    # Co-occurrence group prototypes (Block C subj_cooc_exp / obj_cooc_exp).
    # When set, Stage 2 uses these to generate entity CSes (instead of type_relation_probs)
    # and assigns types post-hoc via log P(CS|t) argmax.  None → fall back to the
    # type-based CS path (existing behaviour).  See docs/generator.md §"Co-occurrence groups".
    subj_group_probs: np.ndarray | None = None    # (COOC_NUM_GROUPS, |R|) forward group prototypes
    subj_group_weights: np.ndarray | None = None  # (COOC_NUM_GROUPS,) Zipf weights from subj spectrum
    obj_group_probs: np.ndarray | None = None     # (COOC_NUM_GROUPS, |R|) inverse group prototypes
    obj_group_weights: np.ndarray | None = None   # (COOC_NUM_GROUPS,) Zipf weights from obj spectrum
