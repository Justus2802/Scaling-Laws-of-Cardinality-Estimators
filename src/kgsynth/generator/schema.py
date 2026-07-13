"""Stage 1 output: the abstract Schema dataclass handed to Stage 2."""

from dataclasses import dataclass, field

import numpy as np

from ..signature import QUANTILE_LEVELS

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
    # Block C-derived pair-level edge multiplicity (overlap) targets. Defaults = 1.0
    # reproduce the legacy near-simple graph (no shared pairs). Stage 2 biases the
    # stub pairing toward already-used pairs (parallel) / reversed pairs (bidir) to
    # hit these — degree-neutral, since it only correlates which pending stub a
    # subject pairs with. See docs/notes/motif_reachability_and_edge_multiplicity.md.
    edge_multiplicity: float = 1.0       # directed content edges / distinct directed pairs (≥1)
    bidirectional_ratio: float = 1.0     # distinct directed pairs / distinct undirected pairs
    # (in [1,2])
    # Per-relation reciprocity (Block B recip_symmetric_frac/recip_symmetric_value,
    # looked up by each relation's own frequency rank in Stage 1 — not an independent
    # marginal draw), one value in [0,1] per relation, indexed like `relations`.
    # Drives the shared-pool bidirectional construction in Stage 2: a relation with
    # reciprocity ρ_r builds mutual (a↔b) pairs for a ρ_r
    # fraction of its edges. None → all-asymmetric (legacy behaviour).
    relation_reciprocity: "np.ndarray | None" = None
    # Block D-derived CS structure. Blocks B/D/F are mandatory (see docs/generator.md
    # §"Target signature must be complete"), so these are always populated from real
    # measurements — no "0/None → degraded mode" sentinels.
    cs_num_templates: int = 0       # number of reusable CS templates (Block D num_distinct_cs)
    cs_template_zipf: float = 2.0   # Zipf exponent for template frequency (cs_freq α)
    # Support of the reuse draw — the bounds cs_freq's truncated power law was fitted
    # over (v_min/v_max). NaN → flat reuse weights (degenerate fit).
    cs_template_vmin: float = float("nan")
    cs_template_vmax: float = float("nan")
    # Per-entity target degree sequences sampled from Block B's signature-vector
    # components (degree power-law α, p90/max scalars, mean degree); replace the
    # old global max-degree caps.
    target_out_degrees: "np.ndarray | None" = None
    target_in_degrees: "np.ndarray | None" = None
    # Per-relation multiplicity shape + G2b offset (Block B); CS-size shape (Block D).
    # Defaults are NEUTRAL (no tail shape / no offset), not the old wiring — Stage 2
    # falls back to uniform per-subject weights when these are NaN (small-R fallback,
    # not a Block-absence fallback — see docs/generator.md).
    obj_alpha_q: tuple = field(default_factory=lambda: _NAN_Q)   # per-relation obj-mult α quantiles
    a_obj: float = 0.0                   # G2b cs_size^a out-degree offset (0 → no effect)
    # per-relation subj-mult α quantiles
    subj_alpha_q: tuple = field(default_factory=lambda: _NAN_Q)
    a_subj: float = 0.0                  # G2b inv_cs_size^a in-degree offset (0 → no effect)
    # Upper bounds of the two multiplicity laws (Block B obj/subj_mult_max); the
    # per-relation α is a truncated MLE over [1, max], so the draw is too.
    obj_mult_max: float = float("nan")
    subj_mult_max: float = float("nan")
    cs_size_q: tuple = field(default_factory=lambda: _NAN_Q)     # forward CS-size quantiles
    # Inverse CS (object side), symmetric to forward CS.
    inv_cs_size_q: tuple = field(default_factory=lambda: _NAN_Q)
    inv_cs_num_templates: int = 0        # number of reusable inverse-CS templates
    inv_cs_template_zipf: float = 2.0    # inverse-CS reuse skew (inv_cs_freq α)
    inv_cs_template_vmin: float = float("nan")  # reuse-draw support (inv_cs_freq v_min/v_max)
    inv_cs_template_vmax: float = float("nan")
    # Block F-derived connectivity targets.
    target_num_components: int = 1    # target weakly-connected component count
    target_lcc: float = 1.0           # target largest-component fraction of entity nodes
    # Co-occurrence group prototypes (Block C subj_cooc_exp / obj_cooc_exp). Stage 2
    # uses these to generate entity CSes (instead of type_relation_probs) and assigns
    # types post-hoc via log P(CS|t) argmax. See docs/generator.md §"Co-occurrence groups".
    # (COOC_NUM_GROUPS, |R|) forward group prototypes
    subj_group_probs: np.ndarray | None = None
    # (COOC_NUM_GROUPS,) Zipf weights from subj spectrum
    subj_group_weights: np.ndarray | None = None
    # (COOC_NUM_GROUPS, |R|) inverse group prototypes
    obj_group_probs: np.ndarray | None = None
    # (COOC_NUM_GROUPS,) Zipf weights from obj spectrum
    obj_group_weights: np.ndarray | None = None
