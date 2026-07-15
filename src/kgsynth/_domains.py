"""Per-feature domains and size-scaling classification for the 126-key signature.

One source of truth for "what values may this feature legally take", shared by:

- :mod:`kgsynth.signature_sampler` — clamps its uniform draws (it defined these
  tables first; they were lifted here so the perturber could reuse them),
- :meth:`kgsynth.Signature.from_features` — int-casts the count-like features,
- :mod:`kgsynth.transform` — clamps perturbed values back into range.

The ``EXTENSIVE`` / ``INTENSIVE`` split is transcribed from
``docs/notes/signature_size_dependence.md``: extensive features scale with the
number of entities ``V`` or edges ``E``, intensive ones are shapes, exponents or
ratios that do not. It is unused by the perturber (which holds ``V`` fixed) and
exists for the size-rescaling transform that reads it — see
``kgsynth.transform.scale``.
"""

from dataclasses import dataclass

_INF = float("inf")


@dataclass(frozen=True)
class Domain:
    """Legal value range for one feature.

    :param lo: Inclusive lower bound.
    :param hi: Inclusive upper bound.
    :param integer: Whether the feature is count-like and must be a whole number.
    """

    lo: float = -_INF
    hi: float = _INF
    integer: bool = False

    def clamp(self, value: float) -> float:
        """Clamp *value* into ``[lo, hi]`` and round if the domain is integral.

        NaN passes through unchanged: it is a real "fit did not converge"
        measurement outcome (e.g. ``aids``'s per-relation alpha quantiles), not a
        missing value to be repaired.

        :param value: The value to constrain.
        :returns: The constrained value.
        """
        if value != value:  # NaN
            return value
        value = min(self.hi, max(self.lo, value))
        return float(round(value)) if self.integer else value

    def clamped(self, value: float) -> bool:
        """Whether :meth:`clamp` would move *value* — i.e. it sits outside the domain."""
        return value == value and not (self.lo <= value <= self.hi)


# ── named domains ──────────────────────────────────────────────────────────────

UNBOUNDED = Domain()
POSITIVE = Domain(lo=0.0)
COUNT = Domain(lo=1.0, integer=True)      # counts floored at 1: zero entities is not a graph
NON_NEG_COUNT = Domain(lo=0.0, integer=True)  # motif counts, num_classes: zero is legitimate
UNIT = Domain(lo=0.0, hi=1.0)
SIGNED_UNIT = Domain(lo=-1.0, hi=1.0)     # degree_assortativity
GE_ONE = Domain(lo=1.0)                   # edge_multiplicity: ≥1 by construction
BIDIR = Domain(lo=1.0, hi=2.0)            # bidirectional_ratio: directed/undirected pair ratio
# Per-relation multiplicity exponents are fit inside a fixed window and the
# quantile function is *pinned* to its ends — see block_b._ALPHA_LO/_ALPHA_HI and
# the "pinned constants" note in feature_domain() below.
ALPHA = Domain(lo=1.4, hi=3.0)

# Count-like features, rounded to whole numbers. A float num_entities propagates
# into range() and array shapes downstream, so this is load-bearing, not cosmetic.
INTEGER_FEATURES: frozenset[str] = frozenset({
    "num_entities", "num_relations", "num_classes", "num_distinct_cs",
    "inv_num_distinct_cs", "num_components",
    "out_degree_xmin", "in_degree_xmin", "class_size_xmin",
    "cs_freq_vmin", "cs_freq_vmax", "inv_cs_freq_vmin", "inv_cs_freq_vmax",
    "two_step_vmin", "two_step_vmax",
    "obj_mult_max", "subj_mult_max",
    "triangle_count", "four_cycle_count", "five_cycle_count", "six_cycle_count",
    "diamond_count", "k4_count", "tailed_triangle_count",
})

UNIT_INTERVAL: frozenset[str] = frozenset({
    "subj_cooc_density", "obj_cooc_density",
    "largest_component_fraction", "clustering_coefficient",
    "subject_frac", "object_frac",
})

SIGNED_UNIT_FEATURES: frozenset[str] = frozenset({"degree_assortativity"})

# Counts / thresholds floored at 1: a negative count is invalid (not merely
# implausible), and a widened sampling range can dip below zero.
MIN_ONE: frozenset[str] = frozenset({
    "num_entities", "num_relations", "num_distinct_cs", "inv_num_distinct_cs",
    "num_components",
    "out_degree_xmin", "in_degree_xmin", "class_size_xmin",
    "cs_freq_vmin", "cs_freq_vmax", "inv_cs_freq_vmin", "inv_cs_freq_vmax",
    "two_step_vmin", "two_step_vmax",
    "obj_mult_max", "subj_mult_max",   # a measured maximum multiplicity is ≥ 1
})

# Type-block parameters held at the untyped default (NaN) by the population
# sampler until more typed KGs exist. Kept here so signature_sampler keeps its
# single import site; the perturber does not use it (it perturbs a *measured*
# signature, whose type params are real).
TYPE_PARAM_FEATURES: frozenset[str] = frozenset({
    "class_size_alpha", "class_size_xmin",
    "type_rel_spectrum_rate", "type_rel_spectrum_scale",
    "per_type_entropy_rate", "per_type_entropy_scale",
})

# Raw motif counts: zero is legitimate (a triangle-free graph), so they floor at 0.
MOTIF_COUNTS: tuple[str, ...] = (
    "triangle_count", "four_cycle_count", "five_cycle_count", "six_cycle_count",
    "diamond_count", "k4_count", "tailed_triangle_count",
)


def feature_domain(name: str) -> Domain:
    """Return the :class:`Domain` for feature *name*.

    Specific rules win over general ones; anything unrecognised falls back to
    :data:`POSITIVE`, since the overwhelming majority of signature features are
    non-negative magnitudes. ``a_obj`` / ``a_subj`` are the deliberate exception:
    they are regression offsets and may legitimately be negative.

    :param name: Public feature name (a key of ``Signature.as_features()``).
    :returns: The domain constraining that feature.
    """
    if name in ("a_obj", "a_subj"):
        return UNBOUNDED
    if name == "degree_assortativity":
        return SIGNED_UNIT
    if name == "edge_multiplicity":
        return GE_ONE
    if name == "bidirectional_ratio":
        return BIDIR
    if name.startswith(("obj_mult_alpha_", "subj_mult_alpha_")):
        return ALPHA
    if name in UNIT_INTERVAL or name.startswith("recip_symmetric_"):
        return UNIT
    if name in MOTIF_COUNTS or name == "num_classes":
        return NON_NEG_COUNT
    if name in MIN_ONE:
        return COUNT
    if name in INTEGER_FEATURES:
        return NON_NEG_COUNT
    return POSITIVE


# ── size-scaling classification (docs/notes/signature_size_dependence.md) ───────

# Scale with V or E. A transform that changes graph size must rescale these;
# holding them fixed while moving V asks the generator for a target no real graph
# satisfies. Unused by Perturb (which holds V fixed) — see kgsynth.transform.scale.
EXTENSIVE: frozenset[str] = frozenset({
    "num_entities", "num_relations", "num_classes",
    "num_distinct_cs", "inv_num_distinct_cs", "num_components",
    *MOTIF_COUNTS,
    "two_step_vmax",
})

# Weakly / logarithmically size-dependent: drift upward with size but do not
# scale linearly. Called out separately because a linear rescaling law is wrong
# for them.
WEAKLY_EXTENSIVE: frozenset[str] = frozenset({
    "out_degree_xmin", "in_degree_xmin", "class_size_xmin",
    "cs_freq_vmax", "inv_cs_freq_vmax", "two_step_vmin",
    "shortest_path_max", "shortest_path_mean", "shortest_path_var",
})
