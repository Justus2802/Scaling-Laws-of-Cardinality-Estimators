"""The perturbation surface: which signature features a transform may move.

Only **79 of the 127** signature features are read by the generator. Perturbing
any of the other 48 is a silent no-op — the graph comes out identical — so a
config naming one is a user error, not a lenient default: :func:`validate` raises.

Two further categories are accepted but warned about, because "perturbing" them
cannot mean what the caller expects:

- :data:`INERT` — read by the generator, but provably unable to change its output.
- :data:`CONSTANT` — pinned to a fixed value by the fitter, identical on every
  graph, so they carry no information *about* the graph they were measured on.

And :data:`COUPLED` records the feature groups that must move together to stay
valid (a quantile function must remain non-decreasing to be invertible).

See ``docs/dataset.md`` for the derivations behind each set.
"""

from .._domains import MOTIF_COUNTS
from ..signature._fits import QUANTILE_SUFFIXES

_Q = QUANTILE_SUFFIXES  # ("q00", "q10", "q25", "q50", "q75", "q90", "q100")


def _q_group(prefix: str) -> tuple[str, ...]:
    """The 7 feature names of one quantile function, in level order."""
    return tuple(f"{prefix}_{s}" for s in _Q)


# ── coupled groups ─────────────────────────────────────────────────────────────

# Quantile functions back an inverse-CDF sampler (see signature/_fits.py:
# "invertible CDF for inverse-transform sampling"), so their entries must stay
# non-decreasing. Jittering the seven levels independently can invert them and
# corrupt the sampler, so a transform moves a whole group by one shared factor.
# recip_symmetric_frac is a 6-bin probability vector — same treatment.
_RECIP_BINS = tuple(f"recip_symmetric_frac_bin{i}" for i in range(len(_Q) - 1))

COUPLED: tuple[tuple[str, ...], ...] = (
    _q_group("obj_mult_alpha"),
    _q_group("subj_mult_alpha"),
    _q_group("cs_size"),
    _q_group("inv_cs_size"),
    _q_group("rel_freq_logq"),
    _RECIP_BINS,
)

# feature name -> the group it belongs to (identity groups are omitted).
_GROUP_OF: dict[str, tuple[str, ...]] = {
    name: group for group in COUPLED for name in group
}


# ── the surface ────────────────────────────────────────────────────────────────

# Read by the generator (traced through generator/stage{1,2,3}.py). Grouped by
# block for reviewability; the flat frozenset is what callers use.
_SURFACE_A = ("num_entities", "num_relations", "mean_degree", "type_edge_frac")

_SURFACE_B = (
    "out_degree_alpha", "in_degree_alpha",
    "out_degree_max", "out_degree_p90", "in_degree_max", "in_degree_p90",
    "subject_frac", "object_frac",
    "obj_mult_max", "subj_mult_max",
    "a_obj", "a_subj", "recip_symmetric_value",
    *_q_group("obj_mult_alpha"), *_q_group("subj_mult_alpha"),
    *_q_group("rel_freq_logq"),
    *_RECIP_BINS,
)

_SURFACE_C = (
    "num_classes", "class_size_alpha", "edge_multiplicity", "bidirectional_ratio",
    "subj_cooc_rate", "subj_cooc_scale",
    "obj_cooc_rate", "obj_cooc_scale",
    "type_rel_spectrum_rate", "type_rel_spectrum_scale",
)

_SURFACE_D = (
    "num_distinct_cs", "inv_num_distinct_cs",
    "cs_freq_alpha", "cs_freq_vmin", "cs_freq_vmax",
    "inv_cs_freq_alpha", "inv_cs_freq_vmin", "inv_cs_freq_vmax",
    *_q_group("cs_size"), *_q_group("inv_cs_size"),
)

_SURFACE_E = MOTIF_COUNTS  # the 7 raw motif counts Stage 3 steers toward

_SURFACE_F = (
    "num_components", "largest_component_fraction",
    "clustering_coefficient", "degree_assortativity",
)

SURFACE: frozenset[str] = frozenset(
    _SURFACE_A + _SURFACE_B + _SURFACE_C + _SURFACE_D + _SURFACE_E + _SURFACE_F
)

# Read by the generator, but cannot change its output. Both consumers of the
# exp-decay spectra normalise immediately — `svs / svs.sum()` in stage1 — and a
# constant `scale` cancels exactly under normalisation. Their only surviving role
# is the `isnan(scale)` presence check in _adapters._reconstruct_singular_values.
INERT: frozenset[str] = frozenset({
    "subj_cooc_scale", "obj_cooc_scale", "type_rel_spectrum_scale",
})

# Pinned by the fitter, not measured: fit_quantiles(..., lo=1.4, hi=3.0) writes
# the q00/q100 levels to those bounds for *every* graph (see signature/_fits.py:
# "pass lo/hi to pin them to fixed bounds"). They still act as the generator's
# sampling cutoffs, so perturbing them does something — but it moves a design
# constant, not a property of the graph the signature came from.
CONSTANT: frozenset[str] = frozenset({
    "obj_mult_alpha_q00", "obj_mult_alpha_q100",
    "subj_mult_alpha_q00", "subj_mult_alpha_q100",
})

# num_entities moves graph *size*, and the extensive features (motif counts,
# num_relations, num_distinct_cs, num_components) do not follow it automatically:
# holding them fixed while V moves asks the generator for a target no real graph
# satisfies. Rescaling is the job of a size transform — see kgsynth.transform.scale.
SIZE_FEATURES: frozenset[str] = frozenset({"num_entities"})


def group_of(name: str) -> tuple[str, ...]:
    """Return the coupled group *name* belongs to, or just ``(name,)`` if it is alone.

    :param name: A feature name.
    :returns: The feature names that must move together with *name*.
    """
    return _GROUP_OF.get(name, (name,))


def validate(names) -> list[str]:
    """Check feature *names* against the surface, raising on anything unperturbable.

    :param names: Iterable of feature names (e.g. the keys of a config's
        ``features:`` block).
    :returns: Human-readable warnings for names that are on the surface but whose
        perturbation is degenerate (:data:`INERT`, :data:`CONSTANT`,
        :data:`SIZE_FEATURES`). Empty when there is nothing to flag.
    :raises ValueError: If a name is not a signature feature at all, or is a
        feature the generator never reads (perturbing it cannot change the graph).
    """
    from ..signature import _BLOCK_CLASSES

    known = {n for cls in _BLOCK_CLASSES.values() for n in cls.feature_names()}
    names = list(names)

    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(
            f"Not signature features: {sorted(unknown)}. "
            f"See Signature.as_features() for the {len(known)} valid names."
        )

    off_surface = [n for n in names if n not in SURFACE]
    if off_surface:
        raise ValueError(
            f"Features not read by the generator: {sorted(off_surface)}. "
            "Perturbing them cannot change the generated graph. "
            f"The {len(SURFACE)} perturbable features are in kgsynth.transform.SURFACE."
        )

    warnings: list[str] = []
    for n in sorted(set(names) & INERT):
        warnings.append(
            f"{n!r} is read but inert: the exp-decay spectrum is normalised before use, "
            "so a constant scale factor cancels. Perturbing it yields an identical graph."
        )
    for n in sorted(set(names) & CONSTANT):
        warnings.append(
            f"{n!r} is pinned by the fitter to a fixed cutoff (identical on every graph), "
            "not measured. Perturbing it moves the sampler's truncation bound."
        )
    for n in sorted(set(names) & SIZE_FEATURES):
        warnings.append(
            f"{n!r} changes graph size, but the extensive features (motif counts, "
            "num_relations, num_distinct_cs, num_components) will not follow it. "
            "For a size change, use a scaling transform instead — see kgsynth.transform.scale."
        )
    return warnings
