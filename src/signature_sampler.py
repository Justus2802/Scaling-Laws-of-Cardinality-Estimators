"""Population samplers for the reduced graph signature (doc-Stage-1).

Draw a *novel* reduced signature from the distribution of real KGs, so the
generator (doc-Stage-2) has something to instantiate. See
``docs/plan/stage1_population_sampler.md`` for the design and the data-reality
analysis behind it.

This module provides the sampler **class hierarchy**:

- :class:`SignatureSampler` — the reusable ABC. It loads the measured corpus
  (``data/graphs/<name>/signature/signature.json``), exposes the per-feature
  finite values, and runs the shared sample → post-process → emit pipeline.
  Subclasses implement only :meth:`_sample_one` (how one feature is drawn).
- :class:`UniformRangeSampler` — the v0 baseline: each feature is drawn
  independently from a uniform distribution over its observed corpus range,
  widened by ±10 % of that range.

Output is the **88-value feature dict** (the same shape as a measured
``signature.json``'s ``"features"`` block), so sampled signatures are drop-in
compatible with the existing readers. Reconstruction into a
``ReducedGraphSignature`` object is intentionally not done here.

Scope note — **motifs / Block E (G5) are out of scope**, exactly as in the
reduced signature itself (the sampler cannot be more complete than its target
signature). Because :data:`FEATURE_ORDER` is derived from the block classes, if a
reduced Block E is ever added to ``_ALL_BLOCKS`` its features flow in here with no
sampler change — though as raw, size-dependent counts they belong to a
size-conditioned sampler, not this uniform baseline.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from signature import BlockA, BlockB, BlockC, BlockD, BlockF

# Repo root: src/signature_sampler.py -> parents[1] == repo root.
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CORPUS = _ROOT / "data" / "graphs"

# Canonical 88-key order, in the same order as ``ReducedGraphSignature.as_dict()``.
# Derived from the block classes so it never drifts from the measured schema.
_BLOCKS = [BlockA, BlockB, BlockC, BlockD, BlockF]
FEATURE_ORDER: list[str] = [name for blk in _BLOCKS for name in blk.feature_names()]

# ── Feature metadata (the only hand-maintained classification) ──────────────────

# Count-like features rounded to whole numbers after sampling.
_INTEGER_FEATURES: frozenset[str] = frozenset({
    "num_entities", "num_relations", "num_classes", "num_distinct_cs",
    "num_components", "out_degree_xmin", "in_degree_xmin", "relation_zipf_xmin",
    "class_size_xmin", "cs_freq_vmin", "cs_freq_vmax",
    "inv_cs_freq_vmin", "inv_cs_freq_vmax", "two_step_vmin", "two_step_vmax",
})

# Type-block parameters held at the untyped default (NaN) until more typed KGs
# exist (the "acquire typed KGs first" decision). ``num_classes`` is pinned to 0
# separately so the count and its params stay consistent.
_TYPE_PARAM_FEATURES: frozenset[str] = frozenset({
    "class_size_alpha", "class_size_xmin",
    "type_rel_spectrum_rate", "type_rel_spectrum_scale",
    "per_type_entropy_rate", "per_type_entropy_scale",
})

# Bounded-domain features clamped post-sampling (the ±10 % widening can overshoot).
_UNIT_INTERVAL: frozenset[str] = frozenset({
    "subj_cooc_density", "obj_cooc_density",
    "largest_component_fraction", "clustering_coefficient",
})
_SIGNED_UNIT: frozenset[str] = frozenset({"degree_assortativity"})
# Counts / thresholds floored at 1: a negative count is invalid (not merely
# implausible), and the ±10 % widening of a wide range can dip below zero.
_MIN_ONE: frozenset[str] = frozenset({
    "num_entities", "num_relations", "num_distinct_cs", "num_components",
    "out_degree_xmin", "in_degree_xmin", "relation_zipf_xmin", "class_size_xmin",
    "cs_freq_vmin", "cs_freq_vmax", "inv_cs_freq_vmin", "inv_cs_freq_vmax",
    "two_step_vmin", "two_step_vmax",
})

_MIN_FINITE_SUPPORT = 2  # need ≥2 finite corpus values to form a range


class SignatureSampler(ABC):
    """Base class for reduced-signature population samplers.

    Holds the measured corpus and runs the shared pipeline; subclasses implement
    only :meth:`_sample_one`. The corpus is ``{graph_name: {feature: value}}``;
    NaN entries (uncomputed/degenerate features) are excluded per feature.
    """

    def __init__(self, corpus: dict[str, dict[str, float]]) -> None:
        if not corpus:
            raise ValueError("Empty corpus; need at least one measured signature.")
        self.corpus = corpus
        # Per-feature array of finite values across graphs, in FEATURE_ORDER.
        self._finite: dict[str, np.ndarray] = {}
        for feat in FEATURE_ORDER:
            vals = np.array(
                [row.get(feat, float("nan")) for row in corpus.values()], dtype=float
            )
            self._finite[feat] = vals[np.isfinite(vals)]

    # ── construction ───────────────────────────────────────────────────────────

    @classmethod
    def load_corpus(cls, root: str | Path = _DEFAULT_CORPUS) -> "SignatureSampler":
        """Build a sampler from ``<root>/<name>/signature/signature.json`` files.

        Args:
            root: corpus directory (default ``data/graphs/``).

        Returns:
            An instance of *cls* holding the loaded corpus.
        """
        root = Path(root)
        corpus: dict[str, dict[str, float]] = {}
        for sig_path in sorted(root.glob("*/signature/signature.json")):
            data = json.loads(sig_path.read_text())
            feats = data.get("features")
            if not feats:
                continue  # skip old-format files without named features
            name = sig_path.parent.parent.name  # data/graphs/<name>/signature/...
            corpus[name] = feats
        if not corpus:
            raise RuntimeError(
                f"No signature.json files with features found under {root}/"
            )
        return cls(corpus)

    # ── sampling pipeline ───────────────────────────────────────────────────────

    @abstractmethod
    def _sample_one(self, feature: str, rng: np.random.Generator) -> float:
        """Draw a raw value for *feature* (before shared post-processing)."""

    def sample(self, seed: int | None = None) -> dict[str, float]:
        """Sample a full reduced signature as an 88-key feature dict.

        Args:
            seed: RNG seed for reproducibility.

        Returns:
            ``{feature_name: value}`` in :data:`FEATURE_ORDER` order.
        """
        rng = np.random.default_rng(seed)
        return {
            feat: self._postprocess(feat, self._sample_one(feat, rng))
            for feat in FEATURE_ORDER
        }

    def _postprocess(self, feature: str, value: float) -> float:
        """Apply the shared validity rules: type block, support, clamps, rounding."""
        # Type block → untyped default.
        if feature == "num_classes":
            return 0.0
        if feature in _TYPE_PARAM_FEATURES:
            return float("nan")
        # Too few finite corpus values to define a population → NaN.
        if self._finite[feature].size < _MIN_FINITE_SUPPORT or math.isnan(value):
            return float("nan")
        # Domain clamps.
        if feature in _UNIT_INTERVAL:
            value = min(1.0, max(0.0, value))
        elif feature in _SIGNED_UNIT:
            value = min(1.0, max(-1.0, value))
        if feature in _MIN_ONE:
            value = max(1.0, value)
        # Integer rounding for count-like features.
        if feature in _INTEGER_FEATURES:
            value = float(round(value))
        return value

    # ── output ──────────────────────────────────────────────────────────────────

    def to_json(
        self, features: dict[str, float], *, source: str | None = None
    ) -> dict:
        """Wrap a sampled feature dict in the ``{source, features}`` contract."""
        return {"source": source or f"sampled:{type(self).__name__}", "features": features}

    def write(
        self, path: str | Path, features: dict[str, float], *, source: str | None = None
    ) -> None:
        """Write a sampled feature dict as ``signature.json``-shaped JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(features, source=source), indent=2))


class UniformRangeSampler(SignatureSampler):
    """v0: each feature ~ Uniform over its corpus range widened by ±10 %.

    For a feature with finite corpus values ``[lo, hi]`` and range ``r = hi - lo``,
    draws from ``Uniform(lo - 0.1·r, hi + 0.1·r)``. Constant features (``r = 0``)
    are reproduced exactly, so fixed cutoffs (e.g. ``obj_mult_alpha_q00 = 1.4``)
    need no special-casing.
    """

    WIDEN: float = 0.10  # fraction of the min–max range added to each side

    def _sample_one(self, feature: str, rng: np.random.Generator) -> float:
        finite = self._finite[feature]
        if finite.size == 0:
            return float("nan")
        lo, hi = float(finite.min()), float(finite.max())
        pad = self.WIDEN * (hi - lo)
        return float(rng.uniform(lo - pad, hi + pad))
