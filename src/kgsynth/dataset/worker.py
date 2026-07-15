"""Generate one graph from one :class:`~kgsynth.dataset.plan.WorkUnit`.

Runs in its own process, so everything here is module-level and picklable. The
worker owns the whole per-graph lifecycle: load the baseline signature, perturb it,
check the result is still a legal signature, generate, and write the artifacts.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import __version__
from .._logging import get_logger
from ..corpus import DEFAULT_SEARCH_DIRS, load_target_from_corpus
from ..generator.pipeline import Generator, Signature
from ..kg_io import save_kg
from ..signature import _distance
from ..signature._fits import QUANTILE_SUFFIXES
from .plan import WorkUnit

log = get_logger(__name__)

# Quantile groups must stay non-decreasing to remain invertible (see transform).
_QUANTILE_PREFIXES = ("obj_mult_alpha", "subj_mult_alpha", "cs_size", "inv_cs_size")

_MOTIFS = (
    "triangle_count", "four_cycle_count", "five_cycle_count", "six_cycle_count",
    "diamond_count", "k4_count", "tailed_triangle_count",
)


@dataclass
class UnitResult:
    """What one work unit produced (or how it failed)."""

    index: int
    label: str
    ok: bool
    elapsed: float
    out_dir: str
    error: str = ""
    num_entities: int = 0
    num_edges: int = 0
    saturated: list = None  # coupled groups whose perturbation the domain absorbed

    def as_json(self) -> dict:
        """A manifest row."""
        return {
            "index": self.index,
            "label": self.label,
            "ok": self.ok,
            "elapsed_s": round(self.elapsed, 2),
            "out_dir": self.out_dir,
            "num_entities": self.num_entities,
            "num_edges": self.num_edges,
            "saturated": self.saturated or [],
            **({"error": self.error} if self.error else {}),
        }


class InvalidSignature(ValueError):
    """A perturbation produced a signature no graph could satisfy."""


def validate_features(feats: dict[str, float]) -> None:
    """Reject a perturbed signature that is not physically realizable.

    Checked **before** generating, because generation is the expensive step and a
    nonsensical target either crashes deep inside a stage or silently produces
    something meaningless.

    :param feats: The perturbed feature dict.
    :raises InvalidSignature: On any violated invariant.
    """
    v = feats["num_entities"]
    if not v >= 1:
        raise InvalidSignature(f"num_entities = {v}")

    edges = round(v * feats["mean_degree"])
    if edges < 1:
        raise InvalidSignature(
            f"mean_degree {feats['mean_degree']:.4g} × V {v:.0f} → {edges} edges"
        )

    for name in ("num_distinct_cs", "inv_num_distinct_cs", "num_components"):
        if feats[name] > v:
            raise InvalidSignature(f"{name} = {feats[name]:.0f} exceeds num_entities = {v:.0f}")

    # A multiplicity law's upper bound below 1 has no support to draw from (every
    # multiplicity is ≥ 1 by construction). NaN is a legitimate "no relations
    # measured" outcome and passes.
    for name in ("obj_mult_max", "subj_mult_max"):
        if feats[name] < 1:
            raise InvalidSignature(f"{name} = {feats[name]:.4g}")

    lcc = feats["largest_component_fraction"]
    if lcc == lcc and not 0 < lcc <= 1:
        raise InvalidSignature(f"largest_component_fraction = {lcc}")

    for motif in _MOTIFS:
        if feats[motif] < 0:
            raise InvalidSignature(f"{motif} = {feats[motif]}")

    for prefix in _QUANTILE_PREFIXES:
        q = [feats[f"{prefix}_{s}"] for s in QUANTILE_SUFFIXES]
        finite = [x for x in q if x == x]
        if finite != sorted(finite):
            raise InvalidSignature(f"{prefix} quantiles are not non-decreasing: {q}")


def _write_json(path: Path, payload: dict) -> None:
    """Write *payload* as JSON, allowing NaN (an unconverged fit is real data)."""
    path.write_text(json.dumps(payload, indent=2))


def _distances(target: Signature, achieved: Signature) -> dict:
    """Per-block distances between the requested target and what was generated.

    Two views, matching ``scripts/signature_roundtrip.py``: per-feature relative
    error over the 135-vector, and the normalised Wasserstein-1 between the fitted
    distributions (blocks B/C/D expose ``distribution_fits()``).
    """
    t_feats, a_feats = target.as_features(), achieved.as_features()
    names = list(t_feats)
    t_vec = np.array([t_feats[n] for n in names], dtype=float)
    a_vec = np.array([a_feats[n] for n in names], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(t_vec - a_vec) / np.maximum(np.abs(t_vec), 1e-9)

    w1: dict[str, float] = {}
    for letter in ("b", "c", "d"):
        t_blk, a_blk = getattr(target, letter), getattr(achieved, letter)
        if t_blk is None or a_blk is None:
            continue
        for (name, t_fit, kind), (_, a_fit, _) in zip(
            t_blk.distribution_fits(), a_blk.distribution_fits()
        ):
            dist = _distance.wasserstein1(t_fit, a_fit, kind)
            iqr = _distance.reconstructed_iqr(t_fit, kind)
            w1[f"{letter}:{name}"] = (
                dist / iqr if (iqr is not None and iqr > 0) else float("nan")
            )

    return {
        # Block E is colour-coding *estimated* for k>=4, so part of any Block E
        # distance is estimator variance rather than generator error. See
        # scripts/cc_variance.py for the noise floor it has to clear.
        "block_e_estimated": True,
        "per_feature_relative_error": dict(zip(names, rel.tolist())),
        "mean_relative_error": float(np.nanmean(rel)),
        "median_relative_error": float(np.nanmedian(rel)),
        "normalised_w1": w1,
        "mean_normalised_w1": float(np.nanmean(list(w1.values()))) if w1 else float("nan"),
    }


def run_unit(unit: WorkUnit) -> UnitResult:
    """Generate one graph and write its artifacts. Never raises: failures are returned.

    A worker exception must not take the pool down — one bad unit in a 169-graph
    sweep should cost that graph, not the run — so everything is caught and reported
    as a failed :class:`UnitResult`.

    :param unit: The unit to run.
    :returns: What it produced, or the error that stopped it.
    """
    started = time.monotonic()
    try:
        base, _, _ = load_target_from_corpus(unit.base, DEFAULT_SEARCH_DIRS)
        feats, report = unit.transform.apply(
            base.as_features(), np.random.default_rng(unit.perturb_seed)
        )
        validate_features(feats)

        target = Signature.from_features(feats)
        opts = dict(unit.generator_opts)
        graph = Generator(target).sample(seed=unit.generate_seed, **opts)

        unit.out_dir.mkdir(parents=True, exist_ok=True)
        save_kg(graph, unit.out_dir / "graph.ttl", fmt="turtle")

        # The flat feature dict, not a block_*.json tree: a signature rebuilt from
        # features has no plot arrays, so to_serializable() would emit a file that
        # looks measured but silently is not. The dict is exactly what from_features
        # consumes, so this round-trips cleanly.
        _write_json(unit.out_dir / "target.json", {
            "source": f"perturbed:{unit.base}:{unit.label}",
            "features": feats,
        })

        if unit.measure:
            achieved = Signature.from_graph(graph)
            _write_json(unit.out_dir / "achieved.json", {
                "source": f"measured:{unit.out_dir.name}",
                "features": achieved.as_features(),
            })
            _write_json(unit.out_dir / "distance.json", _distances(target, achieved))

        elapsed = time.monotonic() - started
        saturated = [",".join(g) for g in report.saturated()]
        _write_json(unit.out_dir / "meta.json", {
            "index": unit.index,
            "label": unit.label,
            "base": unit.base,
            "transform": unit.transform.describe(),
            "perturb_seed": unit.perturb_seed,
            "generate_seed": unit.generate_seed,
            "generator_opts": opts,
            "clamp_report": report.as_json(),
            "num_entities": graph.vcount(),
            "num_edges": graph.ecount(),
            "elapsed_s": round(elapsed, 2),
            "kgsynth_version": __version__,
            # The tracked corpus has not been regenerated since the pinned-xmin fit
            # change, so signatures measured before and after are not comparable.
            # Stamped here so two datasets can never be silently mixed.
            "corpus_regenerated": False,
        })

        if saturated:
            log.warning(
                "unit %d (%s): perturbation mostly absorbed by domain clamps: %s",
                unit.index, unit.label, saturated,
            )
        return UnitResult(
            index=unit.index, label=unit.label, ok=True, elapsed=elapsed,
            out_dir=str(unit.out_dir), num_entities=graph.vcount(),
            num_edges=graph.ecount(), saturated=saturated,
        )

    except Exception as exc:  # noqa: BLE001 — one bad unit must not kill the pool
        elapsed = time.monotonic() - started
        log.error("unit %d (%s) failed: %s", unit.index, unit.label, exc)
        return UnitResult(
            index=unit.index, label=unit.label, ok=False, elapsed=elapsed,
            out_dir=str(unit.out_dir), error=f"{type(exc).__name__}: {exc}",
        )
