"""Generate a *dataset* of synthetic KGs by perturbing one measured signature.

The loop, per graph: load the baseline signature from the corpus → perturb its
feature dict (:mod:`kgsynth.transform`) → rebuild a target
(:meth:`kgsynth.Signature.from_features`) → generate → write. One process per
graph.

Driven by a YAML config::

    kgsynth dataset run.yaml --dry-run     # review the plan; generate nothing
    kgsynth dataset run.yaml --measure     # + re-measure each graph, record distances

Two designs, selected by the config's ``design:`` key:

``joint``
    Every configured feature is jittered at once. A diverse cloud of signatures
    around the baseline — for building a synthetic-KG corpus.

``ofat``
    One-factor-at-a-time: one baseline graph, then one graph per (knob, level).
    Only one thing changes per graph, so any difference in the output is
    attributable to that feature — a sensitivity analysis over the signature.

See ``user_docs/dataset.md``.
"""

from .config import DatasetConfig
from .plan import WorkUnit, build_units
from .runner import describe, load_manifest, run
from .worker import InvalidSignature, UnitResult, run_unit

__all__ = [
    "DatasetConfig",
    "WorkUnit",
    "build_units",
    "run",
    "describe",
    "load_manifest",
    "run_unit",
    "UnitResult",
    "InvalidSignature",
]
