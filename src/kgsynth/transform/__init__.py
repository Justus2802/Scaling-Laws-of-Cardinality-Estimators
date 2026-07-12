"""Signature transforms — seeded, deterministic maps over the flat feature dict.

A transform takes the 124-key feature dict of a measured signature
(:meth:`kgsynth.Signature.as_features`) and returns a modified copy, which
:meth:`kgsynth.Signature.from_features` turns back into a generator-usable target::

    feats = Signature.from_file("swdf.nt").as_features()
    feats, report = Perturb(specs).apply(feats, np.random.default_rng(0))
    graph = Generator(Signature.from_features(feats)).sample(seed=1)

Transforms operate on the **public feature dict**, not on block internals, so they
never depend on a block's private attribute names.

Available transforms
--------------------
:class:`Perturb`      Jitter every configured feature at once (the *joint* design).
:class:`PerturbOne`   Move one feature by a fixed level (*OFAT*: sensitivity analysis).
:class:`Identity`     The null transform — an unperturbed baseline graph.
``ScaleTo``           Resize the graph. **Not implemented** — see :mod:`.scale` for
                      what it needs (a measured rescaling law for the extensive
                      features) and why that is blocked on data.

Only **74 of the 124** features are read by the generator; perturbing any other is
a silent no-op, so :func:`validate` raises on one. See :mod:`._surface`.
"""

from typing import Protocol

import numpy as np

from ._surface import CONSTANT, COUPLED, INERT, SIZE_FEATURES, SURFACE, group_of, validate
from .perturb import ClampReport, FeatureSpec, Identity, Perturb, PerturbOne


class SignatureTransform(Protocol):
    """A seeded, deterministic map over a signature's flat feature dict.

    Structural — a transform needs no base class, only these two methods. Every
    implementation must be **frozen** (picklable, hashable) so a transform can
    cross a process boundary inside a work unit, and must **not mutate** the dict
    it is given: workers share a baseline.
    """

    def apply(
        self, feats: dict[str, float], rng: np.random.Generator
    ) -> tuple[dict[str, float], ClampReport]:
        """Return a perturbed copy of *feats*, plus a report of what the domains clamped."""
        ...

    def describe(self) -> dict:
        """Return a JSON-safe summary of this transform, for a dataset's ``meta.json``."""
        ...


#: Registry of the transforms a dataset config may name under ``design:``.
#: A new transform (e.g. ``ScaleTo``) becomes available by adding one entry here.
TRANSFORMS: dict[str, type] = {
    "joint": Perturb,
    "ofat": PerturbOne,
}

__all__ = [
    "SignatureTransform",
    "Perturb",
    "PerturbOne",
    "Identity",
    "FeatureSpec",
    "ClampReport",
    "TRANSFORMS",
    "SURFACE",
    "INERT",
    "CONSTANT",
    "COUPLED",
    "SIZE_FEATURES",
    "group_of",
    "validate",
]
