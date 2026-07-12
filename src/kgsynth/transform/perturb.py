"""Perturbation transforms: jitter a measured signature's features.

Two designs, sharing one per-feature spec:

- :class:`Perturb` (*joint*) — every configured feature is jittered at once, each
  by its own random draw. Produces a diverse cloud of signatures around the
  baseline; the design for building a synthetic-KG corpus.
- :class:`PerturbOne` (*OFAT*, one-factor-at-a-time) — exactly one feature moves,
  by a fixed level, everything else stays at baseline. Because only one thing
  changed, any difference in the generated graph is attributable to that feature:
  a sensitivity analysis.

Both are :class:`~kgsynth.transform.SignatureTransform`s over the flat feature
dict, and both report which perturbations their domain clamped — see
:class:`ClampReport`.
"""

from dataclasses import dataclass, field

import numpy as np

from .._domains import feature_domain
from ._surface import group_of


@dataclass(frozen=True)
class FeatureSpec:
    """How one feature is perturbed.

    :param dist: ``"lognormal"`` (multiplicative: ``v * exp(N(0, sigma))``, the
        default for positive quantities — it keeps ``V * 1.1`` rather than
        ``V + 1.1``), ``"normal"`` (additive: ``v + N(0, sigma)``, for signed or
        bounded features), ``"uniform"`` (additive, ``U(lo, hi)``) or
        ``"loguniform"`` (multiplicative, ``exp(U(log lo, log hi))``).
    :param sigma: Spread for ``lognormal`` / ``normal``.
    :param lo: Lower bound for ``uniform`` / ``loguniform``.
    :param hi: Upper bound for ``uniform`` / ``loguniform``.
    :param levels: OFAT levels — multipliers for the multiplicative dists, offsets
        for the additive ones. Ignored by :class:`Perturb`.
    :param clamp: Optional ``(lo, hi)`` overriding the feature's default domain.
    """

    dist: str = "lognormal"
    sigma: float = 0.1
    lo: float | None = None
    hi: float | None = None
    levels: tuple[float, ...] = ()
    clamp: tuple[float, float] | None = None

    _MULTIPLICATIVE = frozenset({"lognormal", "loguniform"})
    _VALID = frozenset({"lognormal", "normal", "uniform", "loguniform"})

    def __post_init__(self) -> None:
        if self.dist not in self._VALID:
            raise ValueError(
                f"Unknown dist {self.dist!r}. Valid: {sorted(self._VALID)}"
            )
        if self.dist in ("uniform", "loguniform") and (self.lo is None or self.hi is None):
            raise ValueError(f"dist={self.dist!r} needs both 'lo' and 'hi'")
        if self.dist in ("lognormal", "normal") and self.sigma <= 0:
            raise ValueError(f"dist={self.dist!r} needs sigma > 0, got {self.sigma}")

    @property
    def multiplicative(self) -> bool:
        """Whether this spec scales the baseline rather than shifting it."""
        return self.dist in self._MULTIPLICATIVE

    def draw(self, rng: np.random.Generator) -> float:
        """Draw one perturbation factor (multiplicative) or offset (additive)."""
        if self.dist == "lognormal":
            return float(np.exp(rng.normal(0.0, self.sigma)))
        if self.dist == "normal":
            return float(rng.normal(0.0, self.sigma))
        if self.dist == "loguniform":
            return float(np.exp(rng.uniform(np.log(self.lo), np.log(self.hi))))
        return float(rng.uniform(self.lo, self.hi))


@dataclass
class ClampReport:
    """Which perturbations the feature domains absorbed.

    A perturbation clamped back to its domain boundary is a **no-op wearing a
    costume**: the feature reads as "perturbed" in the config but the generator
    sees the same value. That turns into a false "this feature has no effect"
    reading in an OFAT sweep, so it is recorded rather than swallowed.

    This is not hypothetical. ``swdf``'s ``obj_mult_alpha`` quantiles already sit
    at the ``[1.4, 3.0]`` fit window's upper bound for its top three levels, so
    *any* upward jitter of that group is largely absorbed.

    :param clamped: Feature name → ``(requested, actual)`` for each value the
        domain moved.
    :param absorbed: Coupled group (as a name tuple) → fraction of its entries
        that clamped, in ``[0, 1]``.
    """

    clamped: dict[str, tuple[float, float]] = field(default_factory=dict)
    absorbed: dict[tuple[str, ...], float] = field(default_factory=dict)

    #: Fraction of a group's entries that must clamp before it is called saturated.
    SATURATION_THRESHOLD = 0.5

    def saturated(self) -> list[tuple[str, ...]]:
        """Coupled groups whose perturbation was mostly absorbed by their domain."""
        return [g for g, frac in self.absorbed.items() if frac > self.SATURATION_THRESHOLD]

    def as_json(self) -> dict:
        """A JSON-safe view for ``meta.json``."""
        return {
            "clamped": {k: list(v) for k, v in self.clamped.items()},
            "absorbed": {",".join(g): f for g, f in self.absorbed.items()},
            "saturated": [",".join(g) for g in self.saturated()],
        }


def _apply_to_group(
    feats: dict[str, float],
    group: tuple[str, ...],
    factor: float,
    spec: FeatureSpec,
    report: ClampReport,
) -> None:
    """Apply one shared *factor* to every member of *group*, in place.

    A coupled group takes a single factor rather than per-entry draws so a
    quantile function stays non-decreasing by construction. The result is sorted
    defensively afterwards: clamping can still flatten (never invert) a group
    whose entries straddle a domain bound.
    """
    n_clamped = 0
    for name in group:
        base = feats[name]
        if base != base:  # NaN: an unconverged fit, not a value to perturb
            continue
        domain = spec.clamp and _custom_domain(spec.clamp) or feature_domain(name)
        raw = base * factor if spec.multiplicative else base + factor
        value = domain.clamp(raw)
        if domain.clamped(raw):
            report.clamped[name] = (raw, value)
            n_clamped += 1
        feats[name] = value

    if len(group) > 1:
        report.absorbed[group] = n_clamped / len(group)
        values = [feats[n] for n in group]
        if not _is_quantile_group(group):
            return
        finite = [v for v in values if v == v]
        if finite != sorted(finite):
            for name, value in zip(group, _resorted(values)):
                feats[name] = value


def _is_quantile_group(group: tuple[str, ...]) -> bool:
    """Whether *group* is a quantile function (must stay non-decreasing)."""
    return all(n.rsplit("_", 1)[-1].startswith("q") for n in group)


def _resorted(values: list[float]) -> list[float]:
    """Sort the finite entries of *values* ascending, leaving NaN slots in place."""
    finite = sorted(v for v in values if v == v)
    it = iter(finite)
    return [next(it) if v == v else v for v in values]


def _custom_domain(bounds: tuple[float, float]):
    """Build a Domain from an explicit ``(lo, hi)`` config override."""
    from .._domains import Domain

    return Domain(lo=bounds[0], hi=bounds[1])


@dataclass(frozen=True)
class Perturb:
    """Jitter every configured feature at once (the *joint* design).

    :param specs: Feature name → :class:`FeatureSpec`. Coupled features (quantile
        functions) may be named by any one of their members; the whole group moves
        together by one shared factor.
    """

    specs: dict[str, FeatureSpec]

    def apply(
        self, feats: dict[str, float], rng: np.random.Generator
    ) -> tuple[dict[str, float], ClampReport]:
        """Return a perturbed copy of *feats* and a report of what the domains clamped."""
        out = dict(feats)
        report = ClampReport()
        for name, spec in self.specs.items():
            _apply_to_group(out, group_of(name), spec.draw(rng), spec, report)
        return out, report

    def describe(self) -> dict:
        """A JSON-safe summary for ``meta.json``."""
        return {"design": "joint", "features": sorted(self.specs)}


@dataclass(frozen=True)
class PerturbOne:
    """Move exactly one feature by a fixed level (the *OFAT* design).

    Every other feature stays at baseline, so any difference in the generated
    graph is attributable to this one feature. ``level`` is a multiplier for
    multiplicative specs and an offset for additive ones.

    :param feature: The feature to move (its whole coupled group moves with it).
    :param level: The multiplier / offset to apply.
    :param spec: The feature's spec, which decides which of those two it is.
    """

    feature: str
    level: float
    spec: FeatureSpec

    def apply(
        self, feats: dict[str, float], rng: np.random.Generator
    ) -> tuple[dict[str, float], ClampReport]:
        """Return a copy of *feats* with only this feature's group moved.

        *rng* is unused — an OFAT level is deterministic — but is accepted so the
        transform interface stays uniform.
        """
        out = dict(feats)
        report = ClampReport()
        _apply_to_group(out, group_of(self.feature), self.level, self.spec, report)
        return out, report

    def describe(self) -> dict:
        """A JSON-safe summary for ``meta.json``."""
        return {"design": "ofat", "feature": self.feature, "level": self.level}


@dataclass(frozen=True)
class Identity:
    """The null transform — the OFAT baseline graph, generated from the unperturbed signature."""

    def apply(
        self, feats: dict[str, float], rng: np.random.Generator
    ) -> tuple[dict[str, float], ClampReport]:
        """Return *feats* unchanged."""
        return dict(feats), ClampReport()

    def describe(self) -> dict:
        """A JSON-safe summary for ``meta.json``."""
        return {"design": "baseline"}
