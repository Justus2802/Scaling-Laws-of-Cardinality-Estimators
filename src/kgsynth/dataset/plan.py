"""Enumerate a dataset run's work units. Pure: no IO, no generation.

Keeping the enumeration free of side effects is what makes ``--dry-run`` possible
and lets a 169-unit OFAT plan be unit-tested in milliseconds.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..transform import Identity, Perturb, PerturbOne, group_of
from .config import DatasetConfig


@dataclass(frozen=True)
class WorkUnit:
    """One graph to generate. Crosses a process boundary, so: frozen and picklable.

    It carries **no** ``Signature`` and no graph — only primitives and a frozen
    transform. Each worker loads the (small) base signature JSON itself.

    :param index: Position in the plan; also the output directory's number.
    :param out_dir: Where this graph's artifacts are written.
    :param base: Corpus graph name to load as the baseline signature.
    :param transform: The (frozen) transform to apply to its feature dict.
    :param label: Human-readable description, e.g. ``"mean_degree×1.20"``.
    :param perturb_seed: Seeds the transform's RNG.
    :param generate_seed: Seeds :meth:`kgsynth.Generator.sample`.
    :param measure: Re-measure the generated graph and record distances.
    :param generator_opts: Forwarded to :meth:`kgsynth.Generator.sample`.
    """

    index: int
    out_dir: Path
    base: str
    transform: object
    label: str
    perturb_seed: int
    generate_seed: int
    measure: bool
    generator_opts: tuple  # dict as sorted items, so the unit stays hashable


def build_units(config: DatasetConfig) -> list[WorkUnit]:
    """Enumerate every graph a config asks for.

    ``joint`` yields ``num_graphs`` units, each perturbing all configured features.
    ``ofat`` yields one **baseline** unit plus one unit per ``(knob, level)`` pair:
    it enumerates *knobs*, not features, since a coupled quantile group is one knob
    (naming any member moves all seven). So a sweep of 42 knobs × 4 levels is 169
    graphs, not 4 × the feature count.

    Seeds come from :class:`numpy.random.SeedSequence`, spawned once per unit, so a
    unit's result depends only on its index — never on worker count or completion
    order. (``seed + i`` would not give that: two units could collide after a
    resume.)

    :param config: The validated config.
    :returns: The work units, in plan order.
    """
    specs = config.specs
    plan: list[tuple[object, str]] = []

    if config.design == "joint":
        transform = Perturb(specs)
        plan = [(transform, "joint")] * config.num_graphs
    else:
        plan = [(Identity(), "baseline")]
        for knob in _knobs(specs):
            spec = specs[knob]
            for level in spec.levels:
                op = "×" if spec.multiplicative else "+"
                plan.append((PerturbOne(knob, level, spec), f"{knob}{op}{level:g}"))

    # One independent seed stream per unit; two draws each (perturb, generate).
    children = np.random.SeedSequence(config.seed).spawn(len(plan))
    width = max(4, len(str(len(plan) - 1)))

    return [
        WorkUnit(
            index=i,
            out_dir=config.out_dir / f"graph_{i:0{width}d}",
            base=config.base,
            transform=transform,
            label=label,
            perturb_seed=int(seeds[0]),
            generate_seed=int(seeds[1]),
            measure=config.measure,
            generator_opts=tuple(sorted(config.generator_opts.items())),
        )
        for i, ((transform, label), seeds) in enumerate(
            zip(plan, (c.generate_state(2) for c in children))
        )
    ]


def _knobs(specs: dict) -> list[str]:
    """Collapse the configured features to their coupled groups, preserving order.

    Naming two members of the same quantile group is not two knobs — moving either
    moves the whole group — so the group is swept once, under its first-named member.
    """
    seen: set[tuple[str, ...]] = set()
    knobs: list[str] = []
    for name in specs:
        group = group_of(name)
        if group not in seen:
            seen.add(group)
            knobs.append(name)
    return knobs
