"""Parse and validate a dataset run's YAML config.

All validation happens **here**, before a single graph is generated: a run of tens
of graphs takes minutes each, so a typo in a feature name must not surface on
graph 40. See ``user_docs/dataset.md`` for the config reference.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .._logging import get_logger
from ..corpus import DEFAULT_SEARCH_DIRS
from ..transform import FeatureSpec, validate

log = get_logger(__name__)

_DESIGNS = ("joint", "ofat")


@dataclass(frozen=True)
class DatasetConfig:
    """A dataset run, fully resolved and validated.

    :param base: Corpus graph name whose signature is the perturbation baseline.
    :param design: ``"joint"`` (jitter everything at once) or ``"ofat"`` (one
        feature per graph, swept over its levels).
    :param specs: Feature name → :class:`~kgsynth.transform.FeatureSpec`.
    :param out_dir: Where the dataset tree is written.
    :param seed: Master seed; every unit's sub-seeds are derived from it.
    :param num_graphs: How many graphs to generate (``joint`` only; ``ofat``
        derives its count from the levels).
    :param measure: Re-measure each generated graph and record per-block distances.
    :param generator_opts: Forwarded to :meth:`kgsynth.Generator.sample`.
    """

    base: str
    design: str
    specs: dict[str, FeatureSpec]
    out_dir: Path
    seed: int = 0
    num_graphs: int = 0
    measure: bool = False
    generator_opts: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        """Load, validate and resolve a dataset config.

        :param path: Path to the YAML file.
        :returns: The validated config.
        :raises ValueError: On any invalid or inconsistent field. The message names
            the offending key — a run must fail here, not 40 graphs in.
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}

        design = data.get("design", "joint")
        if design not in _DESIGNS:
            raise ValueError(f"design must be one of {_DESIGNS}, got {design!r}")

        raw_features = data.get("features") or {}
        if not raw_features:
            raise ValueError("config has no 'features:' to perturb")

        # Raises on a feature the generator never reads; warns on the degenerate ones.
        for warning in validate(raw_features):
            log.warning("%s", warning)

        specs = {name: _spec(name, spec) for name, spec in raw_features.items()}

        if design == "ofat":
            missing = sorted(n for n, s in specs.items() if not s.levels)
            if missing:
                raise ValueError(
                    f"design: ofat needs 'levels' on every feature; missing on {missing}"
                )
        num_graphs = int(data.get("num_graphs", 0))
        if design == "joint" and num_graphs < 1:
            raise ValueError("design: joint needs 'num_graphs' >= 1")

        base = data.get("base")
        if not base:
            raise ValueError("config has no 'base:' graph")
        _require_cached_block_e(base)

        out_dir = data.get("out_dir")
        if not out_dir:
            raise ValueError("config has no 'out_dir:'")

        return cls(
            base=base,
            design=design,
            specs=specs,
            out_dir=Path(out_dir),
            seed=int(data.get("seed", 0)),
            num_graphs=num_graphs,
            measure=bool(data.get("measure", False)),
            generator_opts=dict(data.get("generator") or {}),
        )


def _spec(name: str, raw: dict) -> FeatureSpec:
    """Build a :class:`FeatureSpec` from one ``features:`` entry."""
    if not isinstance(raw, dict):
        raise ValueError(f"features.{name} must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - {"dist", "sigma", "lo", "hi", "levels", "clamp"}
    if unknown:
        raise ValueError(f"features.{name} has unknown keys: {sorted(unknown)}")

    clamp = raw.get("clamp")
    if clamp is not None:
        clamp = tuple(float(x) for x in clamp)
        if len(clamp) != 2 or clamp[0] >= clamp[1]:
            raise ValueError(f"features.{name}.clamp must be [lo, hi] with lo < hi")

    try:
        return FeatureSpec(
            dist=raw.get("dist", "lognormal"),
            sigma=float(raw.get("sigma", 0.1)),
            lo=None if raw.get("lo") is None else float(raw["lo"]),
            hi=None if raw.get("hi") is None else float(raw["hi"]),
            levels=tuple(float(x) for x in raw.get("levels", ())),
            clamp=clamp,
        )
    except ValueError as exc:
        raise ValueError(f"features.{name}: {exc}") from exc


def _require_cached_block_e(base: str) -> None:
    """Fail early unless *base* has a cached ``block_e.json``.

    ``load_target_from_corpus`` falls back to *measuring* Block E from the source
    graph when the cache is absent. In a worker process that would be both very slow
    and **not reproducible** — Block E's colour-coding estimator is seeded, but
    ``load_kg``'s vertex numbering is hash-ordered, so the same graph measured in two
    processes yields different motif counts. A dataset whose per-graph targets differ
    from each other by measurement noise is worthless, so refuse to start.
    """
    for root in DEFAULT_SEARCH_DIRS:
        sig_dir = root / base / "signature"
        if sig_dir.is_dir():
            if (sig_dir / "block_e.json").exists():
                return
            raise ValueError(
                f"'{base}' has no cached block_e.json ({sig_dir}). Measuring Block E per "
                "worker is slow and not reproducible across processes; run "
                f"`kgsynth measure` on {base} first."
            )
    available = sorted(
        p.name for root in DEFAULT_SEARCH_DIRS if root.is_dir()
        for p in root.iterdir() if (p / "signature").is_dir()
    )
    raise ValueError(f"base graph '{base}' not found in the corpus. Available: {available}")
