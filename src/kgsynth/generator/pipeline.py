"""High-level API: target Signature + three-stage Generator orchestrator."""

from dataclasses import dataclass
from pathlib import Path

import igraph
import yaml

from ..signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF, _BLOCK_CLASSES

from .._logging import get_logger
from .stage1 import sample_schema
from .stage2 import instantiate
from .stage3 import refine

log = get_logger(__name__)


@dataclass
class Signature:
    """Target signature used by Generator (reduced Blocks A, B, C, D, E, F).

    Block A supplies size/density targets; Block B supplies edge multiplicity
    and degree-distribution PA exponents; Block C supplies schema/class
    structure; Block D supplies CS statistics that enable template-based CS
    reuse; Block E supplies motif counts that Stage 3 optimises toward; Block F
    supplies degree assortativity that Stage 3 also targets.

    Blocks A, B, C, D and F are mandatory — a real graph always measures them,
    and Stage 1/2 have no degraded-mode path for their absence (see
    docs/generator.md §"Target signature must be complete"). Block E stays
    nullable: :func:`kgsynth.corpus.load_target_from_corpus` legitimately skips
    it (``with_block_e=False``) for callers that only drive Stages 1-2, since
    it is the expensive block to measure and neither stage reads it.
    """

    a: "BlockA"
    b: "BlockB"
    c: "BlockC"
    d: "BlockD"
    e: "BlockE | None"
    f: "BlockF"

    @classmethod
    def from_graph(
        cls,
        g: igraph.Graph,
        skip_stars_and_paths: bool = False,
        skip_shortest_paths: bool = False,
    ) -> "Signature":
        """Measure all six blocks from a graph.

        Parameters
        ----------
        skip_stars_and_paths : bool
            Skip Block E's star, 5/6-cycle, path-template and tree-template
            computations (speeds up sweep analysis).
        skip_shortest_paths : bool
            Skip Block F's shortest-path sampling (speeds up sweep analysis).
        """
        return cls(
            a=BlockA().calculate(g),
            b=BlockB().calculate(g),
            c=BlockC().calculate(g),
            d=BlockD().calculate(g),
            e=BlockE().calculate(g, skip_stars_and_paths=skip_stars_and_paths),
            f=BlockF().calculate(g, skip_shortest_paths=skip_shortest_paths),
        )

    @classmethod
    def from_file(cls, path) -> "Signature":
        from ..kg_io import load_kg
        return cls.from_graph(load_kg(Path(path)))

    @classmethod
    def from_config(cls, path: "Path | str") -> "Signature":
        """Load a target signature from a YAML config file.

        The file holds one top-level key per block letter (``a``, ``b``, ``c``,
        ``d``, ``e``, ``f``), each mapping to that block's serialized state —
        the same shape each block's ``to_serializable()`` produces and
        :meth:`to_config` writes — so a config can be hand-edited or produced
        by re-saving a measured/cached signature. All six blocks are required:
        a hand-edited config describes a complete target signature, matching
        what any real graph measures (see docs/generator.md §"Target signature
        must be complete").

        :param path: Path to the YAML config file.
        :returns: The reconstructed target ``Signature``.
        :raises KeyError: If a block is missing (also raised for an empty or
            comment-only file, which parses to no data at all).
        """
        data = yaml.safe_load(Path(path).read_text()) or {}
        blocks: dict = {}
        for letter, block_cls in _BLOCK_CLASSES.items():
            if letter not in data:
                raise KeyError(f"Signature config {path} is missing required block {letter!r}")
            blocks[letter] = block_cls.from_serializable(data[letter])
        return cls(**blocks)

    def to_config(self, path: "Path | str") -> None:
        """Write this signature to a YAML config file readable by :meth:`from_config`.

        :param path: Destination path for the YAML file.
        """
        data = {
            letter: block.to_serializable()
            for letter, block in zip(_BLOCK_CLASSES, self._blocks())
            if block is not None
        }
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    def as_features(self) -> dict[str, float]:
        """Flatten to the public 126-key feature dict.

        The same ``{name: value}`` mapping stored under ``"features"`` in a
        measured ``signature.json``. Requires every block to be present —
        including Block E, so this is not for the ``with_block_e=False``
        Stage-1/2-only signatures (see the class docstring); those never need
        the full feature vector.

        :returns: Feature name → value, in block order (A, B, C, D, E, F).
        """
        feats: dict[str, float] = {}
        for letter, block in zip(_BLOCK_CLASSES, self._blocks()):
            cls = _BLOCK_CLASSES[letter]
            feats.update(zip(cls.feature_names(), block.as_vector()))
        return feats

    @classmethod
    def from_features(cls, feats: dict[str, float]) -> "Signature":
        """Rebuild a generator-usable Signature from a flat feature dict.

        The inverse of :meth:`as_features`, and the counterpart to
        :meth:`from_config` for callers who hold features rather than block
        state — a perturbed signature, a sampled one
        (:mod:`kgsynth.signature_sampler`), or one read from a ``signature.json``
        ``"features"`` block.

        Every value the generator reads is reconstructed exactly: the fit objects
        are rebuilt from their named features (``PowerLawStats.alpha`` from
        ``out_degree_alpha``, ``QuantileFit`` from the seven ``*_q00``..``*_q100``
        keys, and so on).

        **What is not restored:** the raw sample arrays each block keeps for
        plotting — degree lists, class sizes, singular-value spectra. They are not
        in the feature vector, and nothing in the generator reads them. A block
        rebuilt this way therefore **cannot** ``visualize()``, and re-serializing
        it with ``to_serializable()`` would emit a lossy file that looks like a
        measured one. Persist the feature dict itself instead.

        Fit-quality diagnostics absent from the vector (``PowerLawStats``'s ``ks``
        and the three ``D_*`` comparison distances) are filled with NaN — they
        describe a fit that was performed elsewhere, and no consumer reads them.

        :param feats: Feature name → value; must hold all 126 keys.
        :returns: A ``Signature`` whose blocks reproduce every generator-consumed
            value of the signature *feats* came from.
        :raises KeyError: If a feature key is missing.
        """
        return cls(**{
            letter: _BLOCK_CLASSES[letter].from_features(feats)
            for letter in _BLOCK_CLASSES
        })

    def _blocks(self) -> list:
        """Return the six blocks in signature order (``a``..``f``); ``e`` may be ``None``."""
        return [self.a, self.b, self.c, self.d, self.e, self.f]


class Generator:
    """Full three-stage KG generator.

    Usage
    -----
    >>> sig = Signature.from_file("target.ttl")
    >>> gen = Generator(sig)
    >>> g = gen.sample(seed=42)          # reproducible
    >>> g2 = gen.sample(seed=99)         # structurally different

    Parameters
    ----------
    target : Signature
        Measured signature of the target KG.  All three stages read from it.
    """

    def __init__(self, target: Signature) -> None:
        self.target = target

    def sample_pre_refine(
        self,
        *,
        seed: int = 0,
        relation_zipf_exponent: float = 2.0,
    ) -> igraph.Graph:
        """Run Stages 1 and 2 only, returning the graph :meth:`sample` would hand to Stage 3.

        Uses the same derived sub-seeds as :meth:`sample`, so for a given ``seed`` this
        is bit-for-bit the graph ``refine()`` starts from. The diagnostic scripts
        (Stage-3 delta profiling, edge multiplicity, estimator variance) use it to study
        the pre-refinement state without paying for the rewiring loop.

        :param seed: Master seed; Stage 1 uses ``seed``, Stage 2 ``seed + 1``.
        :param relation_zipf_exponent: Passed to Stage 1 (relation-frequency skew).
        :returns: The post-Stage-2, pre-refinement synthetic graph.
        """
        schema = sample_schema(
            self.target.a,
            self.target.c,
            d=self.target.d,
            b=self.target.b,
            f=self.target.f,
            relation_zipf_exponent=relation_zipf_exponent,
            seed=seed,
        )
        return instantiate(schema, seed=seed + 1)

    def sample(
        self,
        *,
        seed: int = 0,
        relation_zipf_exponent: float = 2.0,
        rewire_budget: int = 50_000,
        initial_temp: float = 0.05,
        cooling_rate: float = 0.99993,
        skip_c5: bool = False,
        skip_c6: bool = False,
        adaptive_weights: bool = False,
        convergence_log: "Path | str | None" = None,
        swap_log: "Path | str | None" = None,
        checkpoint_steps: "list[int] | None" = None,
        checkpoint_callback=None,
    ) -> igraph.Graph:
        """Generate one synthetic KG from the target signature.

        Parameters
        ----------
        seed : int
            Master seed; all three stages derive sub-seeds from it so the
            entire pipeline is reproducible from a single integer.
        relation_zipf_exponent : float
            Passed to Stage 1; controls skewness of relation frequency.
        rewire_budget : int
            Number of rewiring attempts in Stage 3.
        initial_temp, cooling_rate : float
            Simulated-annealing parameters for Stage 3. Defaults (0.05, 0.99993)
            are tuned for a ~100k ``rewire_budget`` — the temperature sweeps
            ~0.05 → ~0.001 over the run. For a much smaller budget, raise the
            cooling (e.g. ~0.998 for 5k) so the walk actually reaches cold.
        skip_c5, skip_c6 : bool
            Force 5-/6-cycle steering off in Stage 3 regardless of the target
            count, dropping that cycle size's per-swap delta and loss term.
        adaptive_weights : bool
            If True, Stage 3 scales each loss term's weight linearly by its own
            current error, with a high fixed multiplier (``weight = base_weight
            * ADAPTIVE_WEIGHT_SCALE * error``) instead of a fixed weight, so
            terms further from target are pushed harder. See ``stage3.refine``.
        convergence_log : Path or str, optional
            If given, write per-metric error CSV during Stage 3 rewiring
            (see ``stage3.CONVERGENCE_LOG_INTERVAL`` for the row interval).
        swap_log : Path or str, optional
            If given, write one CSV row per evaluated Stage-3 swap proposal
            (per-motif deltas, Δloss, accept decision — see ``stage3.refine``).
        checkpoint_steps : list of int, optional
            Stage-3 loop indices at which to snapshot the walk's current graph
            and invoke ``checkpoint_callback`` with it (``0`` = the post-Stage-2
            graph, before any rewiring). See ``stage3.refine`` for exact
            semantics. Ignored if ``checkpoint_callback`` is ``None``.
        checkpoint_callback : callable, optional
            ``(step: int, graph: igraph.Graph) -> None``, called once per step
            in ``checkpoint_steps``. See ``stage3.refine``.

        Returns
        -------
        igraph.Graph
            Synthetic KG with the same vertex/edge attribute schema as a
            graph loaded by kg_io.load_kg.
        """
        log.info("Generator: sampling synthetic KG (master seed=%d)", seed)
        g = self.sample_pre_refine(seed=seed, relation_zipf_exponent=relation_zipf_exponent)
        g_refined = refine(
            g,
            self.target.e,
            target_f=self.target.f,
            budget=rewire_budget,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            seed=seed + 2,
            skip_c5=skip_c5,
            skip_c6=skip_c6,
            adaptive_weights=adaptive_weights,
            convergence_log=convergence_log,
            swap_log=swap_log,
            checkpoint_steps=checkpoint_steps,
            checkpoint_callback=checkpoint_callback,
        )
        log.info(
            "Generator: done — synthetic KG V=%d, E=%d",
            g_refined.vcount(),
            g_refined.ecount(),
        )
        return g_refined
