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
    """

    a: "BlockA"
    c: "BlockC"
    e: "BlockE"
    b: "BlockB | None" = None   # optional: enables multi-object edges + data-driven PA
    d: "BlockD | None" = None   # optional: enables CS template reuse
    f: "BlockF | None" = None   # optional: enables assortativity targeting in Stage 3

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
        by re-saving a measured/cached signature. ``a``, ``c`` and ``e`` are
        required; ``b``, ``d``, ``f`` are optional and default to ``None`` if
        their key is absent, matching :class:`Signature`'s own optionality.

        :param path: Path to the YAML config file.
        :returns: The reconstructed target ``Signature``.
        :raises KeyError: If a required block (``a``, ``c``, ``e``) is missing
            (also raised for an empty or comment-only file, which parses to
            no data at all).
        """
        data = yaml.safe_load(Path(path).read_text()) or {}
        blocks: dict = {}
        for letter, block_cls in _BLOCK_CLASSES.items():
            if letter in data:
                blocks[letter] = block_cls.from_serializable(data[letter])
            elif letter in ("a", "c", "e"):
                raise KeyError(f"Signature config {path} is missing required block {letter!r}")
            else:
                blocks[letter] = None
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

    def _blocks(self) -> list:
        """Return the six blocks in signature order (``a``..``f``), ``None`` where optional."""
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
        schema = sample_schema(
            self.target.a,
            self.target.c,
            d=self.target.d,
            b=self.target.b,
            f=self.target.f,
            relation_zipf_exponent=relation_zipf_exponent,
            seed=seed,
        )
        g = instantiate(schema, seed=seed + 1)
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
