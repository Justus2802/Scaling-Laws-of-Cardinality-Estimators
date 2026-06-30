"""High-level API: target Signature + three-stage Generator orchestrator."""

from dataclasses import dataclass
from pathlib import Path

import igraph

from signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF

from ._logging import get_logger
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
        from kg_io import load_kg
        return cls.from_graph(load_kg(Path(path)))


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
        remeasure_interval: int = 2000,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.9999,
        convergence_log: "Path | str | None" = None,
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
        remeasure_interval : int
            Accepted-swap interval between full 4-node motif remeasurements in Stage 3.
        initial_temp, cooling_rate : float
            Simulated-annealing parameters for Stage 3.
        convergence_log : Path or str, optional
            If given, write per-metric error CSV during Stage 3 rewiring
            (see ``stage3.CONVERGENCE_LOG_INTERVAL`` for the row interval).

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
            remeasure_interval=remeasure_interval,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            seed=seed + 2,
            convergence_log=convergence_log,
        )
        log.info("Generator: done — synthetic KG V=%d, E=%d", g_refined.vcount(), g_refined.ecount())
        return g_refined
