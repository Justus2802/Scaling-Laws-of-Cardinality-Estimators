"""Graph signature measurement for KGs loaded via kg_io.load_kg."""

from pathlib import Path
from dataclasses import dataclass

from ._logging import get_logger
from ._utils import RDF_TYPE, MIN_SAMPLES_FOR_FIT, PowerLawStats
from .block_a import BlockA
from .block_b import BlockB
from .block_c import BlockC
from .block_d import BlockD, _TOP_K_PAIRS
from .block_e import BlockE, _SAMPLE_BUDGET
from .block_f import BlockF, _SAMPLE_K, _N_BOOTSTRAP


@dataclass
class GraphSignature:
    """All six measurement blocks for a single KG."""
    a: BlockA
    b: BlockB
    c: BlockC
    d: BlockD
    e: BlockE
    f: BlockF

    def as_vector(self) -> list[float]:
        vec: list[float] = []
        for block in (self.a, self.b, self.c, self.d, self.e, self.f):
            vec.extend(block.as_vector())
        return vec


def compute_signature(
    path: str | Path,
    *,
    sample_budget: int = _SAMPLE_BUDGET,
    sample_k: int = _SAMPLE_K,
    n_bootstrap: int = _N_BOOTSTRAP,
    verbose: bool = False,
) -> GraphSignature:
    """Load a .ttl or .nt file and compute its full graph signature."""
    from kg_io import load_kg

    def _step(label: str) -> None:
        if verbose:
            print(f"  Computing {label} …", flush=True)

    _step("loading KG")
    g = load_kg(path)
    _step("Block A (size & density)")
    a = BlockA().calculate(g)
    _step("Block B (degree structure)")
    b = BlockB().calculate(g)
    _step("Block C (schema & co-occurrence)")
    c = BlockC().calculate(g)
    _step("Block D (characteristic sets)")
    d = BlockD().calculate(g)
    _step("Block E (motifs & structural patterns)")
    e = BlockE().calculate(g, sample_budget=sample_budget)
    _step("Block F (connectivity)")
    f = BlockF().calculate(g, sample_k=sample_k, n_bootstrap=n_bootstrap)
    return GraphSignature(a=a, b=b, c=c, d=d, e=e, f=f)


__all__ = [
    "get_logger",
    "RDF_TYPE", "MIN_SAMPLES_FOR_FIT", "PowerLawStats",
    "BlockA",
    "BlockB",
    "BlockC",
    "BlockD", "_TOP_K_PAIRS",
    "BlockE", "_SAMPLE_BUDGET",
    "BlockF", "_SAMPLE_K", "_N_BOOTSTRAP",
    "GraphSignature", "compute_signature",
]
