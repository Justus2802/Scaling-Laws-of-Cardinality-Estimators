"""Graph signature measurement for KGs loaded via kg_io.load_kg."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from ._logging import get_logger
from ._utils import RDF_TYPE, MIN_SAMPLES_FOR_FIT, PowerLawStats
from .block_a import BlockA
from .block_b import BlockB
from .block_c import BlockC
from .block_d import BlockD, _TOP_K_PAIRS
from .block_e import BlockE, _SAMPLE_BUDGET
from .block_f import BlockF, _SAMPLE_K, _N_BOOTSTRAP

_ALL_BLOCKS: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")
_BLOCK_NA_VEC: dict[str, type] = {
    "a": BlockA,
    "b": BlockB,
    "c": BlockC,
    "d": BlockD,
    "e": BlockE,
    "f": BlockF,
}


@dataclass
class GraphSignature:
    """All six measurement blocks for a single KG.

    Blocks that were not computed are ``None``; ``as_vector()`` fills their
    positions with NaN values so the vector length is always fixed.
    """
    a: Optional[BlockA] = field(default=None)
    b: Optional[BlockB] = field(default=None)
    c: Optional[BlockC] = field(default=None)
    d: Optional[BlockD] = field(default=None)
    e: Optional[BlockE] = field(default=None)
    f: Optional[BlockF] = field(default=None)

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length vector; uncomputed blocks are NaN-filled."""
        vec: list[float] = []
        for char, block in zip(_ALL_BLOCKS, (self.a, self.b, self.c, self.d, self.e, self.f)):
            if block is None:
                vec.extend(_BLOCK_NA_VEC[char].get_na_vec())
            else:
                vec.extend(block.as_vector())
        return vec


def compute_signature(
    path: str | Path,
    *,
    blocks: list[str] | None = None,
    sample_budget: int = _SAMPLE_BUDGET,
    sample_k: int = _SAMPLE_K,
    n_bootstrap: int = _N_BOOTSTRAP,
    verbose: bool = False,
) -> GraphSignature:
    """Load a .ttl or .nt file and compute the graph signature.

    Args:
        path: Path to the KG file (.ttl or .nt).
        blocks: Which blocks to compute, e.g. ``["a", "c", "f"]``.
            Defaults to all blocks (``["a", "b", "c", "d", "e", "f"]``).
            Skipped blocks appear as NaN in ``GraphSignature.as_vector()``.
        sample_budget: Passed to BlockE.
        sample_k: Passed to BlockF.
        n_bootstrap: Passed to BlockF.
        verbose: Print progress to stdout.
    """
    from kg_io import load_kg

    active = set(blocks) if blocks is not None else set(_ALL_BLOCKS)
    unknown = active - set(_ALL_BLOCKS)
    if unknown:
        raise ValueError(f"Unknown block(s): {sorted(unknown)}. Valid: {list(_ALL_BLOCKS)}")

    def _step(label: str) -> None:
        if verbose:
            print(f"  Computing {label} …", flush=True)

    _step("loading KG")
    g = load_kg(path)

    a: Optional[BlockA] = None
    if "a" in active:
        _step("Block A (size & density)")
        a = BlockA().calculate(g)

    b: Optional[BlockB] = None
    if "b" in active:
        _step("Block B (degree structure)")
        b = BlockB().calculate(g)

    c: Optional[BlockC] = None
    if "c" in active:
        _step("Block C (schema & co-occurrence)")
        c = BlockC().calculate(g)

    d: Optional[BlockD] = None
    if "d" in active:
        _step("Block D (characteristic sets)")
        d = BlockD().calculate(g)

    e: Optional[BlockE] = None
    if "e" in active:
        _step("Block E (motifs & structural patterns)")
        e = BlockE().calculate(g, sample_budget=sample_budget)

    f: Optional[BlockF] = None
    if "f" in active:
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
    "_ALL_BLOCKS",
]
