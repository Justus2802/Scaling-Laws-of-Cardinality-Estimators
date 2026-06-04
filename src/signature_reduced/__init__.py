"""Reduced (non-over-determined) graph signature.

A coexisting alternative to the ``signature`` package: it measures the same KGs
but stores the **parameters of the distribution family** each quantity follows
(skew-normal, exponential-decay, truncated power-law, …) instead of redundant
moments, dropping every value guaranteed by the stored parameters. See
``docs/signature_redesign.md`` for the design and ``docs/signature_measurement_plan.md``
for the mapping onto blocks.

Scope: Blocks A, B, C, D, F (G0–G4). Block E (motifs, G5) is deferred. The shared
block infrastructure — the ``SignatureBlock`` ABC, JSON serialization and logging —
is reused from the ``signature`` package.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from signature._block_base import SignatureBlock
from .block_a import BlockA
from .block_b import BlockB
from .block_c import BlockC
from .block_d import BlockD
from .block_f import BlockF
from ._fits import (
    SkewNormFit,
    ExpDecayFit,
    TruncPowerLawFit,
    ZipfFit,
    fit_skewnorm,
    fit_exp_decay_rank,
    fit_truncated_powerlaw,
    fit_zipf,
    fit_cs_size_offset,
)

# Block E (motifs) is intentionally excluded from the reduced signature for now.
_ALL_BLOCKS: tuple[str, ...] = ("a", "b", "c", "d", "f")
_BLOCK_CLASSES: dict[str, type] = {
    "a": BlockA,
    "b": BlockB,
    "c": BlockC,
    "d": BlockD,
    "f": BlockF,
}


@dataclass
class ReducedGraphSignature:
    """The reduced measurement blocks for a single KG.

    Blocks that were not computed are ``None``; ``as_vector()`` fills their
    positions with NaN so the vector length is always fixed.
    """
    a: Optional[BlockA] = field(default=None)
    b: Optional[BlockB] = field(default=None)
    c: Optional[BlockC] = field(default=None)
    d: Optional[BlockD] = field(default=None)
    f: Optional[BlockF] = field(default=None)

    def _blocks(self) -> list:
        return [self.a, self.b, self.c, self.d, self.f]

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length vector; uncomputed blocks are NaN-filled."""
        vec: list[float] = []
        for char, block in zip(_ALL_BLOCKS, self._blocks()):
            if block is None:
                vec.extend(_BLOCK_CLASSES[char].get_na_vec())
            else:
                vec.extend(block.as_vector())
        return vec

    def as_dict(self) -> dict[str, float]:
        """Return named feature→value pairs; NaN-filled for uncomputed blocks."""
        result: dict[str, float] = {}
        for char, block in zip(_ALL_BLOCKS, self._blocks()):
            cls = _BLOCK_CLASSES[char]
            names = cls.feature_names()
            values = block.as_vector() if block is not None else cls.get_na_vec()
            result.update(zip(names, values))
        return result


def compute_reduced_signature(
    path: str | Path,
    *,
    blocks: list[str] | None = None,
    verbose: bool = False,
) -> ReducedGraphSignature:
    """Load a .ttl or .nt file and compute the reduced graph signature.

    Args:
        path: Path to the KG file (.ttl or .nt).
        blocks: Which blocks to compute, e.g. ``["a", "c", "f"]``. Defaults to
            all reduced blocks (``["a", "b", "c", "d", "f"]``). Skipped blocks
            appear as NaN in ``ReducedGraphSignature.as_vector()``.
        verbose: Print progress to stdout.

    Returns:
        A ``ReducedGraphSignature`` with the requested blocks populated.
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

    computed: dict[str, SignatureBlock] = {}
    labels = {
        "a": "Block A (size & vocabulary)",
        "b": "Block B (relation frequency & multiplicity)",
        "c": "Block C (schema & co-occurrence)",
        "d": "Block D (characteristic sets & two-step)",
        "f": "Block F (connectivity)",
    }
    for char in _ALL_BLOCKS:
        if char in active:
            _step(labels[char])
            computed[char] = _BLOCK_CLASSES[char]().calculate(g)

    return ReducedGraphSignature(
        a=computed.get("a"),
        b=computed.get("b"),
        c=computed.get("c"),
        d=computed.get("d"),
        f=computed.get("f"),
    )


__all__ = [
    "BlockA", "BlockB", "BlockC", "BlockD", "BlockF",
    "ReducedGraphSignature", "compute_reduced_signature",
    "_ALL_BLOCKS",
    "SkewNormFit", "ExpDecayFit", "TruncPowerLawFit", "ZipfFit",
    "fit_skewnorm", "fit_exp_decay_rank", "fit_truncated_powerlaw",
    "fit_zipf", "fit_cs_size_offset",
]
