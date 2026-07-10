"""Reduced (non-over-determined) graph signature.

Measures a KG's statistical signature and stores a **compact distribution
summary** for each quantity — a quantile function for sample distributions,
or the parameters of a parametric family (exponential-decay, truncated
power-law, …) — instead of redundant moments, dropping every value guaranteed
by the stored summary. See ``docs/signature.md`` for the design and
``docs/notes/signature_measurement_plan.md`` for the mapping onto blocks.

Scope: Blocks A, B, C, D, E, F (G0–G5).
"""

import contextlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ._block_base import SignatureBlock
from .block_a import BlockA
from .block_b import BlockB
from .block_c import BlockC
from .block_d import BlockD
from .block_e import BlockE
from .block_f import BlockF
from ._fits import (
    QuantileFit,
    QUANTILE_LEVELS,
    ExpDecayFit,
    TruncPowerLawFit,
    ZipfFit,
    fit_quantiles,
    fit_exp_decay_rank,
    fit_truncated_powerlaw,
    fit_zipf,
    fit_cs_size_offset,
)

_ALL_BLOCKS: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")
_BLOCK_CLASSES: dict[str, type] = {
    "a": BlockA,
    "b": BlockB,
    "c": BlockC,
    "d": BlockD,
    "e": BlockE,
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
    e: Optional[BlockE] = field(default=None)
    f: Optional[BlockF] = field(default=None)

    def _blocks(self) -> list:
        return [self.a, self.b, self.c, self.d, self.e, self.f]

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
            all reduced blocks (``["a", "b", "c", "d", "e", "f"]``). Skipped blocks
            appear as NaN in ``ReducedGraphSignature.as_vector()``.
        verbose: Print progress to stdout.

    Returns:
        A ``ReducedGraphSignature`` with the requested blocks populated.
    """
    from ..kg_io import load_kg

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
        "e": "Block E (motifs & templates)",
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
        e=computed.get("e"),
        f=computed.get("f"),
    )


def write_signature_outputs(
    sig: ReducedGraphSignature,
    out_dir: Path | str,
    source: str,
    fmt: str = "png",
    show: bool = False,
) -> list[Path]:
    """Write a signature's plots, per-block JSON, summary and combined JSON to a directory.

    Mirrors the on-disk layout produced for measured graphs (``block_<x>.<fmt>``,
    ``block_<x>.json``, ``summary.txt``, ``signature.json``) so a measured
    ``signature/`` and a generated ``signature_synth/`` directory are structurally
    identical. Blocks that are ``None`` are skipped.

    :param sig: The computed reduced signature to persist.
    :param out_dir: Destination directory (created if missing).
    :param source: Value stored under ``"source"`` in ``signature.json``.
    :param fmt: Image format for the per-block plots (``png``/``pdf``/``svg``).
    :param show: If true, also display each block's plot interactively.
    :returns: The list of written file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    computed = [(c, b) for c, b in zip(_ALL_BLOCKS, sig._blocks()) if b is not None]
    written: list[Path] = []

    # One plot per computed block.
    for label, block in computed:
        plot_path = out_dir / f"block_{label}.{fmt}"
        block.visualize(mode="plot", path=str(plot_path))
        written.append(plot_path)
        if show:
            block.visualize(mode="plot")

    # Combined text summary (each block's text visualization).
    sections: list[str] = []
    for _label, block in computed:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            block.visualize(mode="text", path=None)
        sections.append(buf.getvalue().rstrip())
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n\n".join(sections) + "\n")
    written.append(summary_path)

    # Combined named-feature JSON (key:value, NaN-filled for absent blocks).
    json_path = out_dir / "signature.json"
    json_path.write_text(json.dumps({"source": str(source), "features": sig.as_dict()}, indent=2))
    written.append(json_path)

    # Each block's full internal state for later reconstruction.
    for label, block in computed:
        block_path = out_dir / f"block_{label}.json"
        block_path.write_text(json.dumps(block.to_serializable(), indent=2))
        written.append(block_path)

    return written


__all__ = [
    "BlockA", "BlockB", "BlockC", "BlockD", "BlockE", "BlockF",
    "ReducedGraphSignature", "compute_reduced_signature", "write_signature_outputs",
    "_ALL_BLOCKS",
    "QuantileFit", "QUANTILE_LEVELS", "ExpDecayFit", "TruncPowerLawFit", "ZipfFit",
    "fit_quantiles", "fit_exp_decay_rank", "fit_truncated_powerlaw",
    "fit_zipf", "fit_cs_size_offset",
]
