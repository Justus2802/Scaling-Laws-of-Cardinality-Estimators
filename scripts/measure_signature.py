#!/usr/bin/env python3
"""Measure a KG file's signature (Blocks A-F) and write results to a directory.

A thin wrapper over ``kgsynth.signature`` for running without installing the
package; ``kgsynth measure`` is the equivalent installed CLI and shares the same
implementation. By default writes a ``signature/`` directory **next to the graph
file**, matching the ``data/graphs/<name>/`` corpus layout (so
``data/graphs/aids/AIDS.nt`` → ``data/graphs/aids/signature/``). Override with
``--output-dir``.

Re-measuring a subset (``--blocks e``) updates that block and folds it into the
existing ``signature.json`` / ``summary.txt``, leaving the other blocks intact.
"""

import argparse
import logging
from pathlib import Path

from kgsynth.signature import _ALL_BLOCKS, compute_reduced_signature, write_signature_outputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def measure(kg_file: Path, out_dir: Path | None = None, blocks: list[str] | None = None,
            fmt: str = "png", show: bool = False) -> list[Path]:
    """Measure one KG and write its signature outputs.

    :param kg_file: Path to the input KG (``.ttl``/``.nt``).
    :param out_dir: Destination (default: a ``signature/`` dir next to the graph).
    :param blocks: Block labels to compute (default: all).
    :param fmt: Image format for the block plots.
    :param show: Display each block's plot after saving.
    :returns: The written file paths.
    """
    out_dir = out_dir or kg_file.parent / "signature"
    sig = compute_reduced_signature(kg_file, verbose=True, blocks=blocks or list(_ALL_BLOCKS))
    return write_signature_outputs(sig, out_dir, source=str(kg_file), fmt=fmt, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kg_file", type=Path, help="Path to the input KG (.ttl or .nt)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write results into (default: a 'signature/' dir next to the graph file)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show each block's plot interactively after saving"
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "svg"], default="png", dest="fmt",
        help="Image format for block plots (default: png)",
    )
    parser.add_argument(
        "--blocks", default=",".join(_ALL_BLOCKS),
        help=f"Comma-separated list of blocks to compute (default: all reduced blocks "
             f"{list(_ALL_BLOCKS)}). Example: --blocks a,b,f",
    )
    args = parser.parse_args()

    print(f"Loading  : {args.kg_file}")
    written = measure(
        args.kg_file,
        out_dir=args.output_dir,
        blocks=[b.strip() for b in args.blocks.split(",") if b.strip()],
        fmt=args.fmt,
        show=args.show,
    )
    for path in written:
        print(f"  Saved  : {path}")
    print(f"\nDone. {len(written)} files written to {written[0].parent}/")


if __name__ == "__main__":
    main()
