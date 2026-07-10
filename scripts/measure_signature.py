"""Measure the graph signature of a KG file and write results to an output directory."""

import argparse
import logging
from pathlib import Path
from kgsynth.signature import (
    compute_reduced_signature as compute_signature,
    write_signature_outputs,
    _ALL_BLOCKS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kg_file", help="Path to the input KG (.ttl or .nt)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write results into (default: a 'signature/' dir next to the graph file, "
             "i.e. data/graphs/<name>/signature/)",
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
        help=f"Comma-separated list of blocks to compute (default: all). "
             f"Example: --blocks a,b,f",
    )
    args = parser.parse_args()

    selected_blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]

    # Default: write a 'signature/' directory next to the graph file, matching the
    # data/graphs/<name>/ corpus layout (so data/graphs/aids/AIDS.nt →
    # data/graphs/aids/signature/). Override with --output-dir.
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.kg_file).parent / "signature"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading  : {args.kg_file}")
    sig = compute_signature(args.kg_file, verbose=True, blocks=selected_blocks)

    written = write_signature_outputs(
        sig, out_dir, source=str(args.kg_file), fmt=args.fmt, show=args.show
    )
    for path in written:
        print(f"  Saved  : {path}")

    print()
    print(f"Done. {len(written)} files written to {out_dir}/")
    print(f"Feature count: {len(sig.as_dict())}")


if __name__ == "__main__":
    main()
