"""Measure the graph signature of a KG file and write results to an output directory."""

import argparse
import contextlib
import io
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signature import compute_signature, _ALL_BLOCKS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kg_file", help="Path to the input KG (.ttl or .nt)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write results into (default: sig_out/<graph name>_signature/)",
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

    graph_name = Path(args.kg_file).stem
    out_dir = Path(args.output_dir) if args.output_dir else Path("sig_out") / f"{graph_name}_signature"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading  : {args.kg_file}")
    sig = compute_signature(args.kg_file, verbose=True, blocks=selected_blocks)

    all_blocks = [("a", sig.a), ("b", sig.b), ("c", sig.c), ("d", sig.d), ("e", sig.e), ("f", sig.f)]
    computed_blocks = [(label, block) for label, block in all_blocks if block is not None]
    written: list[Path] = []

    # Save one plot per computed block
    for label, block in computed_blocks:
        plot_path = out_dir / f"block_{label}.{args.fmt}"
        block.visualize(mode="plot", path=str(plot_path))
        written.append(plot_path)
        print(f"  Saved  : {plot_path}")
        if args.show:
            block.visualize(mode="plot")

    # Write combined text summary
    summary_path = out_dir / "summary.txt"
    sections: list[str] = []
    for _label, block in computed_blocks:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            block.visualize(mode="text", path=None)
        sections.append(buf.getvalue().rstrip())
    summary_path.write_text("\n\n".join(sections) + "\n")
    written.append(summary_path)
    print(f"  Saved  : {summary_path}")

    # Write JSON vector
    json_path = out_dir / "signature.json"
    vector = sig.as_vector()
    json_path.write_text(json.dumps({"source": str(args.kg_file), "vector": vector}, indent=2))
    written.append(json_path)
    print(f"  Saved  : {json_path}")

    print()
    print(f"Done. {len(written)} files written to {out_dir}/")
    print(f"Vector length: {len(vector)} features")


if __name__ == "__main__":
    main()
