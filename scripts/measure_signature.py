"""Measure a KG file's signature (Blocks A-F) and write results to a directory.

By default writes a ``signature/`` directory **next to the graph file**,
matching the ``data/graphs/<name>/`` corpus layout (so
``data/graphs/aids/AIDS.nt`` → ``data/graphs/aids/signature/``). Override
with ``--output-dir``. Prefer the installed ``kgsynth measure`` CLI for
everyday use; this script remains for direct invocation without installing
the package.
"""

import argparse
import contextlib
import io
import json
import logging
from pathlib import Path
from kgsynth.signature import compute_reduced_signature, _ALL_BLOCKS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kg_file", help="Path to the input KG (.ttl or .nt)")
    parser.add_argument(
        "--output-dir", default=None,
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

    selected_blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]

    # Default: write a 'signature/' directory next to the graph file, matching the
    # data/graphs/<name>/ corpus layout. Override with --output-dir.
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.kg_file).parent / "signature"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading  : {args.kg_file}")
    sig = compute_reduced_signature(args.kg_file, verbose=True, blocks=selected_blocks)

    all_blocks = [("a", sig.a), ("b", sig.b), ("c", sig.c),
                  ("d", sig.d), ("e", sig.e), ("f", sig.f)]
    computed_blocks = [(label, block) for label, block in all_blocks if block is not None]
    written: list[Path] = []

    # Save one plot per computed block.
    for label, block in computed_blocks:
        plot_path = out_dir / f"block_{label}.{args.fmt}"
        block.visualize(mode="plot", path=str(plot_path))
        written.append(plot_path)
        print(f"  Saved  : {plot_path}")
        if args.show:
            block.visualize(mode="plot")

    # Write combined text summary.
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

    # Write combined named-feature JSON (key:value, no raw array).
    json_path = out_dir / "signature.json"
    named = sig.as_dict()
    json_path.write_text(
        json.dumps({"source": str(args.kg_file), "features": named}, indent=2)
    )
    written.append(json_path)
    print(f"  Saved  : {json_path}")

    # Serialize each block's full internal state for later reconstruction.
    for label, block in computed_blocks:
        block_path = out_dir / f"block_{label}.json"
        block_path.write_text(json.dumps(block.to_serializable(), indent=2))
        written.append(block_path)
        print(f"  Saved  : {block_path}")

    print()
    print(f"Done. {len(written)} files written to {out_dir}/")
    print(f"Feature count: {len(named)}")


if __name__ == "__main__":
    main()
