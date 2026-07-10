"""Re-render block plots for all signature directories already collected under ./data.

Finds every ``block_<x>.json`` file under ``data/``, loads it via
``from_serializable``, and re-writes the corresponding ``block_<x>.png``.
Useful after changing visualize() or plot helpers without re-running measurements.

Usage::

    python scripts/rerender_signatures.py                  # all blocks
    python scripts/rerender_signatures.py --blocks b c d   # specific blocks
    python scripts/rerender_signatures.py --fmt pdf        # different format
    python scripts/rerender_signatures.py --dry-run        # list without writing
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from kgsynth.signature import _ALL_BLOCKS, _BLOCK_CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rerender")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks", nargs="+", default=list(_ALL_BLOCKS),
        choices=list(_ALL_BLOCKS), metavar="BLOCK",
        help="Which blocks to re-render (default: all)",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Root directory to search for signature dirs (default: data/)",
    )
    parser.add_argument(
        "--fmt", default="png", choices=["png", "pdf", "svg"],
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without writing anything",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.exists():
        log.error("Data directory %s does not exist", data_root)
        sys.exit(1)

    wanted = set(args.blocks)
    total = rendered = skipped = errors = 0

    # Walk every block_<x>.json under data_root.
    for json_path in sorted(data_root.rglob("block_[abcdef].json")):
        label = json_path.stem.split("_")[1]   # "block_b.json" → "b"
        if label not in wanted:
            continue

        total += 1
        out_path = json_path.with_suffix(f".{args.fmt}")

        if args.dry_run:
            log.info("would render %s → %s", json_path, out_path.name)
            continue

        try:
            data = json.loads(json_path.read_text())
            block = _BLOCK_CLASSES[label].from_serializable(data)
            block.visualize(mode="plot", path=str(out_path))
            log.info("rendered %s", out_path)
            rendered += 1
        except Exception as exc:
            log.warning("SKIP %s: %s", json_path, exc)
            skipped += 1
            errors += 1

    if not args.dry_run:
        log.info("done — %d/%d rendered, %d errors", rendered, total, errors)
    else:
        log.info("dry-run — %d files would be rendered", total)


if __name__ == "__main__":
    main()
