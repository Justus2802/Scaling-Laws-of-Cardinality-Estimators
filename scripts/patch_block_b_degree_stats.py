"""Patch existing measured signatures with the new BlockB degree-stat fields.

Adds ``out_degree_max``, ``out_degree_p90``, ``in_degree_max``, ``in_degree_p90``
to every ``block_b.json`` and ``signature.json`` found under the given roots,
computing them from the ``_out_degrees`` / ``_in_degrees`` arrays that are already
stored in each ``block_b.json``.

Usage
-----
    python scripts/patch_block_b_degree_stats.py
    python scripts/patch_block_b_degree_stats.py --roots data/graphs data/test_graphs
    python scripts/patch_block_b_degree_stats.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
from kgsynth.signature import BlockB  # noqa: E402
from kgsynth.signature._block_base import _NOT_CALCULATED  # noqa: E402

_NEW_FIELDS = ("out_degree_max", "out_degree_p90", "in_degree_max", "in_degree_p90")
_DEFAULT_ROOTS = [_REPO / "data" / "graphs", _REPO / "data" / "test_graphs"]


def _compute_stats(b: BlockB) -> dict[str, float] | None:
    """Return the 4 new scalars from the block's stored degree arrays, or None if unavailable."""
    out_deg = b._out_degrees
    in_deg = b._in_degrees
    if out_deg is _NOT_CALCULATED or in_deg is _NOT_CALCULATED:
        return None
    if not isinstance(out_deg, np.ndarray) or not isinstance(in_deg, np.ndarray):
        return None
    return {
        "out_degree_max": int(out_deg.max()) if out_deg.size else 0,
        "out_degree_p90": float(np.percentile(out_deg, 90)) if out_deg.size else 0.0,
        "in_degree_max": int(in_deg.max()) if in_deg.size else 0,
        "in_degree_p90": float(np.percentile(in_deg, 90)) if in_deg.size else 0.0,
    }


def patch_directory(sig_dir: Path, dry_run: bool) -> bool:
    """Patch block_b.json and signature.json in one signature directory.

    Returns True if any file was updated.
    """
    block_b_path = sig_dir / "block_b.json"
    sig_path = sig_dir / "signature.json"

    if not block_b_path.exists():
        return False

    raw = json.loads(block_b_path.read_text())
    b = BlockB.from_serializable(raw)

    stats = _compute_stats(b)
    if stats is None:
        print(f"  SKIP {sig_dir} — degree arrays not found in block_b.json")
        return False

    already_done = all(f"_{k}" in raw for k in _NEW_FIELDS)
    if already_done:
        print(f"  OK   {sig_dir} — already patched")
        return False

    print(f"  PATCH {sig_dir}")
    for k, v in stats.items():
        print(f"         {k} = {v}")

    if dry_run:
        return True

    # Update block_b.json: re-serialize via the block (picks up new fields)
    b._out_degree_max = stats["out_degree_max"]
    b._out_degree_p90 = stats["out_degree_p90"]
    b._in_degree_max = stats["in_degree_max"]
    b._in_degree_p90 = stats["in_degree_p90"]
    block_b_path.write_text(json.dumps(b.to_serializable(), indent=2))

    # Update signature.json: inject the 4 new feature keys
    if sig_path.exists():
        sig_data = json.loads(sig_path.read_text())
        sig_data["features"].update(stats)
        sig_path.write_text(json.dumps(sig_data, indent=2))

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", type=Path, default=_DEFAULT_ROOTS,
                        help="Root directories to search for signature/ subdirectories")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be patched without writing anything")
    args = parser.parse_args()

    sig_dirs: list[Path] = []
    for root in args.roots:
        if not root.exists():
            print(f"Warning: root not found: {root}")
            continue
        sig_dirs.extend(sorted(root.rglob("signature.json")))

    if not sig_dirs:
        sys.exit("No signature.json files found.")

    n_patched = 0
    for sig_json in sig_dirs:
        sig_dir = sig_json.parent
        if patch_directory(sig_dir, args.dry_run):
            n_patched += 1

    verb = "Would patch" if args.dry_run else "Patched"
    print(f"\n{verb} {n_patched}/{len(sig_dirs)} directories.")


if __name__ == "__main__":
    main()
