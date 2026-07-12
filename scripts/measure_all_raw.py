#!/usr/bin/env python3
"""Measure graph signatures for every graph in the corpus.

Each ``data/graphs/<name>/`` directory holds one graph file (``.nt``/``.ttl``) plus
its ``signature/`` output. The held-out test graphs live in the parallel
``data/test_graphs/`` corpus and are measured alongside the main corpus. Runs the
full signature (Blocks A–F, incl. the Block E motifs) in-process via
``kgsynth.signature``; the ``signature/`` directory is written **next to each graph
file** (i.e. ``data/graphs/<name>/signature/``).

``--blocks`` measures only a subset for every graph (e.g. ``--blocks e`` to backfill
Block E after a change). Subset runs fold into each graph's existing
``signature.json`` / ``summary.txt`` rather than replacing them, so the other blocks
survive. ``--skip-measured`` then skips graphs whose selected blocks are all already
on disk, making a long backfill re-runnable.

``--graphs`` restricts the run to specific graphs by directory name.
"""

import argparse
import sys
import traceback
from pathlib import Path

from kgsynth.corpus import DEFAULT_SEARCH_DIRS, iter_corpus_graphs
from kgsynth.signature import _ALL_BLOCKS

from measure_signature import measure

ALL_BLOCKS = ",".join(_ALL_BLOCKS)


def _already_measured(graph: Path, blocks: list[str]) -> bool:
    """True when every selected block already has a ``block_<x>.json`` next to ``graph``."""
    sig_dir = graph.parent / "signature"
    return all((sig_dir / f"block_{b}.json").exists() for b in blocks)


def main() -> int:
    """Measure each corpus graph in turn and print a success/failure summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks", default=ALL_BLOCKS,
        help=f"Comma-separated subset of blocks to compute for every graph "
             f"(default: all, '{ALL_BLOCKS}'). Example: --blocks e only re-measures Block E.",
    )
    parser.add_argument(
        "--graphs", nargs="+", default=None, metavar="NAME",
        help="Restrict the run to these graphs by directory name (e.g. --graphs "
             "aids fb237_v4). Default: every graph in data/graphs/ and data/test_graphs/.",
    )
    parser.add_argument(
        "--skip-measured", action="store_true",
        help="Skip graphs whose selected blocks are all already measured (re-runnable backfill).",
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "svg"], default="png", dest="fmt",
        help="Image format for block plots (default: png)",
    )
    args = parser.parse_args()

    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    selected = set(args.graphs) if args.graphs is not None else None
    graphs = iter_corpus_graphs(selected)
    if not graphs:
        roots = " or ".join(str(c) for c in DEFAULT_SEARCH_DIRS)
        target = f"matching {sorted(selected)} " if selected is not None else ""
        print(f"No graphs {target}found under {roots} (expected <corpus>/<name>/<graph>.nt).")
        return 1

    # Warn about names that matched nothing so typos do not pass silently.
    if selected is not None:
        for missing in sorted(selected - {g.parent.name for g in graphs}):
            print(f"!!! No graph directory named {missing!r}; skipping.")

    ok: list[str] = []
    fail: list[str] = []
    skipped: list[str] = []
    for g in graphs:
        if args.skip_measured and _already_measured(g, blocks):
            print(f"--- Skipping {g} (blocks {args.blocks} already measured)")
            skipped.append(str(g))
            continue

        print("=" * 60)
        print(f">>> Measuring: {g}")
        print("=" * 60)
        try:
            measure(g, blocks=blocks, fmt=args.fmt)
            ok.append(str(g))
        except Exception:  # one bad graph must not abort a long corpus run
            traceback.print_exc()
            fail.append(str(g))
            print(f"!!! FAILED: {g}")

    print()
    print("=" * 60)
    print(f"SUMMARY (blocks={args.blocks})")
    for label, group in (("Succeeded", ok), ("Skipped", skipped), ("Failed", fail)):
        print(f"  {label} ({len(group)}):")
        for g in group:
            print(f"    - {g}")
    print("=" * 60)

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
