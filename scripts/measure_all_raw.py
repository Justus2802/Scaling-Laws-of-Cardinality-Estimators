#!/usr/bin/env python3
"""Measure graph signatures for every graph in the ``data/graphs/`` corpus.

Each ``data/graphs/<name>/`` directory holds one graph file (``.nt``/``.ttl``)
plus its ``signature/`` output. The held-out test graphs live in the parallel
``data/test_graphs/`` corpus and are measured alongside the main corpus. By
default runs the reduced (non-over-determined) signature (Blocks A–F, incl. the
Block E motifs); pass ``--full`` for the original full signature. Either way the
``signature/`` directory is written **next to each graph file** (i.e.
``data/graphs/<name>/signature/``).

Pass ``--blocks`` to measure only a subset for every graph (e.g. ``--blocks c`` to
re-measure just Block C after a change); the labels are forwarded to the per-graph
measurement script unchanged.

Pass ``--graphs`` to restrict the run to specific graphs by directory name (e.g.
``--graphs aids fb237_v4``); the default measures every graph in both corpora.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Repo root (this file lives in scripts/).
ROOT = Path(__file__).resolve().parent.parent
# Main corpus plus the parallel held-out test corpus; both share the same layout.
CORPORA = [ROOT / "data" / "graphs", ROOT / "data" / "test_graphs"]

PY = ROOT / ".venv" / "bin" / "python"
# Both signatures now cover all six blocks (A–F, incl. the Block E motifs).
FULL_BLOCKS = "a,b,c,d,e,f"
REDUCED_BLOCKS = "a,b,c,d,e,f"

# Graph files carry one of these extensions; everything else (signature/, caches) is ignored.
GRAPH_SUFFIXES = {".nt", ".ttl"}


def discover_graphs(names: set[str] | None = None) -> list[Path]:
    """Return one graph file per ``<corpus>/<name>/`` dir, smallest first.

    Scans every corpus in :data:`CORPORA` (the main and test graph corpora).
    Each graph directory is expected to contain exactly one ``.nt``/``.ttl``
    file alongside its ``signature/`` output; the first match per directory is
    used. Sorted by file size so quick wins land first.

    :param names: If given, only directories whose name is in this set are
        returned; otherwise every discovered graph is returned.
    :returns: Graph file paths, smallest first.
    """
    graphs: list[Path] = []
    for corpus in CORPORA:
        if not corpus.is_dir():
            continue
        for d in sorted(corpus.iterdir()):
            if not d.is_dir():
                continue
            if names is not None and d.name not in names:
                continue
            files = sorted(p for p in d.iterdir()
                           if p.suffix in GRAPH_SUFFIXES and not p.stem.endswith("_synth"))
            if files:
                graphs.append(files[0])
    graphs.sort(key=lambda p: p.stat().st_size)
    return graphs


def main() -> int:
    """Measure each corpus graph in turn and print a success/failure summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Measure the original full signature instead of the (default) reduced "
             "signature. Both write to data/graphs/<name>/signature/.",
    )
    parser.add_argument(
        "--blocks", default=None,
        help="Comma-separated subset of blocks to compute for every graph "
             f"(default: all, '{FULL_BLOCKS}'). Example: --blocks c only re-measures Block C.",
    )
    parser.add_argument(
        "--graphs", nargs="+", default=None, metavar="NAME",
        help="Restrict the run to these graphs by directory name (e.g. --graphs "
             "aids fb237_v4). Default: every graph in data/graphs/ and data/test_graphs/.",
    )
    args = parser.parse_args()

    if args.full:
        script = ROOT / "scripts/measure_signature.py"
        blocks = FULL_BLOCKS
        kind = "full"
    else:
        script = ROOT / "scripts/measure_signature_reduced.py"
        blocks = REDUCED_BLOCKS
        kind = "reduced"

    # An explicit --blocks overrides the per-kind default and is passed straight
    # through to the per-graph measurement script (which validates the labels).
    if args.blocks is not None:
        blocks = args.blocks

    selected = set(args.graphs) if args.graphs is not None else None
    graphs = discover_graphs(selected)
    if not graphs:
        roots = " or ".join(str(c) for c in CORPORA)
        if selected is not None:
            print(f"No graphs matching {sorted(selected)} found under {roots}.")
        else:
            print(f"No graphs found under {roots} (expected <corpus>/<name>/<graph>.nt).")
        return 1

    # Warn about names that matched nothing so typos do not pass silently.
    if selected is not None:
        found = {g.parent.name for g in graphs}
        for missing in sorted(selected - found):
            print(f"!!! No graph directory named {missing!r}; skipping.")

    ok: list[str] = []
    fail: list[str] = []
    for g in graphs:
        print("=" * 60)
        print(f">>> Measuring: {g}")
        print("=" * 60)
        result = subprocess.run([str(PY), str(script), str(g), "--blocks", blocks])
        if result.returncode == 0:
            ok.append(str(g))
        else:
            fail.append(str(g))
            print(f"!!! FAILED: {g}")

    print()
    print("=" * 60)
    print(f"SUMMARY ({kind} signature, blocks={blocks})")
    print(f"  Succeeded ({len(ok)}):")
    for g in ok:
        print(f"    - {g}")
    print(f"  Failed ({len(fail)}):")
    for g in fail:
        print(f"    - {g}")
    print("=" * 60)

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
