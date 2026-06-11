#!/usr/bin/env python3
"""Measure graph signatures for every graph in the ``data/graphs/`` corpus.

Each ``data/graphs/<name>/`` directory holds one graph file (``.nt``/``.ttl``)
plus its ``signature/`` output. By default runs the original full signature
(all blocks incl. motifs, Block E) and writes to ``sig_out/``. With ``--reduced``
it runs the reduced (non-over-determined) signature (Blocks A, B, C, D, F — no
motifs), which writes its ``signature/`` directory **next to each graph file**
(i.e. ``data/graphs/<name>/signature/``).
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Repo root (this file lives in scripts/).
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "graphs"

PY = ROOT / ".venv" / "bin" / "python"
# Original signature has all six blocks; the reduced one has no motif block (E).
FULL_BLOCKS = "a,b,c,d,e,f"
REDUCED_BLOCKS = "a,b,c,d,f"

# Graph files carry one of these extensions; everything else (signature/, caches) is ignored.
GRAPH_SUFFIXES = {".nt", ".ttl"}


def discover_graphs() -> list[Path]:
    """Return the graph file in each ``data/graphs/<name>/`` dir, smallest first.

    Each corpus directory is expected to contain exactly one ``.nt``/``.ttl``
    file alongside its ``signature/`` output; the first match per directory is
    used. Sorted by file size so quick wins land first.
    """
    graphs: list[Path] = []
    for d in sorted(CORPUS.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(p for p in d.iterdir() if p.suffix in GRAPH_SUFFIXES)
        if files:
            graphs.append(files[0])
    graphs.sort(key=lambda p: p.stat().st_size)
    return graphs


def main() -> int:
    """Measure each corpus graph in turn and print a success/failure summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduced", action="store_true",
        help="Measure the reduced signature (Blocks A,B,C,D,F → data/graphs/<name>/signature/) "
             "instead of the original full signature (all blocks → sig_out/).",
    )
    args = parser.parse_args()

    if args.reduced:
        script = ROOT / "scripts/measure_signature_reduced.py"
        blocks = REDUCED_BLOCKS
        kind = "reduced"
    else:
        script = ROOT / "scripts/measure_signature.py"
        blocks = FULL_BLOCKS
        kind = "full"

    graphs = discover_graphs()
    if not graphs:
        print(f"No graphs found under {CORPUS}/ (expected data/graphs/<name>/<graph>.nt).")
        return 1

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
    print(f"SUMMARY ({kind} signature)")
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
