#!/usr/bin/env python3
"""Measure graph signatures of all raw KGs under ``graphs/data/*/raw/``.

By default runs the original signature with every block including the motif
block (Block E). With ``--reduced`` it runs the reduced (non-over-determined)
signature instead (Blocks A, B, C, D, F — no motifs), writing to
``sig_out_reduced/``. Yago is skipped (too large for available RAM). The
extensionless Freebase file ``graphs/data/raw/59621618`` is included via a
temporary ``.nt`` copy.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root (this file lives in scripts/).
ROOT = Path(__file__).resolve().parent.parent

PY = ROOT / ".venv" / "bin" / "python"
# Original signature has all six blocks; the reduced one has no motif block (E).
FULL_BLOCKS = "a,b,c,d,e,f"
REDUCED_BLOCKS = "a,b,c,d,f"

# Temp .nt copy so the loader (which needs .nt/.ttl) accepts the extensionless file.
FREEBASE_SRC = ROOT / "graphs/data/raw/59621618"
FREEBASE_NT = ROOT / "graphs/data/raw/59621618.nt"


def main() -> int:
    """Measure each graph in turn and print a success/failure summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduced", action="store_true",
        help="Measure the reduced signature (Blocks A,B,C,D,F → sig_out_reduced/) "
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

    tmp_made = False
    if FREEBASE_SRC.is_file() and not FREEBASE_NT.is_file():
        shutil.copy(FREEBASE_SRC, FREEBASE_NT)
        tmp_made = True

    # Smallest -> largest so quick wins land first.
    graphs = [
        ROOT / "graphs/data/fb237_v4/raw/fb237_v4.nt",
        FREEBASE_NT,
        ROOT / "graphs/data/aids/raw/AIDS.nt",
        ROOT / "graphs/data/codex_l/raw/codex_l.nt",
        ROOT / "graphs/data/lubm/raw/59410577.ttl",
        ROOT / "graphs/data/hetionet/raw/hetionet.nt",
    ]

    ok: list[str] = []
    fail: list[str] = []
    try:
        for g in graphs:
            print("=" * 60)
            print(f">>> Measuring: {g}")
            print("=" * 60)
            result = subprocess.run(
                [str(PY), str(script), str(g), "--blocks", blocks]
            )
            if result.returncode == 0:
                ok.append(str(g))
            else:
                fail.append(str(g))
                print(f"!!! FAILED: {g}")
    finally:
        # Clean up the temp copy we created.
        if tmp_made:
            FREEBASE_NT.unlink(missing_ok=True)

    print()
    print("=" * 60)
    print(f"SUMMARY ({kind} signature)")
    print(f"  Succeeded ({len(ok)}):")
    for g in ok:
        print(f"    - {g}")
    print(f"  Failed ({len(fail)}):")
    for g in fail:
        print(f"    - {g}")
    print("  Skipped: graphs/data/yago/raw/Yago.{nt,ttl} (too large for RAM)")
    print("=" * 60)

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
