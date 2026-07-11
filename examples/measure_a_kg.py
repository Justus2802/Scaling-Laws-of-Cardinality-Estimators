#!/usr/bin/env python3
"""Example 1 — measure a real knowledge graph's statistical signature.

Reduces a KG file to its 124-feature *reduced signature* (Blocks A–F) and prints
a per-block summary. This is the "measure" step of the measure → generate →
compare loop; the resulting :class:`~kgsynth.Signature` is exactly what
``examples/generate_and_compare.py`` feeds to the generator.

Run it (from the repo root, after ``pip install -e .``)::

    python examples/measure_a_kg.py                       # bundled sample graph
    python examples/measure_a_kg.py path/to/your_kg.ttl   # any .ttl / .nt file

The equivalent one-liner on the CLI, which also writes plots + JSON next to the
graph, is ``kgsynth measure <file>``.

**Runtime note.** This measures the *full-fidelity* signature: Block E's motif
and path/tree-template counts use colour-coding sampling (100k walks by default),
so on a few-thousand-vertex graph the whole measurement takes several minutes —
Block E dominates. That is the genuine cost of measuring a graph once; the
tracked corpus was measured this way offline and its signatures cached under
``data/graphs/<name>/signature/`` for instant reuse.
"""

import sys
from pathlib import Path

from kgsynth import Signature

# A small graph that ships with the repo, so the example runs with no arguments.
_DEFAULT_KG = Path(__file__).resolve().parent.parent / "tests" / "graphs" / "test_generated.ttl"


def main() -> None:
    kg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_KG
    if not kg_path.is_file():
        raise SystemExit(f"No such KG file: {kg_path}")

    print(f"Measuring {kg_path.name} …")
    # Signature.from_file loads the graph and computes all six reduced blocks.
    sig = Signature.from_file(kg_path)

    # Each block exposes a named-feature dict via as_dict(); print a compact
    # per-block view of the signature that drives generation.
    blocks = [("A size/vocabulary", sig.a), ("B relation freq/multiplicity", sig.b),
              ("C schema/co-occurrence", sig.c), ("D characteristic sets", sig.d),
              ("E motifs/templates", sig.e), ("F connectivity", sig.f)]
    for label, block in blocks:
        print(f"\n── Block {label} ──")
        for name, value in block.as_dict().items():
            print(f"  {name:<32} {value:>14.4f}")

    total = sum(len(b.as_dict()) for _, b in blocks)
    print(f"\n{total} features measured.")


if __name__ == "__main__":
    main()
