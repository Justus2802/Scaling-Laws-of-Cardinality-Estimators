#!/usr/bin/env python3
"""Example 2 — generate a synthetic KG from a target signature, then compare.

Runs the full measure → generate → compare round-trip on one graph:

1. **measure** the target KG's reduced signature,
2. **generate** a synthetic KG that targets it (Stages 1/2/3), and
3. **compare** the synthetic graph's re-measured signature to the target,
   block by block, reporting each block's mean relative feature error.

This is the minimal, public-API version of ``scripts/signature_roundtrip.py``
(which adds convergence/swap logging, auto-named outputs, and CLI options).

Run it (from the repo root, after ``pip install -e .``)::

    python examples/generate_and_compare.py                     # bundled sample graph
    python examples/generate_and_compare.py path/to/kg.ttl      # any .ttl / .nt file
    python examples/generate_and_compare.py path/to/kg.ttl 7    # + a specific seed

A perfect match is not expected: Stage 3 steers motif/connectivity structure
under a fixed degree sequence, so some blocks converge tightly and others
(e.g. the path/tree templates) only approximately — see user_docs/generator.md.

**Runtime note.** This measures the target *and* re-measures the synthetic graph
at full fidelity, so it invokes Block E's colour-coding sampler twice; on a
few-thousand-vertex graph expect several minutes, dominated by that measurement
(not by generation). Point it at a smaller graph for a quicker demonstration.
"""

import math
import sys
from pathlib import Path

from kgsynth import Generator, Signature

_DEFAULT_KG = Path(__file__).resolve().parent.parent / "tests" / "graphs" / "test_generated.ttl"


def _block_rel_error(target_block, synth_block) -> float:
    """Mean relative error between two blocks' feature vectors.

    Each feature's relative error is ``|synth - target| / max(|target|, 1e-9)``;
    NaN features (a fit that did not converge on either graph) are skipped so
    they don't swamp the mean. Returns NaN if every feature was skipped.
    """
    errors = []
    for t, s in zip(target_block.as_vector(), synth_block.as_vector()):
        if math.isnan(t) or math.isnan(s):
            continue
        errors.append(abs(s - t) / max(abs(t), 1e-9))
    return sum(errors) / len(errors) if errors else float("nan")


def main() -> None:
    kg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_KG
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    if not kg_path.is_file():
        raise SystemExit(f"No such KG file: {kg_path}")

    print(f"1. Measuring target signature of {kg_path.name} …")
    target = Signature.from_file(kg_path)

    print(f"2. Generating a synthetic KG (seed={seed}) …")
    synth_graph = Generator(target).sample(seed=seed, rewire_budget=20_000)
    print(f"   → {synth_graph.vcount()} vertices, {synth_graph.ecount()} edges")

    print("3. Re-measuring the synthetic graph and comparing per block …")
    synth = Signature.from_graph(synth_graph)

    print(f"\n  {'Block':<28} {'mean rel. error':>16}")
    print("  " + "─" * 46)
    for label, tb, sb in [
        ("A size/vocabulary", target.a, synth.a),
        ("B relation freq/multiplicity", target.b, synth.b),
        ("C schema/co-occurrence", target.c, synth.c),
        ("D characteristic sets", target.d, synth.d),
        ("E motifs/templates", target.e, synth.e),
        ("F connectivity", target.f, synth.f),
    ]:
        err = _block_rel_error(tb, sb)
        print(f"  {label:<28} {err:>16.3f}")


if __name__ == "__main__":
    main()
