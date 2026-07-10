"""Sweep Stage 3 configs (rewire_budget × seed) and save per-run synthetic
signatures to disk for later error analysis.

Output layout
-------------
  experiments/<graph>_target.json   — target signature (written once)
  experiments/<graph>.jsonl         — one JSON record per run:
      {"graph": ..., "budget": ..., "seed": ...,
       "synth": {"a": ..., "b": ..., "c": ..., "d": ..., "e": ..., "f": ...}}

Blocks that are None are stored as null.

Usage
-----
    python scripts/sweep_collect.py fb237_v4_ind
    python scripts/sweep_collect.py fb237_v4_ind \\
        --budgets 500 2000 5000 --seeds 0 1 2 3 4
    python scripts/sweep_collect.py fb237_v4_ind --append
"""

import argparse
import json
import logging
from itertools import product
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
from kgsynth.generator import Generator, Signature  # noqa: E402
from kgsynth.signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF  # noqa: E402
from kgsynth.corpus import DEFAULT_SEARCH_DIRS, load_target_from_corpus

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_LETTERS = ("a", "b", "c", "d", "e", "f")
_BLOCK_CLS = {"a": BlockA, "b": BlockB, "c": BlockC, "d": BlockD, "e": BlockE, "f": BlockF}


def _sig_to_dict(sig: Signature) -> dict:
    """Serialize a Signature to a plain dict using each block's to_serializable()."""
    out: dict = {}
    for letter in _LETTERS:
        blk = getattr(sig, letter, None)
        out[letter] = blk.to_serializable() if blk is not None else None
    return out


def _sig_from_dict(data: dict) -> Signature:
    """Reconstruct a Signature from a serialized dict."""
    blocks: dict = {}
    for letter, cls in _BLOCK_CLS.items():
        raw = data.get(letter)
        blocks[letter] = cls.from_serializable(raw) if raw is not None else None
    return Signature(
        a=blocks["a"], b=blocks["b"], c=blocks["c"],
        d=blocks["d"], e=blocks["e"], f=blocks["f"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", help="Graph name in the corpus (e.g. fb237_v4_ind)")
    parser.add_argument("--budgets", nargs="+", type=int, default=[500, 2000, 5000, 10_000],
                        metavar="N", help="rewire_budget values to sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)),
                        metavar="N", help="random seeds to sweep")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSONL path (default: experiments/<graph>.jsonl)")
    parser.add_argument("--graphs-dir", type=Path, default=None,
                        help="corpus root directory")
    parser.add_argument("--append", action="store_true",
                        help="append to existing JSONL instead of overwriting")
    args = parser.parse_args()

    out_path: Path = args.out or (_REPO / "experiments/sweeps" / f"{args.graph}.jsonl")
    target_path: Path = out_path.parent / f"{args.graph}_target.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    search_dirs = [args.graphs_dir] if args.graphs_dir else DEFAULT_SEARCH_DIRS
    print(f"Loading target signature for '{args.graph}' …")
    target_sig, _blocks, _graph_dir = load_target_from_corpus(args.graph, search_dirs)

    target_path.write_text(json.dumps(_sig_to_dict(target_sig), indent=2))
    print(f"Target saved → {target_path}")

    configs = list(product(args.budgets, args.seeds))
    total = len(configs)
    mode = "a" if args.append else "w"

    gen = Generator(target_sig)

    with out_path.open(mode) as fh:
        for i, (budget, seed) in enumerate(configs, 1):
            print(f"[{i}/{total}] budget={budget} seed={seed} … ", end="", flush=True)
            g_synth = gen.sample(seed=seed, rewire_budget=budget)
            synth_sig = Signature.from_graph(
                g_synth,
                skip_stars_and_paths=True,
                skip_shortest_paths=True,
            )
            record = {
                "graph": args.graph,
                "budget": budget,
                "seed": seed,
                "synth": _sig_to_dict(synth_sig),
            }
            fh.write(json.dumps(record) + "\n")
            print("✓")

    print(f"\nDone — {total} runs written to {out_path}")


if __name__ == "__main__":
    main()
