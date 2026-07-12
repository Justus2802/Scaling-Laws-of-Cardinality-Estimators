"""Sweep Stage 3's ADAPTIVE_WEIGHT_SCALE and report the value that minimises the
accumulated (unweighted) error across all steered motifs/metrics after a fixed
rewire budget.

Stage 1 and Stage 2 are run once to build a fixed pre-Stage-3 graph; every
candidate scale then runs Stage 3 (`adaptive_weights=True`) from a fresh copy of
that same graph, so only the loss-weighting scheme differs between runs. The
comparison metric is the *unweighted* sum of relative errors over all active
terms at the best snapshot (`stage3_best_unweighted_error_sum`) — independent of
the weight scale itself, so runs with different scales are directly comparable
(unlike `stage3_best_loss`, which bakes the scale into its magnitude).

Usage
-----
    python scripts/sweep_adaptive_weight_scale.py wn18rr_v4
    python scripts/sweep_adaptive_weight_scale.py wn18rr_v4 \\
        --scales 1 5 10 20 30 50 75 100 --rewire-budget 100000
    python scripts/sweep_adaptive_weight_scale.py wn18rr_v4 --seed 7 --out sweep.csv
"""

import argparse
import csv
import logging
from pathlib import Path

import kgsynth.generator.stage3 as stage3  # noqa: E402
from kgsynth.generator.stage1 import sample_schema  # noqa: E402
from kgsynth.generator.stage2 import instantiate  # noqa: E402
from kgsynth.corpus import DEFAULT_SEARCH_DIRS, load_target_from_corpus

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_DEFAULT_SCALES = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graph", help="Graph name in the corpus (e.g. 'wn18rr_v4')")
    parser.add_argument("--graphs-dir", default=None,
                        help="Corpus root holding <graph>/signature/. "
                             "Default: searches data/graphs/ then data/test_graphs/.")
    parser.add_argument("--scales", nargs="+", type=float, default=_DEFAULT_SCALES,
                        metavar="X",
                        help=f"ADAPTIVE_WEIGHT_SCALE candidates (default: {_DEFAULT_SCALES})")
    parser.add_argument("--rewire-budget", type=int, default=100_000,
                        help="Stage 3 swap budget per candidate (default: 100000).")
    parser.add_argument("--initial-temp", type=float, default=0.05)
    parser.add_argument("--cooling-rate", type=float, default=0.99993)
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for Stage 1/2 (fixed pre-Stage-3 graph) and Stage 3.")
    parser.add_argument("--skip-c5", action="store_true")
    parser.add_argument("--skip-c6", action="store_true")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write per-candidate results to this CSV (default: no file).")
    args = parser.parse_args()

    search_dirs = [Path(args.graphs_dir)] if args.graphs_dir else DEFAULT_SEARCH_DIRS
    print(f"Loading target signature for '{args.graph}' from {[str(d) for d in search_dirs]}")
    target_sig, _tblocks, _graph_dir = load_target_from_corpus(args.graph, search_dirs)

    # Stage 1 + Stage 2 run once: every candidate scale rewires a fresh copy of
    # the same pre-Stage-3 graph, so the sweep isolates the effect of the scale.
    print("Building fixed pre-Stage-3 graph (Stage 1 + Stage 2) ...")
    schema = sample_schema(
        target_sig.a, target_sig.c, d=target_sig.d, b=target_sig.b, f=target_sig.f,
        seed=args.seed,
    )
    base_graph = instantiate(schema, seed=args.seed + 1)
    print(f"  V={base_graph.vcount():,}, E={base_graph.ecount():,}")

    results: list[dict] = []
    best = None

    for scale in args.scales:
        stage3.ADAPTIVE_WEIGHT_SCALE = scale
        g_copy = base_graph.copy()
        out = stage3.refine(
            g_copy, target_sig.e, target_f=target_sig.f,
            budget=args.rewire_budget,
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            seed=args.seed + 2,
            skip_c5=args.skip_c5,
            skip_c6=args.skip_c6,
            adaptive_weights=True,
        )
        error_sum = float(out["stage3_best_unweighted_error_sum"])
        row = {
            "scale": scale,
            "error_sum": error_sum,
            "best_loss": float(out["stage3_best_loss"]),
            "best_accepted": int(out["stage3_best_accepted"]),
            "executed_steps": int(out["stage3_executed_steps"]),
        }
        results.append(row)
        print(f"  scale={scale:<8g} error_sum={error_sum:.6f}  "
              f"best_loss={row['best_loss']:.4f}  accepted={row['best_accepted']}")

        if best is None or error_sum < best["error_sum"]:
            best = row

    print()
    print(f"Best ADAPTIVE_WEIGHT_SCALE = {best['scale']:g}  "
          f"(accumulated error {best['error_sum']:.6f} after {args.rewire_budget} swaps)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved   : {args.out}")


if __name__ == "__main__":
    main()
