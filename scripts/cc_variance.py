"""Measure CC estimator variance for Block E 4-node motif counts.

Runs _cc_run with N different seeds on the same graph and records the
estimated counts for each seed.  Triangle count is included as a reference
(it is exact via list_triangles, so variance should be 0).

Output
------
  <out>.csv   — one row per seed, columns: seed, triangle_count,
                four_cycle_count, diamond_count, k4_count, tailed_triangle_count
  <out>.png   — boxplot per feature (saved) or interactive (omit --out)

Usage
-----
    python scripts/cc_variance.py fb237_v4_ind
    python scripts/cc_variance.py fb237_v4_ind --n-runs 100 --n-samples 10000
    python scripts/cc_variance.py fb237_v4_ind --out experiments/cc_variance
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_SCRIPTS))

from signature.block_e import BlockE as _BlockE  # noqa: E402
from signature_roundtrip import _DEFAULT_SEARCH_DIRS, _load_target_from_corpus  # noqa: E402

_MOTIF4_FEATURES = [
    ("four_cycle_count",      (2, 2, 2, 2)),
    ("diamond_count",         (2, 2, 3, 3)),
    ("k4_count",              (3, 3, 3, 3)),
    ("tailed_triangle_count", (1, 2, 2, 3)),
]
_ALL_FEATURES = ["triangle_count"] + [name for name, _ in _MOTIF4_FEATURES]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graph", help="Graph name in the corpus (e.g. fb237_v4_ind)")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Number of CC estimator runs with different seeds (default: 50)")
    parser.add_argument("--n-samples", type=int, default=100_000,
                        help="CC samples per run (default: 100000)")
    parser.add_argument("--graphs-dir", type=Path, default=None,
                        help="Corpus root directory")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path prefix (no extension); "
                             "default: experiments/<graph>_cc_variance")
    args = parser.parse_args()

    out_prefix: Path = args.out or (_REPO / "experiments" / f"{args.graph}_cc_variance")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")

    search_dirs = [args.graphs_dir] if args.graphs_dir else _DEFAULT_SEARCH_DIRS
    print(f"Loading '{args.graph}' …")
    _, tblocks, graph_dir = _load_target_from_corpus(args.graph, search_dirs)

    import igraph
    from kg_io import load_kg

    kg_files = list(graph_dir.glob("*.ttl")) + list(graph_dir.glob("*.nt"))
    if not kg_files:
        sys.exit(f"No .ttl/.nt file found in {graph_dir}")
    g = load_kg(kg_files[0])

    # Build simple undirected graph (same as BlockE.calculate() does internally)
    g_und = g.as_undirected(combine_edges="first").simplify()
    n = g_und.vcount()
    print(f"  Graph: {n:,} nodes, {g_und.ecount():,} edges")

    # Exact triangle count (deterministic reference)
    tri_exact = len(g_und.list_triangles()) if n >= 3 else 0
    print(f"  Exact triangle count: {tri_exact:,}")
    print(f"Running {args.n_runs} CC seeds with {args.n_samples:,} samples each …")

    rows: list[dict] = []
    for seed in range(args.n_runs):
        rng = np.random.default_rng(seed)
        motifs4 = _BlockE._cc_run(g_und, 4, args.n_samples, rng)
        row = {
            "seed":                  seed,
            "triangle_count":        tri_exact,
            "four_cycle_count":      motifs4.get((2, 2, 2, 2), 0),
            "diamond_count":         motifs4.get((2, 2, 3, 3), 0),
            "k4_count":              motifs4.get((3, 3, 3, 3), 0),
            "tailed_triangle_count": motifs4.get((1, 2, 2, 3), 0),
        }
        rows.append(row)
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{args.n_runs} done")

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed"] + _ALL_FEATURES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV → {csv_path}")

    # Also print the target values from the pre-measured signature for reference
    te = tblocks.get("e")
    if te is not None:
        print("\nTarget signature values:")
        for feat in _ALL_FEATURES:
            val = getattr(te, feat, "n/a")
            print(f"  {feat:<28} {val}")

    _plot(rows, png_path)


def _plot(rows: list[dict], png_path: Path) -> None:
    import matplotlib.pyplot as plt

    # Exclude triangle_count from boxplots (variance = 0, exact measurement)
    plot_features = [name for name, _ in _MOTIF4_FEATURES]
    data = [[r[feat] for r in rows] for feat in plot_features]

    fig, axes = plt.subplots(1, len(plot_features),
                             figsize=(4 * len(plot_features), 4), squeeze=False)
    fig.suptitle("CC estimator variance (4-node motifs)", fontsize=12)

    for ax, feat, vals in zip(axes[0], plot_features, data):
        ax.boxplot(vals, patch_artist=True)
        mean = float(np.mean(vals))
        ax.axhline(mean, color="red", linewidth=0.8, linestyle="--", label=f"mean={mean:.0f}")
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel(f"n={len(vals)} seeds")
        ax.set_ylabel("estimated count")
        ax.legend(fontsize=7)
        ax.set_xticks([])

        cv = float(np.std(vals) / mean) if mean > 0 else float("nan")
        ax.text(0.5, 0.02, f"CV={cv:.2%}", transform=ax.transAxes,
                ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Plot → {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
