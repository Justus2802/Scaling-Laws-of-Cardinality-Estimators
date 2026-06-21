"""Measure CC estimator variance for Block E motif and induced-star counts.

Runs the colour-coding estimator (``CCMotifCounter``) with N different seeds on
the same graph and records the estimated 4-/5-node motif counts and induced
k-star counts (k=2..10) for each seed.  The exact ground-truth count for every
feature is computed once via ``ExactMotifCounter`` and overlaid on each boxplot
so the estimator's bias and spread can be read off directly.  Triangle count is
exact (via list_triangles), so its variance is 0 and it is excluded from the
boxplots.

Stars use the same colour-coding machinery (``cc_run_stars``), which now also
averages over ``n_colorings`` colourings — so their boxplots tighten along the
n_colorings axis just like the motif estimators, and the all-zero collapse at
high k (single-colouring failure) is visibly mitigated by larger n_colorings.

Sweeps both ``--n-colorings`` and ``--n-samples`` (a 2-D grid) so the variance
reduction from averaging more independent colourings (Alon–Yuster–Zwick 1995;
Motivo / Bressan et al. 2021) and from drawing more path samples can be read
off directly.

Output (default prefix: experiments/cc_variance_sweeps/<graph>_sweep)
------
  <out>.csv   — one row per (n_samples, n_colorings, seed); columns: n_samples,
                n_colorings, seed, triangle_count, four_cycle_count,
                diamond_count, k4_count, tailed_triangle_count, five_cycle_count
  <out>.png   — a grid of boxplots (rows = features, columns = n_samples); each
                subplot shows the estimate spread at every n_colorings value
                with the exact ground-truth as a horizontal line.  The
                coefficient of variation is also printed per (n_samples, feature).

Usage
-----
    python scripts/cc_variance.py wn18rr_v4
    python scripts/cc_variance.py fb237_v4_ind --n-runs 100 --n-samples 10000 50000
    python scripts/cc_variance.py wn18rr_v4 --n-colorings 1 4 16 64 --n-samples 1000 10000 100000
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_SCRIPTS))

from motif_counter import CCMotifCounter, ExactMotifCounter  # noqa: E402
from signature_roundtrip import _DEFAULT_SEARCH_DIRS, _load_target_from_corpus  # noqa: E402

# (feature name, sorted degree sequence) for the CC-estimated 4-node motifs.
_MOTIF4_FEATURES = [
    ("four_cycle_count",      (2, 2, 2, 2)),
    ("diamond_count",         (2, 2, 3, 3)),
    ("k4_count",              (3, 3, 3, 3)),
    ("tailed_triangle_count", (1, 2, 2, 3)),
]
# (feature name, sorted degree sequence) for the CC-estimated 5-node motifs.
_MOTIF5_FEATURES = [
    ("five_cycle_count",      (2, 2, 2, 2, 2)),
]
# (feature name, k) for the CC-estimated induced k-stars (k=2..10).
_STAR_FEATURES = [(f"star_count_k{k}", k) for k in range(2, 11)]
# Estimated features shown as boxplots (triangle excluded — it is exact).
_PLOT_FEATURES = [name for name, _ in _MOTIF4_FEATURES + _MOTIF5_FEATURES + _STAR_FEATURES]
_ALL_FEATURES = ["triangle_count"] + _PLOT_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graph", help="Graph name in the corpus (e.g. fb237_v4_ind)")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Number of CC estimator runs with different seeds (default: 50)")
    parser.add_argument("--n-samples", type=int, nargs="+", default=[10_000, 100_000],
                        help="Path-sample counts to sweep; variance is collected for each "
                             "(default: 10000 100000).")
    parser.add_argument("--n-colorings", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="Colouring counts to sweep; variance is collected for each "
                             "(default: 1 2 4 8 16 32). Each estimate averages that many "
                             "independent colourings.")
    parser.add_argument("--graphs-dir", type=Path, default=None,
                        help="Corpus root directory")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path prefix (no extension); "
                             "default: experiments/cc_variance_sweeps/<graph>_sweep")
    args = parser.parse_args()

    out_prefix: Path = args.out or (
        _REPO / "experiments" / "cc_variance_sweeps" / f"{args.graph}_sweep"
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")

    search_dirs = [args.graphs_dir] if args.graphs_dir else _DEFAULT_SEARCH_DIRS
    print(f"Loading '{args.graph}' …")
    _, tblocks, graph_dir = _load_target_from_corpus(args.graph, search_dirs)
    assert graph_dir is not None

    from kg_io import load_kg

    kg_files = sorted(
        p for p in graph_dir.iterdir()
        if p.suffix in {".nt", ".ttl"} and not p.stem.endswith("_synth")
    )
    if not kg_files:
        sys.exit(f"No .ttl/.nt file found in {graph_dir}")
    g = load_kg(kg_files[0])

    # Build simple undirected graph (same as BlockE.calculate() does internally)
    g_und = g.as_undirected(combine_edges="first").simplify()
    n = g_und.vcount()
    print(f"  Graph: {n:,} nodes, {g_und.ecount():,} edges")

    # Exact ground truth (deterministic reference) via full enumeration.
    truth = _exact_ground_truth(g_und)
    print("  Exact ground-truth counts:")
    for feat in _ALL_FEATURES:
        val = truth.get(feat)
        print(f"    {feat:<28} {'n/a' if val is None else f'{val:,}'}")

    n_colorings_list = sorted(set(args.n_colorings))
    n_samples_list = sorted(set(args.n_samples))
    print(f"Sweeping n_samples={n_samples_list} × n_colorings={n_colorings_list}; "
          f"{args.n_runs} CC seeds each …")
    rows: list[dict] = []
    _t_all = time.perf_counter()
    for ns in n_samples_list:
        for nc in n_colorings_list:
            print(f"  n_samples={ns:,}, n_colorings={nc}: running {args.n_runs} seeds …",
                  flush=True)
            _t_nc = time.perf_counter()
            for seed in range(args.n_runs):
                # Fresh CCMotifCounter per seed so each run is an independent estimate.
                cc = CCMotifCounter(n_samples=ns, seed=seed, n_colorings=nc)
                motifs4 = cc.count_motifsk(g_und, 4)
                motifs5 = cc.count_motifsk(g_und, 5)
                stars = cc.count_stars(g_und)
                row = {"n_samples": ns, "n_colorings": nc, "seed": seed,
                       "triangle_count": truth["triangle_count"]}
                for name, ds in _MOTIF4_FEATURES:
                    row[name] = motifs4.get(ds, 0)
                for name, ds in _MOTIF5_FEATURES:
                    row[name] = motifs5.get(ds, 0)
                for name, k in _STAR_FEATURES:
                    row[name] = stars.get(k, 0)
                rows.append(row)
                if (seed + 1) % 10 == 0 or seed + 1 == args.n_runs:
                    _el = time.perf_counter() - _t_nc
                    print(f"    seed {seed + 1}/{args.n_runs}  "
                          f"({_el:.1f}s, {_el / (seed + 1) * 1e3:.0f} ms/seed)  "
                          f"c5~{row['five_cycle_count']}", flush=True)
            print(f"  n_samples={ns:,}, n_colorings={nc} done in "
                  f"{time.perf_counter() - _t_nc:.1f}s", flush=True)
    print(f"All sweeps done in {time.perf_counter() - _t_all:.1f}s", flush=True)

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["n_samples", "n_colorings", "seed"] + _ALL_FEATURES
        )
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

    _plot(rows, truth, png_path, n_colorings_list, n_samples_list)


def _exact_ground_truth(g_und) -> dict[str, int | None]:
    """Compute exact counts for every feature via ExactMotifCounter.

    Returns a ``{feature_name: count}`` dict.  The 5-cycle entry is ``None``
    when the exact ESCAPE enumeration is infeasible (high-degree hub) or
    unsupported, so callers can skip drawing its ground-truth line.
    """
    exact = ExactMotifCounter()
    truth: dict[str, int | None] = {"triangle_count": exact.count_triangles(g_und)}

    motifs4 = exact.count_motifs4(g_und)
    for name, ds in _MOTIF4_FEATURES:
        truth[name] = motifs4.get(ds, 0)

    try:
        motifs5 = exact.count_motifsk(g_und, 5)
        for name, ds in _MOTIF5_FEATURES:
            truth[name] = motifs5.get(ds, 0)
    except (RuntimeError, NotImplementedError) as exc:
        print(f"  ! exact 5-node count unavailable ({exc}); skipping its ground-truth line")
        for name, _ in _MOTIF5_FEATURES:
            truth[name] = None

    stars = exact.count_stars(g_und)
    for name, k in _STAR_FEATURES:
        truth[name] = stars.get(k, 0)

    return truth


def _plot(rows: list[dict], truth: dict[str, int | None], png_path: Path,
          n_colorings_list: list[int], n_samples_list: list[int]) -> None:
    import matplotlib.pyplot as plt

    # Print the coefficient-of-variation table (the core variance data) to console,
    # one block per n_samples value.
    for ns in n_samples_list:
        print(f"\nCoefficient of variation (std/mean) vs n_colorings  (n_samples={ns:,}):")
        header = "  " + "feature".ljust(24) + "".join(f"nc={nc:<8}" for nc in n_colorings_list)
        print(header)
        for feat in _PLOT_FEATURES:
            cells = ""
            for nc in n_colorings_list:
                vals = [r[feat] for r in rows
                        if r["n_colorings"] == nc and r["n_samples"] == ns]
                mean = float(np.mean(vals)) if vals else 0.0
                cv = (float(np.std(vals)) / mean) if mean > 0 else float("nan")
                cells += f"{cv:<11.2%}"
            print("  " + feat.ljust(24) + cells)

    # Grid: rows = features, columns = n_samples. Within each subplot, one boxplot
    # per n_colorings value so both sweep axes are visible at once.
    nrows, ncols = len(_PLOT_FEATURES), len(n_samples_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    fig.suptitle("CC estimator variance vs. n_samples × n_colorings "
                 "(4- and 5-node motifs)", fontsize=12)

    positions = list(range(len(n_colorings_list)))
    for i, feat in enumerate(_PLOT_FEATURES):
        exact_val = truth.get(feat)
        for j, ns in enumerate(n_samples_list):
            ax = axes[i][j]
            data_by_nc = [[r[feat] for r in rows
                           if r["n_colorings"] == nc and r["n_samples"] == ns]
                          for nc in n_colorings_list]
            ax.boxplot(data_by_nc, patch_artist=True, positions=positions)
            ax.set_xticks(positions)
            ax.set_xticklabels([str(nc) for nc in n_colorings_list])

            if exact_val is not None:
                ax.axhline(exact_val, color="green", linewidth=1.2, linestyle="-",
                           label=f"exact={exact_val:.0f}")
                ax.legend(fontsize=7)

            if i == 0:
                ax.set_title(f"n_samples={ns:,}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{feat}\nestimated count", fontsize=8)
            if i == nrows - 1:
                ax.set_xlabel("n_colorings  (more → less variance)", fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Plot → {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
