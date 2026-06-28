"""Visualise CC estimator variance collected by ``scripts/cc_variance.py``.

Reads the per-run CSV and the ``_meta.json`` sidecar written by the collector
and renders a grid of boxplots (rows = features, columns = n_samples); each
subplot shows the estimate spread at every n_colorings value with the exact
ground-truth drawn as a horizontal line.  The coefficient of variation
(std/mean) is printed per (n_samples, feature) so the variance reduction from
averaging more colourings and drawing more samples can be read off directly.

Usage
-----
    python scripts/cc_variance_viz.py experiments/cc_variance_sweeps/wn18rr_v4_sweep.csv
    python scripts/cc_variance_viz.py <csv> --out fig.png
    python scripts/cc_variance_viz.py <csv> --meta path/to/meta.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Estimated features shown as boxplots (triangle excluded — it is exact).
_PLOT_FEATURES = (
    ["four_cycle_count", "diamond_count", "k4_count", "tailed_triangle_count"]
    + ["five_cycle_count"]
    + [f"star_count_k{k}" for k in range(2, 11)]
)


def _load_rows(csv_path: Path) -> list[dict]:
    """Read the per-run CSV into a list of dicts with numeric values."""
    rows: list[dict] = []
    with csv_path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            # Every column is an integer count except none here are floats.
            rows.append({k: int(v) for k, v in raw.items()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path, help="CSV produced by cc_variance.py")
    parser.add_argument("--meta", type=Path, default=None,
                        help="Meta JSON (default: <csv-stem>_meta.json next to the CSV)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save figure here instead of showing interactively "
                             "(default: <csv>.png)")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")

    # Collector writes "<prefix>.csv" and "<prefix>_meta.json".
    meta_path = args.meta or args.csv.with_name(args.csv.stem + "_meta.json")
    if not meta_path.exists():
        sys.exit(f"Meta file not found: {meta_path}\nRun cc_variance.py first, or pass --meta.")

    meta = json.loads(meta_path.read_text())
    truth: dict[str, int | None] = meta["truth"]
    n_colorings_list: list[int] = meta["n_colorings_list"]
    n_samples_list: list[int] = meta["n_samples_list"]
    graph_name: str = meta.get("graph", "?")

    rows = _load_rows(args.csv)
    if not rows:
        sys.exit("CSV is empty.")

    png_path = args.out or args.csv.with_suffix(".png")
    _plot(rows, truth, png_path, n_colorings_list, n_samples_list, graph_name)


def _plot(rows: list[dict], truth: dict[str, int | None], png_path: Path,
          n_colorings_list: list[int], n_samples_list: list[int],
          graph_name: str) -> None:
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
    fig.suptitle(f"CC estimator variance vs. n_samples × n_colorings — {graph_name}",
                 fontsize=12)

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
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot → {png_path}")


if __name__ == "__main__":
    main()
