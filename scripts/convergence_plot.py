"""Plot Stage 3 convergence curves from one or more CSV files.

Each CSV is produced by refine() when convergence_log is set.  Columns:
  accepted, loss, tri_err, [c4_err, diamond_err, k4_err, paw_err,] [c5_err, c6_err,]
  [cc_err,] [assort_err], sig_tri_err, [sig_c4_err, …], sig_c5_err

All metric columns are relative errors (plotted against a 0 reference line).  The
``sig_*_err`` columns are ground-truth errors measured periodically on the full
graph; ``sig_c5_err`` is the global (induced) 5-cycle error, validating the
incremental cycle delta.

Usage
-----
    python scripts/convergence_plot.py experiments/conv_a.csv experiments/conv_b.csv
    python scripts/convergence_plot.py experiments/conv_a.csv \\
        --features tri_err cc_err --out experiments/convergence.png
    python scripts/convergence_plot.py experiments/conv_a.csv --list-features
"""

import argparse
import csv
import sys
from pathlib import Path


def _load_csv(path: Path) -> dict[str, list]:
    """Return {column_name: [values]} for a convergence CSV."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns: dict[str, list] = {}
        for row in reader:
            for k, v in row.items():
                columns.setdefault(k, []).append(float(v))
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csvfiles", nargs="+", type=Path, metavar="CSV",
                        help="One or more convergence CSV files")
    parser.add_argument("--features", nargs="+", metavar="NAME",
                        help="Metric columns to plot (default: all except accepted/loss)")
    parser.add_argument("--include-loss", action="store_true",
                        help="Add total loss as an extra panel")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save figure to this path instead of showing interactively")
    parser.add_argument("--list-features", action="store_true",
                        help="Print available metric columns from the first file and exit")
    args = parser.parse_args()

    # Load all files
    datasets: dict[str, dict[str, list]] = {}
    for p in args.csvfiles:
        if not p.exists():
            sys.exit(f"File not found: {p}")
        datasets[p.stem] = _load_csv(p)

    # Collect available metric columns (union across files, excluding accepted/loss)
    _skip = {"accepted", "loss"}
    all_metrics: list[str] = []
    seen: set[str] = set()
    for data in datasets.values():
        for col in data:
            if col not in _skip and col not in seen:
                all_metrics.append(col)
                seen.add(col)

    if args.list_features:
        first = next(iter(datasets.values()))
        print("Available columns:")
        for col in first:
            if col not in _skip:
                print(f"  {col}")
        return

    selected = args.features if args.features else all_metrics
    if args.include_loss:
        selected = list(selected) + ["loss"]

    # Validate
    unknown = [f for f in selected if f not in seen and f != "loss"]
    if unknown:
        print(f"Unknown feature(s): {unknown}")
        print(f"Available: {all_metrics}")
        sys.exit(1)

    import matplotlib.pyplot as plt

    n = len(selected)
    if n == 0:
        sys.exit("No features to plot.")

    fig, axes = plt.subplots(1, n, figsize=(max(5, 4 * n), 4), squeeze=False)
    fig.suptitle("Stage 3 convergence", fontsize=12)

    for col_idx, feat in enumerate(selected):
        ax = axes[0][col_idx]
        for label, data in datasets.items():
            if feat not in data:
                continue
            xs = data.get("accepted", list(range(len(data[feat]))))
            ax.plot(xs, data[feat], marker=".", markersize=4, linewidth=1, label=label)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("accepted swaps")
        if col_idx == 0:
            ax.set_ylabel("error")
        if len(datasets) > 1:
            ax.legend(fontsize=7)

    fig.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
