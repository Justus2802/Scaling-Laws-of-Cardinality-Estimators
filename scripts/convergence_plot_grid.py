"""Plot a fixed 2x2 grid of Stage 3 convergence curves: triangle, diamond, c6, paw.

Same CSV input as convergence_plot.py (one or more files produced by
stage3.refine() with convergence_log set), but always renders exactly these
four `*_err` columns in a 2x2 grid instead of an arbitrary --features list —
handy for a fixed side-by-side view across runs (e.g. fixed-weight vs.
adaptive-weight) without re-specifying --features every time.

Usage
-----
    python scripts/convergence_plot_grid.py experiments/conv_a.csv
    python scripts/convergence_plot_grid.py experiments/conv_a.csv experiments/conv_b.csv \\
        --out experiments/convergence_grid.png
"""

import argparse
import csv
import sys
from pathlib import Path

# Fixed 2x2 layout: (row, col) -> convergence-CSV column stem.
_GRID = [
    ["tri_err", "diamond_err"],
    ["c6_err", "paw_err"],
]


def _load_csv(path: Path) -> dict[str, list]:
    """Return {column_name: [values]} for a convergence CSV."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns: dict[str, list] = {}
        for row in reader:
            for k, v in row.items():
                columns.setdefault(k, []).append(float(v))
    return columns


def _short_label(stem: str) -> str:
    """Shorten a convergence-CSV stem to 'adaptive weights' / 'constant weights'
    for the legend, based on the auto-named '_adaptive' token (see
    signature_roundtrip.py's --adaptive-weights). Falls back to the full stem
    when a filename doesn't match either pattern (e.g. a custom --out path)."""
    return "adaptive weights" if "_adaptive" in stem else "constant weights"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csvfiles", nargs="+", type=Path, metavar="CSV",
                        help="One or more convergence CSV files")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save figure to this path instead of showing interactively")
    args = parser.parse_args()

    # Keyed by the full stem (unique per file) so same-labelled runs (e.g. two
    # "constant weights" files) don't collide; the short label is looked up
    # separately at plot time.
    datasets: dict[str, dict[str, list]] = {}
    for p in args.csvfiles:
        if not p.exists():
            sys.exit(f"File not found: {p}")
        datasets[p.stem] = _load_csv(p)

    all_cols: set[str] = {col for data in datasets.values() for col in data}
    missing = [feat for row in _GRID for feat in row if feat not in all_cols]
    if missing:
        sys.exit(f"None of the input CSVs have column(s): {missing}. "
                 f"(This graph likely wasn't steered toward that term — "
                 f"check with convergence_plot.py --list-features.)")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    fig.suptitle("Stage 3 convergence", fontsize=12)

    for r, row_feats in enumerate(_GRID):
        for c, feat in enumerate(row_feats):
            ax = axes[r][c]
            for stem, data in datasets.items():
                if feat not in data:
                    continue
                xs = data.get("step") or data.get("accepted") or list(range(len(data[feat])))
                ax.plot(xs, data[feat], marker=".", markersize=4, linewidth=1,
                        label=_short_label(stem))
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_title(feat, fontsize=9)
            ax.set_xlabel("proposals")
            if c == 0:
                ax.set_ylabel("error")
            if len(datasets) > 1:
                ax.legend(fontsize=7)

    fig.tight_layout()

    out_path = args.out
    if out_path is None:
        first_csv = args.csvfiles[0]
        out_path = first_csv.with_name(f"{first_csv.stem}__tri_diamond_c6_paw_grid.png")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
