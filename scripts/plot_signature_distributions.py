"""Load all signature.json files and plot component-wise value distributions
across graphs.  One figure is produced per block; each subplot shows the
per-graph values for a single feature, annotated with its name and block context.

Both the full signature and the reduced signature (``--reduced``;
signature_reduced, no motif block) are read from the canonical graph store
data/graphs/. The full signature writes to data/graphs/distribution_plots/; the
reduced one writes to data/graph_population/. ``--source`` / ``--out`` override either.

Signatures are discovered as ``<source>/*/signature.json`` (flat layout) or
``<source>/*/signature/signature.json`` (the data/graphs/<name>/signature/ bundle
layout); both are scanned, so the same script serves either store.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent

# Per-letter subplot colour, shared by both signatures.
_BLOCK_COLOURS: dict[str, str] = {
    "a": "#4C72B0", "b": "#DD8452", "c": "#55A868",
    "d": "#C44E52", "e": "#8172B3", "f": "#937860",
}


def _block_config(reduced: bool) -> tuple[list[tuple[str, type, str, str]], Path]:
    """Return (block metadata, sig_out dir) for the selected signature.

    The reduced signature (``signature_reduced``) has no motif block (E); both
    signatures read from the canonical store ``data/graphs/``. Imports are local
    so the unused package isn't required to run either mode.
    """
    if reduced:
        from signature_reduced import BlockA, BlockB, BlockC, BlockD, BlockF
        blocks = [
            ("a", BlockA, "Block A — Size & Vocabulary"),
            ("b", BlockB, "Block B — Relation Freq & Multiplicity"),
            ("c", BlockC, "Block C — Schema & Co-occurrence"),
            ("d", BlockD, "Block D — Characteristic Sets & Two-step"),
            ("f", BlockF, "Block F — Connectivity"),
        ]
        sig_out = ROOT / "data" / "graphs"
    else:
        from signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF
        blocks = [
            ("a", BlockA, "Block A — Size & Density"),
            ("b", BlockB, "Block B — Degree Structure"),
            ("c", BlockC, "Block C — Schema & Co-occurrence"),
            ("d", BlockD, "Block D — Characteristic Sets"),
            ("e", BlockE, "Block E — Motifs"),
            ("f", BlockF, "Block F — Connectivity"),
        ]
        sig_out = ROOT / "data" / "graphs"
    return [(c, cls, title, _BLOCK_COLOURS[c]) for c, cls, title in blocks], sig_out


_STEM_ALIASES: dict[str, str] = {
    "59410577": "lubm",
    "59621618": "freebase",
}


def _short_name(source: str) -> str:
    """Derive a compact graph label from the source path in signature.json."""
    stem = Path(source).stem
    stem = _STEM_ALIASES.get(stem, stem)
    # Strip common suffixes to keep labels tidy.
    for suffix in ("_v4", "_l", "100k"):
        stem = stem.replace(suffix, "")
    return stem


def load_signatures(sig_out: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Scan *sig_out* for signature.json files with named features.

    Returns:
        graph_names: ordered list of short graph labels
        features:    {feature_name: [value_per_graph]} in graph_names order
    """
    graph_names: list[str] = []
    rows: list[dict[str, float]] = []

    # Accept both the flat sig_out layout (<name>/signature.json) and the
    # data/graphs bundle layout (<name>/signature/signature.json).
    paths = sorted(sig_out.glob("*/signature.json")) + sorted(
        sig_out.glob("*/signature/signature.json")
    )
    for path in paths:
        data = json.loads(path.read_text())
        if "features" not in data or not data["features"]:
            continue  # skip old-format files without named features
        graph_names.append(_short_name(data["source"]))
        rows.append(data["features"])

    if not rows:
        raise RuntimeError(f"No named-feature signature.json files found in {sig_out}")

    # Collect feature order from the first file; all files share the same schema.
    all_feature_names = list(rows[0].keys())
    features: dict[str, list[float]] = {
        name: [row.get(name, float("nan")) for row in rows]
        for name in all_feature_names
    }
    return graph_names, features


def _grid(n: int) -> tuple[int, int]:
    """Return (nrows, ncols) for a roughly square subplot grid of size n."""
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_block(
    block_char: str,
    block_cls: type,
    title: str,
    colour: str,
    graph_names: list[str],
    features: dict[str, list[float]],
    out_dir: Path,
) -> Path | None:
    """Plot component-wise value distributions for one block and save to disk."""
    names = block_cls.feature_names()
    # Keep only features present in the loaded data.
    names = [n for n in names if n in features]
    if not names:
        return None

    n = len(names)
    nrows, ncols = _grid(n)
    fig_w = max(ncols * 2.0, 8)
    fig_h = max(nrows * 2.2, 5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # Flatten axes for uniform indexing; hide any extras.
    ax_flat = np.array(axes).ravel() if n > 1 else np.array([axes])
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    for ax, feat_name in zip(ax_flat[:n], names):
        values = np.array(features[feat_name], dtype=float)
        finite = values[~np.isnan(values)]
        n_nan = int(np.isnan(values).sum())

        if finite.size > 0:
            # Sturges' rule capped at the number of finite points so bins never
            # exceed the data count (avoids empty bins with only a few graphs).
            n_bins = max(1, min(finite.size, int(np.ceil(np.log2(finite.size))) + 1))
            ax.hist(finite, bins=n_bins, color=colour, alpha=0.75,
                    edgecolor="white", linewidth=0.5, zorder=2)

        ax.tick_params(axis="both", labelsize=6)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        label = feat_name if not n_nan else f"{feat_name}\n({n_nan} NaN)"
        ax.set_title(label, fontsize=7, pad=3)
        ax.set_xlabel("value", fontsize=5)
        ax.set_ylabel("count", fontsize=5)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=1)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = out_dir / f"block_{block_char}_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduced", action="store_true",
        help="Plot the reduced signature (signature_reduced) instead of the full "
             "signature; both are read from data/graphs/.",
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Override the signature source directory (default per --reduced).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Override the output directory (default: data/graph_population/ for "
             "--reduced, else <source>/distribution_plots/).",
    )
    args = parser.parse_args()

    blocks, sig_out = _block_config(args.reduced)
    if args.source is not None:
        sig_out = args.source
    if args.out is not None:
        out_dir = args.out
    elif args.reduced:
        out_dir = ROOT / "data" / "graph_population"
    else:
        out_dir = sig_out / "distribution_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning : {sig_out}")
    graph_names, features = load_signatures(sig_out)
    print(f"Graphs   : {graph_names}")
    print(f"Features : {len(features)} total")

    saved: list[Path] = []
    for block_char, block_cls, title, colour in blocks:
        path = plot_block(
            block_char, block_cls, title, colour,
            graph_names, features, out_dir,
        )
        if path:
            print(f"  Saved  : {path}")
            saved.append(path)
        else:
            print(f"  Skipped: {title} (no data)")

    print(f"\nDone. {len(saved)} figures written to {out_dir}/")


if __name__ == "__main__":
    main()
