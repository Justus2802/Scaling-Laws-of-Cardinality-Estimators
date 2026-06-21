"""Visualise per-feature relative error distributions from a sweep JSONL file.

Error is defined as |target - synth| / max(|target|, 1e-9) per feature.
Features with a NaN target value are skipped.

Usage
-----
    python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl \\
        --features triangle_count clustering_coefficient four_cycle_count
    python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl \\
        --features triangle_count --kind violin --out experiments/fig.png
    python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl --list-features
"""

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from signature_reduced import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF  # noqa: E402

_LETTERS = ("a", "b", "c", "d", "e", "f")
_BLOCK_CLS = {"a": BlockA, "b": BlockB, "c": BlockC, "d": BlockD, "e": BlockE, "f": BlockF}


def _reconstruct_block(letter: str, data):
    """Reconstruct a block from serialized data; returns None if data is None."""
    if data is None:
        return None
    return _BLOCK_CLS[letter].from_serializable(data)


def _block_feature_map(blk) -> dict[str, float]:
    """Return {feature_name: value} for a block; empty dict if block is None."""
    if blk is None:
        return {}
    return dict(zip(blk.feature_names(), blk.as_vector()))


def _all_feature_names(sig_dict: dict) -> list[str]:
    names: list[str] = []
    for letter in _LETTERS:
        blk = _reconstruct_block(letter, sig_dict.get(letter))
        if blk is not None:
            names.extend(blk.feature_names())
    return names


def _rel_err(target_val: float, synth_val: float) -> float | None:
    """Signed relative error (synth − target) / |target|; None if target is NaN.

    Positive values mean the synthesised graph over-estimates the feature;
    negative values indicate under-estimation, exposing systematic bias.
    """
    if math.isnan(target_val):
        return None
    return (synth_val - target_val) / max(abs(target_val), 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", type=Path, help="JSONL file produced by sweep_collect.py")
    parser.add_argument("--features", nargs="+", metavar="NAME",
                        help="Feature names to plot (from feature_names() of each block)")
    parser.add_argument("--target", type=Path, default=None,
                        help="Target JSON file (default: <graph>_target.json next to JSONL)")
    parser.add_argument("--kind", choices=["box", "violin"], default="box",
                        help="Plot type (default: box)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save figure to this path instead of showing interactively")
    parser.add_argument("--list-features", action="store_true",
                        help="Print all available feature names and exit")
    args = parser.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"JSONL file not found: {args.jsonl}")

    # Load records
    records: list[dict] = []
    with args.jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        sys.exit("JSONL file is empty.")

    graph_name = records[0]["graph"]

    # Locate target JSON
    target_path = args.target or (args.jsonl.parent / f"{graph_name}_target.json")
    if not target_path.exists():
        sys.exit(f"Target file not found: {target_path}\n"
                 "Run sweep_collect.py first, or pass --target.")
    target_dict = json.loads(target_path.read_text())

    # Build target feature map
    target_features: dict[str, float] = {}
    for letter in _LETTERS:
        blk = _reconstruct_block(letter, target_dict.get(letter))
        target_features.update(_block_feature_map(blk))

    if args.list_features:
        print("Available features:")
        for name in target_features:
            tval = target_features[name]
            nan_note = "  [NaN — skipped in error]" if math.isnan(tval) else ""
            print(f"  {name:<45}  target={tval:.4g}{nan_note}")
        return

    if not args.features:
        sys.exit("Specify --features or use --list-features to see available names.")

    # Validate requested feature names
    unknown = [f for f in args.features if f not in target_features]
    if unknown:
        print(f"Unknown feature(s): {unknown}")
        print("Available features (non-NaN):")
        for name, v in target_features.items():
            if not math.isnan(v):
                print(f"  {name}")
        sys.exit(1)

    # Collect errors per (budget, remeasure_interval) group
    # groups: label → feature_name → list[float]
    groups: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        label = f"B{rec['budget']}/I{rec['remeasure_interval']}"
        if label not in groups:
            groups[label] = {f: [] for f in args.features}

        synth_features: dict[str, float] = {}
        for letter in _LETTERS:
            blk = _reconstruct_block(letter, rec["synth"].get(letter))
            synth_features.update(_block_feature_map(blk))

        for feat in args.features:
            tval = target_features.get(feat, float("nan"))
            sval = synth_features.get(feat, float("nan"))
            err = _rel_err(tval, sval)
            if err is not None and not math.isnan(sval):
                groups[label][feat].append(err)

    # Sort config labels by budget then interval for a natural left-to-right order
    def _label_sort_key(lbl: str):
        # format: "B{budget}/I{interval}"
        parts = lbl.lstrip("B").split("/I")
        return (int(parts[0]), int(parts[1]))

    config_labels = sorted(groups.keys(), key=_label_sort_key)

    import matplotlib.pyplot as plt

    n_feat = len(args.features)
    fig, axes = plt.subplots(1, n_feat, figsize=(max(6, 4 * n_feat), 5), squeeze=False)
    fig.suptitle(f"Signed relative error by config — {graph_name}", fontsize=12)

    for col, feat in enumerate(args.features):
        ax = axes[0][col]
        data_per_config = [groups[lbl][feat] for lbl in config_labels]

        if args.kind == "violin" and all(len(d) >= 2 for d in data_per_config):
            parts = ax.violinplot(data_per_config, positions=range(len(config_labels)),
                                  showmedians=True, showextrema=True)
        else:
            ax.boxplot(data_per_config, labels=config_labels, patch_artist=True)

        # Overlay individual seed points — boxplot uses 1-based positions by default
        base = 0 if args.kind == "violin" else 1
        for x, vals in enumerate(data_per_config):
            ax.scatter([base + x] * len(vals), vals, color="black", s=20, zorder=5, alpha=0.7)

        if args.kind == "violin":
            ax.set_xticks(range(len(config_labels)))
            ax.set_xticklabels(config_labels, rotation=30, ha="right", fontsize=8)
        else:
            ax.set_xticklabels(config_labels, rotation=30, ha="right", fontsize=8)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("signed relative error" if col == 0 else "")

    fig.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
