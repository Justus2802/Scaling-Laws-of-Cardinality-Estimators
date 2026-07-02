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

from signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF  # noqa: E402
from generator import Generator, Signature as GenSignature  # noqa: E402
from generator.stage1 import sample_schema  # noqa: E402

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


def _stage1_caps(target_dict: dict) -> tuple[int, int]:
    """Return (max_out_degree, max_in_degree) as computed by Stage 1 for a target.

    Runs sample_schema with a fixed seed; the caps are deterministic (no randomness).
    Returns (0, 0) if block A or B are unavailable.
    """
    a = _reconstruct_block("a", target_dict.get("a"))
    b = _reconstruct_block("b", target_dict.get("b"))
    c = _reconstruct_block("c", target_dict.get("c"))
    if a is None or c is None:
        return 0, 0
    schema = sample_schema(a, c, b=b, seed=0)
    return schema.max_out_degree, schema.max_in_degree


def _degree_stats(synth_dict: dict) -> dict:
    """Extract max degree and top-10 out-degrees from a serialized synth block-B.

    Returns keys: max_out, max_in, top10_out (list, descending) or None if unavailable.
    """
    import numpy as np

    b_data = synth_dict.get("b")
    if b_data is None:
        return {"max_out": None, "max_in": None, "top10_out": None}

    from signature import BlockB
    b = BlockB.from_serializable(b_data)
    out_deg = b._out_degrees
    in_deg = b._in_degrees

    max_out = int(out_deg.max()) if out_deg is not None and len(out_deg) else None
    max_in = int(in_deg.max()) if in_deg is not None and len(in_deg) else None
    top10_out = (
        [int(d) for d in np.sort(out_deg)[::-1][:10]]
        if out_deg is not None and len(out_deg) else None
    )
    return {"max_out": max_out, "max_in": max_in, "top10_out": top10_out}


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

    # Compute stage-1 degree caps (deterministic from the target).
    cap_out, cap_in = _stage1_caps(target_dict)
    cap_str = (
        f"stage-2 caps: max_out={cap_out or 'none'}  max_in={cap_in or 'none'}"
    )

    # Print per-record degree stats before any feature filtering.
    print(f"\n{cap_str}")
    print(f"\n{'budget':>8}  {'seed':>4}  {'max_out':>7}  {'max_in':>6}  top-10 out-degrees")
    print("-" * 70)
    for rec in records:
        stats = _degree_stats(rec["synth"])
        top10_str = (
            ", ".join(str(d) for d in stats["top10_out"])
            if stats["top10_out"] is not None else "n/a"
        )
        print(
            f"{rec['budget']:>8}  {rec['seed']:>4}  "
            f"{stats['max_out'] if stats['max_out'] is not None else 'n/a':>7}  "
            f"{stats['max_in'] if stats['max_in'] is not None else 'n/a':>6}  "
            f"{top10_str}"
        )
    print()

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

    # Collect errors per budget group (older records may carry a now-removed
    # remeasure_interval field; include it in the label only when present).
    # groups: label → feature_name → list of (seed_idx, float) so dots can be
    # colored consistently across feature subplots for the same record.
    groups: dict[str, dict[str, list[tuple[int, float]]]] = {}
    # seed_idx_map: (label, position_within_label) → global seed index
    label_counters: dict[str, int] = {}
    seed_idx_map: dict[tuple[str, int], int] = {}
    next_seed_idx = 0

    for rec in records:
        label = f"B{rec['budget']}"
        if "remeasure_interval" in rec:
            label += f"/I{rec['remeasure_interval']}"
        if label not in groups:
            groups[label] = {f: [] for f in args.features}
            label_counters[label] = 0

        pos = label_counters[label]
        label_counters[label] += 1
        key = (label, pos)
        if key not in seed_idx_map:
            seed_idx_map[key] = next_seed_idx
            next_seed_idx += 1
        sidx = seed_idx_map[key]

        synth_features: dict[str, float] = {}
        for letter in _LETTERS:
            blk = _reconstruct_block(letter, rec["synth"].get(letter))
            synth_features.update(_block_feature_map(blk))

        for feat in args.features:
            tval = target_features.get(feat, float("nan"))
            sval = synth_features.get(feat, float("nan"))
            err = _rel_err(tval, sval)
            if err is not None and not math.isnan(sval):
                groups[label][feat].append((sidx, err))

    # Sort config labels by budget then interval for a natural left-to-right order
    def _label_sort_key(lbl: str):
        # format: "B{budget}" or "B{budget}/I{interval}"
        parts = lbl.lstrip("B").split("/I")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)

    config_labels = sorted(groups.keys(), key=_label_sort_key)

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_feat = len(args.features)
    fig, axes = plt.subplots(1, n_feat, figsize=(max(6, 4 * n_feat), 5), squeeze=False)
    fig.suptitle(f"Signed relative error by config — {graph_name}", fontsize=12)

    n_seeds = next_seed_idx
    cmap = cm.get_cmap("tab10" if n_seeds <= 10 else "tab20", max(n_seeds, 1))
    seed_colors = {i: cmap(i) for i in range(n_seeds)}

    for col, feat in enumerate(args.features):
        ax = axes[0][col]
        # Strip seed index for boxplot/violin (they just need raw values)
        raw_per_config = [[v for _, v in groups[lbl][feat]] for lbl in config_labels]

        if args.kind == "violin" and all(len(d) >= 2 for d in raw_per_config):
            ax.violinplot(raw_per_config, positions=range(len(config_labels)),
                          showmedians=True, showextrema=True)
        else:
            ax.boxplot(raw_per_config, labels=config_labels, patch_artist=True)

        # Overlay individual seed points colored by seed index
        base = 0 if args.kind == "violin" else 1
        for x, pairs in enumerate(groups[lbl][feat] for lbl in config_labels):
            for sidx, val in pairs:
                ax.scatter([base + x], [val], color=seed_colors[sidx],
                           s=30, zorder=5, alpha=0.85, edgecolors="none")

        if args.kind == "violin":
            ax.set_xticks(range(len(config_labels)))
            ax.set_xticklabels(config_labels, rotation=30, ha="right", fontsize=8)
        else:
            ax.set_xticklabels(config_labels, rotation=30, ha="right", fontsize=8)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("signed relative error" if col == 0 else "")

    # Add a shared seed legend on the last axis
    if n_seeds > 0:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=seed_colors[i],
                       markersize=7, label=f"seed {i}")
            for i in range(n_seeds)
        ]
        axes[0][-1].legend(handles=handles, title="seed", fontsize=7,
                           title_fontsize=7, loc="upper right")

    fig.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
