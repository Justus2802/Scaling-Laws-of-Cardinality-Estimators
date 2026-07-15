"""Boxplot of per-feature/per-distribution relative errors for a ``kgsynth dataset`` run.

Loads every ``graph_*/distance.json`` under a dataset output directory (as
produced by ``kgsynth dataset --measure``, see ``docs/dataset.md``) and pools
their per-feature errors by signature block:

- **Distribution features** (``distance.json``'s ``normalised_w1``, keyed
  ``<block>:<name>``) use the normalised Wasserstein-1 distance already
  computed by the dataset worker (:func:`kgsynth.dataset.worker._distances`).
- **Standalone scalars** (``distance.json``'s ``per_feature_relative_error``,
  keyed by bare feature name) keep plain relative error, attributed to their
  owning block via each block's ``feature_names()``.

This is the population companion to ``signature_error_boxplot.py`` (which
compares one target against one ``signature_roundtrip.py`` synthetic graph):
here every replica in the dataset contributes one error sample per feature, so
each box shows the spread across the whole population instead of a single
draw.

Usage
-----
    python scripts/dataset_error_boxplot.py generated/wn18rr_v4
    python scripts/dataset_error_boxplot.py generated/wn18rr_v4 --out fig.png
    python scripts/dataset_error_boxplot.py generated/wn18rr_v4 --exclude   # disable exclusion
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from kgsynth.signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF
from kgsynth.corpus import REPO_ROOT

# Fixed block order + one colour per block (categorical, never re-cycled) —
# kept identical to signature_error_boxplot.py so the two figures read the same way.
_BLOCK_ORDER = ["a", "b", "c", "d", "e", "f"]
_BLOCK_LABELS = {
    "a": "size & vocab",
    "b": "degree & multiplicity",
    "c": "schema & co-occurrence",
    "d": "char. sets & two-step",
    "e": "motifs & templates",
    "f": "connectivity",
}
_BLOCK_COLOURS = {
    "a": "#4C72B0", "b": "#DD8452", "c": "#55A868",
    "d": "#C44E52", "e": "#8172B3", "f": "#937860",
}
_BLOCK_CLASSES = {"a": BlockA, "b": BlockB, "c": BlockC, "d": BlockD, "e": BlockE, "f": BlockF}

# Excluded by default, same rationale as signature_error_boxplot.py's _DEFAULT_EXCLUDE:
#   - a_obj: a near-zero-by-construction OLS offset, so a small absolute miss
#     produces a runaway relative error (denominator artifact).
#   - cs_freq (W1): the target's CS-frequency distribution has a very narrow
#     IQR, so W1/IQR normalisation inflates even a modest absolute mismatch.
#   - path/tree template Zipf+entropy (Block E, k=2..10 plus the two tree
#     scalars) and shortest-path stats (Block F): excluded by request, not a
#     numeric-artifact case like the two above.
_TEMPLATE_FEATURES = (
    [f"e:path_template_zipf_k{k}" for k in range(2, 11)]
    + [f"e:path_template_entropy_k{k}" for k in range(2, 11)]
    + ["e:tree_template_zipf", "e:tree_template_entropy"]
)
_SHORTEST_PATH_FEATURES = ["f:shortest_path_max", "f:shortest_path_mean", "f:shortest_path_var"]
_DEFAULT_EXCLUDE = {"b:a_obj", "d:cs_freq (W1)", *_TEMPLATE_FEATURES, *_SHORTEST_PATH_FEATURES}


def _feature_to_block() -> dict[str, str]:
    """Map every scalar feature name to its owning block letter, via ``feature_names()``."""
    mapping: dict[str, str] = {}
    for letter, cls in _BLOCK_CLASSES.items():
        for name in cls().feature_names():
            mapping[name] = letter
    return mapping


def _iter_distance_files(dataset_dir: Path) -> list[Path]:
    """Return every ``graph_*/distance.json`` under *dataset_dir*, sorted by graph index."""
    paths = sorted(dataset_dir.glob("graph_*/distance.json"))
    if not paths:
        raise SystemExit(
            f"No graph_*/distance.json under {dataset_dir}. "
            "Run with 'kgsynth dataset --measure' first (see docs/dataset.md)."
        )
    return paths


def _collect_errors(
    distance_files: list[Path], feature_to_block: dict[str, str], exclude: set[str],
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Pool per-feature errors across all replicas, grouped by owning block.

    :returns: ``(per_block, dropped_counts)`` — errors keyed by block letter, and
        the number of excluded samples per dropped label (for the console report).
    """
    per_block: dict[str, list[float]] = {letter: [] for letter in _BLOCK_ORDER}
    dropped_counts: dict[str, int] = {}

    for path in distance_files:
        d = json.loads(path.read_text())

        for name, err in d.get("per_feature_relative_error", {}).items():
            letter = feature_to_block.get(name)
            if letter is None or err != err:  # unknown feature or NaN
                continue
            label = f"{letter}:{name}"
            if label in exclude:
                dropped_counts[label] = dropped_counts.get(label, 0) + 1
                continue
            per_block[letter].append(err)

        for key, w1 in d.get("normalised_w1", {}).items():
            letter = key.split(":", 1)[0]
            label = f"{key} (W1)"
            if w1 != w1:  # NaN
                continue
            if label in exclude:
                dropped_counts[label] = dropped_counts.get(label, 0) + 1
                continue
            per_block.setdefault(letter, []).append(w1)

    return per_block, dropped_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_dir", help="Dataset output directory, e.g. generated/wn18rr_v4.")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: "
                             "data/graph_population/dataset_error_boxplot_<name>.png)")
    parser.add_argument("--exclude", nargs="*", default=None, metavar="LABEL",
                        help="Error labels to drop before plotting, e.g. 'b:a_obj' "
                             f"(default: {sorted(_DEFAULT_EXCLUDE)} — see _DEFAULT_EXCLUDE "
                             "for why). Pass --exclude with no names to disable exclusion.")
    args = parser.parse_args()
    exclude = _DEFAULT_EXCLUDE if args.exclude is None else set(args.exclude)

    dataset_dir = Path(args.dataset_dir)
    distance_files = _iter_distance_files(dataset_dir)
    print(f"Dataset   : {dataset_dir}  ({len(distance_files)} replicas)")

    feature_to_block = _feature_to_block()
    per_block, dropped_counts = _collect_errors(distance_files, feature_to_block, exclude)
    if dropped_counts:
        print(f"  (excluding {len(dropped_counts)} label(s), "
              f"{sum(dropped_counts.values())} sample(s) total: {sorted(dropped_counts)})")

    present = [letter for letter in _BLOCK_ORDER if per_block.get(letter)]
    data = [per_block[letter] for letter in present]
    labels = [_BLOCK_LABELS[letter] for letter in present]
    colours = [_BLOCK_COLOURS[letter] for letter in present]

    for letter, label in zip(present, labels):
        errs = per_block[letter]
        print(f"  {label:<30} n={len(errs):>4}  median={np.median(errs):.3f}  "
              f"mean={np.mean(errs):.3f}  max={np.max(errs):.3f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showmeans=False, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        widths=0.55,
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    ax.set_ylabel("Error  (normalised W1 for distributions, relative error for scalars)")
    ax.set_title(f"Per-feature error by signature block — {dataset_dir.name} "
                 f"({len(distance_files)} replicas)")
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, color="#B0B0B0", linewidth=0.8, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()

    default_name = f"dataset_error_boxplot_{dataset_dir.name}.png"
    out_path = (Path(args.out) if args.out
                else REPO_ROOT / "data" / "graph_population" / default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved     : {out_path}")


if __name__ == "__main__":
    main()
