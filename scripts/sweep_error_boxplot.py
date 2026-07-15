"""Boxplot of per-feature/per-distribution relative errors for a sweep_collect.py run.

Loads a target signature (``<graph>_target.json``) and every synthetic record in
a ``sweep_collect.py`` JSONL file, and pools their per-feature errors by
signature block — the ``sweep_collect.py`` companion to
``dataset_error_boxplot.py`` (which reads ``kgsynth dataset --measure`` output)
and ``signature_error_boxplot.py`` (which compares a single roundtrip pair).

Per record, per block:

- **Distribution features** (quantile functions, power-law/Zipf/truncated-
  power-law/exp-decay fits — anything a block's ``distribution_fits()``
  reports) use the normalised Wasserstein-1 distance (``W1 / target IQR``)
  between the reconstructed target and synthetic distributions.
- **Standalone scalars** (counts, ratios, single numbers not part of a
  reported distribution) keep plain relative error
  ``|target - synth| / max(|target|, eps)``.

Every record in the JSONL file contributes one error sample per feature, so
each box shows the spread across the whole sweep (all budgets/seeds pooled)
rather than a single draw.

Usage
-----
    python scripts/sweep_error_boxplot.py experiments/sweeps/wn18rr_v4.jsonl
    python scripts/sweep_error_boxplot.py experiments/sweeps/wn18rr_v4.jsonl \\
        --out fig.png
    python scripts/sweep_error_boxplot.py experiments/sweeps/wn18rr_v4.jsonl --exclude
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from kgsynth.signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF
from kgsynth.signature import _distance
from kgsynth.signature._fits import QUANTILE_SUFFIXES
from kgsynth.corpus import REPO_ROOT

# Fixed block order + one colour per block (categorical, never re-cycled) —
# kept identical to signature_error_boxplot.py / dataset_error_boxplot.py so
# all three figures read the same way.
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

# Feature names subsumed by a distribution_fits() entry (scored via W1), so the
# scalar pass doesn't double-count them under a per-parameter relative error.
_DIST_PARAM_FEATURES: dict[str, set[str]] = {
    "b": {
        "out_degree_alpha", "out_degree_xmin",
        "in_degree_alpha", "in_degree_xmin",
        "relation_zipf_exponent", "relation_zipf_xmin",
        *[f"obj_mult_alpha_{s}" for s in QUANTILE_SUFFIXES],
        *[f"subj_mult_alpha_{s}" for s in QUANTILE_SUFFIXES],
    },
    "c": {
        "class_size_alpha", "class_size_xmin",
        "subj_cooc_rate", "subj_cooc_scale",
        "obj_cooc_rate", "obj_cooc_scale",
        *[f"subj_row_entropy_{s}" for s in QUANTILE_SUFFIXES],
        *[f"obj_row_entropy_{s}" for s in QUANTILE_SUFFIXES],
    },
    "d": {
        "cs_freq_alpha", "cs_freq_vmin", "cs_freq_vmax",
        *[f"cs_size_{s}" for s in QUANTILE_SUFFIXES],
        "inv_cs_freq_alpha", "inv_cs_freq_vmin", "inv_cs_freq_vmax",
        *[f"inv_cs_size_{s}" for s in QUANTILE_SUFFIXES],
        "two_step_alpha", "two_step_vmin", "two_step_vmax",
    },
}


def _reconstruct_blocks(sig_dict: dict) -> dict[str, object]:
    """Reconstruct every available block from a sweep record's serialized dict."""
    blocks: dict[str, object] = {}
    for letter, cls in _BLOCK_CLASSES.items():
        data = sig_dict.get(letter)
        if data is not None:
            blocks[letter] = cls.from_serializable(data)
    return blocks


def _scalar_relative_error(tv: float, sv: float) -> float:
    """Plain relative error ``|target - synth| / max(|target|, eps)``, NaN if either is NaN."""
    if tv != tv or sv != sv:
        return float("nan")
    return abs(tv - sv) / max(abs(tv), 1e-9)


def _block_errors(letter: str, tblk: object, sblk: object) -> dict[str, float]:
    """Return {label: error} for one block, split into distributions (W1/IQR) and scalars."""
    errs: dict[str, float] = {}

    dist_fits = tblk.distribution_fits() if hasattr(tblk, "distribution_fits") else []
    sdist_fits = (
        dict((n, f) for n, f, _ in sblk.distribution_fits())
        if hasattr(sblk, "distribution_fits")
        else {}
    )
    for name, tfit, kind in dist_fits:
        sfit = sdist_fits.get(name)
        if sfit is None:
            continue
        w1 = _distance.wasserstein1(tfit, sfit, kind)
        iqr = _distance.reconstructed_iqr(tfit, kind)
        w1_norm = w1 / iqr if (iqr is not None and iqr > 0) else float("nan")
        if w1_norm == w1_norm:  # not NaN
            errs[f"{letter}:{name} (W1)"] = w1_norm

    tvec = dict(zip(tblk.feature_names(), tblk.as_vector()))
    svec = dict(zip(sblk.feature_names(), sblk.as_vector()))
    for name, tv in tvec.items():
        if name in _DIST_PARAM_FEATURES.get(letter, set()):
            continue
        sv = svec.get(name)
        if sv is None:
            continue
        err = _scalar_relative_error(tv, sv)
        if err == err:  # not NaN
            errs[f"{letter}:{name}"] = err
    return errs


def _load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"{jsonl_path} is empty.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("jsonl", type=Path, help="JSONL file produced by sweep_collect.py.")
    parser.add_argument("--target", type=Path, default=None,
                        help="Target JSON file (default: <graph>_target.json next to the JSONL).")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: "
                             "data/graph_population/sweep_error_boxplot_<graph>.png)")
    parser.add_argument("--exclude", nargs="*", default=None, metavar="LABEL",
                        help="Error labels to drop before plotting, e.g. 'b:a_obj' "
                             f"(default: {sorted(_DEFAULT_EXCLUDE)} — see _DEFAULT_EXCLUDE "
                             "for why). Pass --exclude with no names to disable exclusion.")
    args = parser.parse_args()
    exclude = _DEFAULT_EXCLUDE if args.exclude is None else set(args.exclude)

    records = _load_records(args.jsonl)
    graph_name = records[0]["graph"]

    target_path = args.target or (args.jsonl.parent / f"{graph_name}_target.json")
    if not target_path.exists():
        raise SystemExit(f"Target file not found: {target_path}\n"
                          "Run sweep_collect.py first, or pass --target.")
    target_dict = json.loads(target_path.read_text())
    tblocks = _reconstruct_blocks(target_dict)

    print(f"Sweep     : {args.jsonl}  ({len(records)} records)")
    print(f"Target    : {target_path}")

    per_block: dict[str, list[float]] = {letter: [] for letter in _BLOCK_ORDER}
    dropped_counts: dict[str, int] = {}

    for rec in records:
        sblocks = _reconstruct_blocks(rec["synth"])
        for letter in _BLOCK_ORDER:
            if letter not in tblocks or letter not in sblocks:
                continue
            block_errs = _block_errors(letter, tblocks[letter], sblocks[letter])
            for label, err in block_errs.items():
                if label in exclude:
                    dropped_counts[label] = dropped_counts.get(label, 0) + 1
                    continue
                per_block[letter].append(err)

    if dropped_counts:
        print(f"  (excluding {len(dropped_counts)} label(s), "
              f"{sum(dropped_counts.values())} sample(s) total: {sorted(dropped_counts)})")

    present = [letter for letter in _BLOCK_ORDER if per_block[letter]]
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
    ax.set_title(f"Per-feature error by signature block — {graph_name} sweep "
                 f"({len(records)} records)")
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, color="#B0B0B0", linewidth=0.8, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()

    default_name = f"sweep_error_boxplot_{graph_name}.png"
    out_path = (Path(args.out) if args.out
                else REPO_ROOT / "data" / "graph_population" / default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved     : {out_path}")


if __name__ == "__main__":
    main()
