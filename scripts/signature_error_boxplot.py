"""Boxplot of per-feature/per-distribution relative errors, grouped by signature block.

Loads a target ``signature/`` and a synthetic ``signature_synth*/`` directory
(roundtrip output, as produced by ``scripts/signature_roundtrip.py``) and
computes, for each block, one error value per reported quantity:

- **Distribution features** (quantile functions, power-law/Zipf/truncated-
  power-law/exp-decay fits — anything a block's ``distribution_fits()``
  reports) use the **normalised Wasserstein-1 distance** (``W1 / target IQR``,
  via ``signature._distance``) between the reconstructed target and synthetic
  distributions, not a per-parameter relative error. Comparing fitted
  parameters directly is misleading — an unstable shape parameter (e.g. a
  power-law alpha estimated from few points) can swing wildly between two
  distributions that are, in W1 terms, nearly identical, and conversely two
  parameter sets can look close while describing distributions that differ a
  lot in mass. W1 measures the actual distributional mismatch instead.
- **Standalone scalars** (counts, ratios, single numbers not part of a
  reported distribution — e.g. ``num_entities``, ``edge_multiplicity``,
  ``clustering_coefficient``) keep plain relative error
  ``|target - synth| / max(|target|, eps)``, since there is no distribution to
  reconstruct.

This is the per-block companion to ``signature_roundtrip.py``'s scalar
mean/median/max and its separate W1 table — it shows *which* blocks carry the
roundtrip error on one comparable 0-ish-to-1-ish scale, instead of two
disjoint tables.

A couple of extreme outliers (``b:a_obj``, ``d:cs_freq (W1)``) are excluded by
default — see ``_DEFAULT_EXCLUDE`` for why each dwarfs every other value
without reflecting a comparable magnitude of real drift.

Usage
-----
    python scripts/signature_error_boxplot.py wn18rr_v4
    python scripts/signature_error_boxplot.py wn18rr_v4 --synth-dir signature_synth_20260706_184120
    python scripts/signature_error_boxplot.py wn18rr_v4 --out data/graph_population/error_boxplot.png
    python scripts/signature_error_boxplot.py wn18rr_v4 --exclude   # disable exclusion
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from kgsynth.signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF
from kgsynth.signature import _distance
from kgsynth.signature._fits import QUANTILE_SUFFIXES

_REPO = Path(__file__).resolve().parent.parent

_DEFAULT_SEARCH_DIRS = [_REPO / "data" / "graphs", _REPO / "data" / "test_graphs"]

# Fixed block order + one colour per block (categorical, never re-cycled).
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

# Excluded by default:
#   - a_obj: a near-zero-by-construction OLS offset, so a small absolute miss
#     produces a runaway relative error (denominator artifact). Observed
#     ~4.3 on wn18rr_v4 vs. every other value under ~1.0.
#   - cs_freq (W1): the target's CS-frequency distribution has a very narrow
#     IQR, so W1/IQR normalisation inflates even a modest absolute mismatch.
#     Observed ~3.1 on wn18rr_v4.
#   - path/tree template Zipf+entropy (Block E, k=2..10 plus the two tree
#     scalars): excluded by request, not a numeric-artifact case like the two
#     above.
#   - shortest-path stats (Block F: max/mean/var): excluded by request, not a
#     numeric-artifact case like a_obj/cs_freq above.
_TEMPLATE_FEATURES = (
    [f"e:path_template_zipf_k{k}" for k in range(2, 11)]
    + [f"e:path_template_entropy_k{k}" for k in range(2, 11)]
    + ["e:tree_template_zipf", "e:tree_template_entropy"]
)
_SHORTEST_PATH_FEATURES = ["f:shortest_path_max", "f:shortest_path_mean", "f:shortest_path_var"]
_DEFAULT_EXCLUDE = {"b:a_obj", "d:cs_freq (W1)", *_TEMPLATE_FEATURES, *_SHORTEST_PATH_FEATURES}


def _load_blocks(sig_dir: Path) -> dict[str, object]:
    """Load every available ``block_*.json`` under ``sig_dir`` into block objects.

    Loading block state directly (rather than the combined ``signature.json``)
    avoids a staleness trap: the combined file is not regenerated when an
    individual ``block_*.json`` is updated, so it can lag behind — the same
    source ``signature_roundtrip.py`` uses.
    """
    blocks: dict[str, object] = {}
    for letter, cls in _BLOCK_CLASSES.items():
        path = sig_dir / f"block_{letter}.json"
        if path.is_file():
            blocks[letter] = cls.from_serializable(json.loads(path.read_text()))
    return blocks


def _find_target(graph_name: str, search_dirs: list[Path]) -> Path:
    """Return the ``signature/`` directory for ``graph_name``."""
    for d in search_dirs:
        candidate = d / graph_name / "signature"
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        f"No cached target signature for '{graph_name}' under {[str(d) for d in search_dirs]}"
    )


def _resolve_synth(graph_dir: Path, synth_dir: str | None) -> Path:
    """Return the synthetic signature directory.

    ``synth_dir`` may be an explicit subdirectory name (e.g.
    ``signature_synth_20260706_184120``) or omitted, in which case the most
    recently modified ``signature_synth*/`` sibling is used.
    """
    if synth_dir:
        path = graph_dir / synth_dir
        if not path.is_dir():
            raise SystemExit(f"No such directory: {path}")
        return path
    candidates = sorted(
        (p.parent for p in graph_dir.glob("signature_synth*/signature.json")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(
            f"No signature_synth*/signature.json under {graph_dir}. "
            f"Run: python scripts/signature_roundtrip.py <graph>"
        )
    return candidates[-1]


def _scalar_relative_error(tv: float, sv: float) -> float:
    """Plain relative error ``|target - synth| / max(|target|, eps)``, NaN if either is NaN."""
    if tv != tv or sv != sv:
        return float("nan")
    return abs(tv - sv) / max(abs(tv), 1e-9)


def _block_errors(letter: str, tblk: object, sblk: object) -> dict[str, float]:
    """Return {label: error} for one block, split into distributions (W1/IQR) and scalars.

    Features covered by ``distribution_fits()`` are scored once per
    distribution (not once per stored parameter) via normalised Wasserstein-1;
    every other feature name keeps plain scalar relative error.
    """
    errs: dict[str, float] = {}

    dist_fits = tblk.distribution_fits() if hasattr(tblk, "distribution_fits") else []
    sdist_fits = dict((n, f) for n, f, _ in sblk.distribution_fits()) if hasattr(sblk, "distribution_fits") else {}
    for name, tfit, kind in dist_fits:
        sfit = sdist_fits.get(name)
        if sfit is None:
            continue
        w1 = _distance.wasserstein1(tfit, sfit, kind)
        iqr = _distance.reconstructed_iqr(tfit, kind)
        w1_norm = w1 / iqr if (iqr is not None and iqr > 0) else float("nan")
        if w1_norm == w1_norm:  # not NaN
            errs[f"{letter}:{name} (W1)"] = w1_norm

    # Parameter names folded into a distribution_fits() entry above are listed
    # in _DIST_PARAM_FEATURES and skipped in the scalar pass below, so each
    # quantity is scored exactly once.
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


# Feature names subsumed by a distribution_fits() entry (scored via W1 above),
# so the scalar pass doesn't double-count them under a per-parameter relative
# error. Built from each block's known distribution_fits() -> feature_names()
# layout (docs/signature.md's per-block vector layout).
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", help="Corpus graph name (e.g. 'wn18rr_v4').")
    parser.add_argument("--graphs-dir", default=None,
                        help="Corpus root holding <graph>/signature/. Default: "
                             "searches data/graphs/ then data/test_graphs/.")
    parser.add_argument("--synth-dir", default=None,
                        help="Synthetic signature subdirectory name under <graph>/ "
                             "(default: the most recently modified signature_synth*/).")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: "
                             "data/graph_population/signature_error_boxplot_<graph>.png)")
    parser.add_argument("--exclude", nargs="*", default=None, metavar="LABEL",
                        help="Error labels to drop before plotting, e.g. 'b:a_obj' "
                             f"(default: {sorted(_DEFAULT_EXCLUDE)} — see _DEFAULT_EXCLUDE "
                             "for why). Pass --exclude with no names to disable exclusion.")
    args = parser.parse_args()
    exclude = _DEFAULT_EXCLUDE if args.exclude is None else set(args.exclude)

    search_dirs = [Path(args.graphs_dir)] if args.graphs_dir else _DEFAULT_SEARCH_DIRS
    target_dir = _find_target(args.graph, search_dirs)
    graph_dir = target_dir.parent
    synth_dir = _resolve_synth(graph_dir, args.synth_dir)

    print(f"Target    : {target_dir}")
    print(f"Synthetic : {synth_dir}")

    tblocks = _load_blocks(target_dir)
    sblocks = _load_blocks(synth_dir)

    # Group errors by owning block, in fixed block order.
    per_block: dict[str, list[float]] = {letter: [] for letter in _BLOCK_ORDER}
    for letter in _BLOCK_ORDER:
        if letter not in tblocks or letter not in sblocks:
            continue
        block_errs = _block_errors(letter, tblocks[letter], sblocks[letter])
        dropped = exclude & block_errs.keys()
        if dropped:
            print(f"  (excluding {len(dropped)} outlier(s): {sorted(dropped)})")
        per_block[letter] = [v for k, v in block_errs.items() if k not in exclude]

    present = [letter for letter in _BLOCK_ORDER if per_block[letter]]
    data = [per_block[letter] for letter in present]
    labels = [_BLOCK_LABELS[letter] for letter in present]
    colours = [_BLOCK_COLOURS[letter] for letter in present]

    for letter, label in zip(present, labels):
        errs = per_block[letter]
        print(f"  {label:<30} n={len(errs):>3}  median={np.median(errs):.3f}  "
              f"mean={np.mean(errs):.3f}  max={np.max(errs):.3f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showmeans=False,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="#808080",
                        markeredgecolor="none", alpha=0.6),
        widths=0.55,
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    ax.set_ylabel("Error  (normalised W1 for distributions, relative error for scalars)")
    ax.set_title(f"Per-feature error by signature block on {args.graph}")
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, color="#B0B0B0", linewidth=0.8, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()

    default_name = f"signature_error_boxplot_{args.graph}.png"
    out_path = Path(args.out) if args.out else _REPO / "data" / "graph_population" / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved     : {out_path}")


if __name__ == "__main__":
    main()
