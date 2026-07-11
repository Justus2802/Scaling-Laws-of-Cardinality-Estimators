"""PCA-plot one or more sweep_collect.py runs: many synthetic seeds vs. their
targets, in the corpus's signature space.

Reuses the JSONL produced by ``sweep_collect.py`` (``experiments/sweeps/<graph>.jsonl``
+ ``<graph>_target.json``) instead of re-running the generator — if that sweep
hasn't been collected yet for a graph/budget combination, run
``sweep_collect.py`` first. The PCA basis is fit on the corpus (see
``plot_signature_pca.py``); each graph's target and every synthetic run at the
chosen budget are projected into it, so you can see the spread of independent
generator draws around the target as a cloud rather than a single pair. When
multiple graphs are given, each gets its own colour, with per-budget marker
shapes shared across all graphs.

Usage
-----
    python scripts/plot_sweep_pca.py wn18rr_v4
    python scripts/plot_sweep_pca.py wn18rr_v4 swdf
    python scripts/plot_sweep_pca.py wn18rr_v4 --budget 50000
    python scripts/plot_sweep_pca.py wn18rr_v4 --size-agnostic
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
from sweep_viz import _reconstruct_block, _block_feature_map, _LETTERS
from plot_signature_pca import (
    _find_corpus_signatures, _load_signature_json, _build_matrix,
    _fit_pca_2d, _project, _PAIR_COLOURS,
)


def _sig_dict_to_features(sig_dict: dict) -> dict[str, float]:
    """Flatten a sweep_collect.py-style {letter: block_data} dict to feature_name -> value."""
    features: dict[str, float] = {}
    for letter in _LETTERS:
        blk = _reconstruct_block(letter, sig_dict.get(letter))
        features.update(_block_feature_map(blk))
    return features


_BUDGET_MARKERS = ["^", "s", "D", "v", "P", "X"]


def _load_sweep(graph: str, sweep_dir: Path, budget_filter: int | None) -> list[dict]:
    """Load and optionally budget-filter a graph's sweep_collect.py JSONL records."""
    jsonl_path = sweep_dir / f"{graph}.jsonl"
    if not jsonl_path.is_file():
        raise SystemExit(
            f"No sweep data at {jsonl_path}. Run: python scripts/sweep_collect.py {graph}"
        )
    print(f"Loading   : {jsonl_path}")
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    if budget_filter is not None:
        records = [r for r in records if r["budget"] == budget_filter]
        if not records:
            raise SystemExit(f"No runs with budget={budget_filter} in {jsonl_path}")
    print(f"  {len(records)} runs — budgets {sorted({r['budget'] for r in records})}, "
          f"seeds {sorted({r['seed'] for r in records})}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graphs", nargs="+",
                        help="Corpus graph name(s) matching a sweep_collect.py run "
                             "(e.g. 'wn18rr_v4' 'swdf'). Each gets its own colour.")
    parser.add_argument("--sweep-dir", default=None,
                        help="Directory holding <graph>.jsonl + <graph>_target.json "
                             "(default: experiments/sweeps/).")
    parser.add_argument("--budget", type=int, default=None,
                        help="Only plot runs with this rewire_budget (default: all "
                             "budgets present, distinguished by marker shape).")
    parser.add_argument("--size-agnostic", action="store_true",
                        help="Fit PCA on scale-free structural features only (see "
                             "plot_signature_pca.py --size-agnostic).")
    parser.add_argument("--exclude", nargs="+", default=[], metavar="GRAPH",
                        help="Corpus graph name(s) to drop entirely — both from the "
                             "PCA fit and from the plotted cloud (e.g. --exclude aids "
                             "hetionet). None of the plotted graphs can be excluded.")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: "
                             "data/graph_population/signature_pca_sweep_<graphs>.png)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else _REPO / "experiments" / "sweeps"

    # ── load every graph's sweep records ──────────────────────────────────────
    per_graph_records = {graph: _load_sweep(graph, sweep_dir, args.budget) for graph in args.graphs}
    all_budgets = sorted({r["budget"] for records in per_graph_records.values() for r in records})
    budget_markers = {b: m for b, m in zip(all_budgets, _BUDGET_MARKERS)}

    # ── PCA basis from the corpus, project every graph's target + synthetic runs ─
    corpus_signatures = _find_corpus_signatures()
    if not corpus_signatures:
        raise SystemExit("No corpus signatures found under data/graphs/ or data/test_graphs/")
    for graph in args.graphs:
        if graph not in corpus_signatures:
            raise SystemExit(
                f"'{graph}' has no cached corpus signature. "
                f"Available: {sorted(corpus_signatures)}"
            )

    excluded = set(args.exclude)
    unknown_excludes = excluded - set(corpus_signatures)
    if unknown_excludes:
        raise SystemExit(f"--exclude name(s) not in corpus: {sorted(unknown_excludes)}. "
                          f"Available: {sorted(corpus_signatures)}")
    plotted_and_excluded = excluded & set(args.graphs)
    if plotted_and_excluded:
        raise SystemExit(
            f"Cannot exclude {sorted(plotted_and_excluded)} — being plotted as a target graph."
        )
    if excluded:
        print(f"Excluding : {sorted(excluded)} (dropped from PCA fit)")
    corpus_names = sorted(name for name in corpus_signatures if name not in excluded)
    corpus_features = [_load_signature_json(corpus_signatures[name]) for name in corpus_names]

    mat, feature_names = _build_matrix(corpus_features, size_agnostic=args.size_agnostic)
    coords, impute, mean, std, components = _fit_pca_2d(mat)

    mode = "size-agnostic (structural only)" if args.size_agnostic else "raw (all features)"
    print(f"Mode      : {mode}")
    print(f"Features  : {len(feature_names)} (union across corpus)")

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(coords[:, 0], coords[:, 1], c="#B0B0B0", s=40, alpha=0.6,
               label="corpus (original graphs)", zorder=2)
    for name, (x, y) in zip(corpus_names, coords):
        ax.annotate(name, (x, y), fontsize=8, color="#808080",
                    xytext=(4, 4), textcoords="offset points")

    total_runs = 0
    for gi, graph in enumerate(args.graphs):
        colour = _PAIR_COLOURS[gi % len(_PAIR_COLOURS)]
        records = per_graph_records[graph]
        total_runs += len(records)
        target_features = _load_signature_json(corpus_signatures[graph])
        synth_features = [_sig_dict_to_features(r["synth"]) for r in records]

        target_xy = _project(target_features, feature_names, impute, mean, std, components)
        synth_xy = np.array([
            _project(feats, feature_names, impute, mean, std, components)
            for feats in synth_features
        ])

        for (x, y) in synth_xy:
            ax.annotate("", xy=(x, y), xytext=target_xy,
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.0, alpha=0.35),
                        zorder=3)

        graph_budgets = sorted({r["budget"] for r in records})
        for budget in graph_budgets:
            idx = [i for i, r in enumerate(records) if r["budget"] == budget]
            ax.scatter(synth_xy[idx, 0], synth_xy[idx, 1], c=colour, s=130,
                       marker=budget_markers[budget], edgecolor="black", linewidth=1.0, zorder=4,
                       label=f"{graph} — synthetic (budget={budget}, n={len(idx)})")
        for r, (x, y) in zip(records, synth_xy):
            ax.annotate(f"s{r['seed']}", (x, y), fontsize=7, color=colour,
                        xytext=(5, -3), textcoords="offset points")

        ax.scatter(*target_xy, c=colour, s=140, marker="o", edgecolor="black",
                   linewidth=1.2, label=f"{graph} — target (original)", zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    title_suffix = " — size-agnostic" if args.size_agnostic else ""
    graphs_label = ", ".join(args.graphs)
    ax.set_title(f"{graphs_label}: {total_runs} sweep runs in PCA space{title_suffix}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    mode_suffix = "_size_agnostic" if args.size_agnostic else ""
    graphs_suffix = "_".join(args.graphs)
    out_path = Path(args.out) if args.out else (
        _REPO / "data" / "graph_population"
        / f"signature_pca_sweep_{graphs_suffix}{mode_suffix}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved     : {out_path}")


if __name__ == "__main__":
    main()
