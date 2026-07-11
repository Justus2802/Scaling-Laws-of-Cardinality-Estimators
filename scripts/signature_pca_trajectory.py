"""Run one roundtrip, snapshot Stage 3 at intervals, and PCA-plot the trajectory.

Generates a synthetic graph from a target's cached signature exactly like
``signature_roundtrip.py``, but instead of only measuring the final output it
also captures the graph right after Stage 2 (before any rewiring) and at
``--num-checkpoints`` equally-spaced points through Stage 3's rewire budget
(via the ``checkpoint_steps``/``checkpoint_callback`` hook on
``generator.stage3.refine``). Each snapshot is measured with the same reduced
signature (Blocks A-F) and projected into the PCA space fit on the corpus (see
``plot_signature_pca.py``), so the plot shows a path from the Stage-2 output
toward (hopefully) the target as Stage 3 progresses.

Snapshots use a lighter Block E sample budget (``--sample-budget``, default
5,000) than a single-shot roundtrip since this script measures 1 + N graphs
instead of one.

Usage
-----
    python scripts/signature_pca_trajectory.py wn18rr_v4
    python scripts/signature_pca_trajectory.py wn18rr_v4 --num-checkpoints 5 --rewire-budget 20000
    python scripts/signature_pca_trajectory.py wn18rr_v4 --size-agnostic
"""

import argparse
import logging
from pathlib import Path

import igraph
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

_REPO = Path(__file__).resolve().parent.parent
from kgsynth.generator import Generator
from kgsynth.signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF
import kgsynth.signature.block_e as _block_e
from kgsynth.motif_counter import HybridMotifCounter

from kgsynth.corpus import DEFAULT_SEARCH_DIRS, load_target_from_corpus
from plot_signature_pca import (
    _find_corpus_signatures, _load_signature_json, _build_matrix,
    _fit_pca_2d, _project,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Block E sample budget for each snapshot's re-measurement. Lower than
# signature_roundtrip.py's 20k default since this script measures several
# graphs per run (1 + num_checkpoints) instead of one.
_SNAPSHOT_SAMPLE_BUDGET = 5_000

# feature name -> owning block letter, for the per-block distance breakdown
# (which block's features are driving the trajectory's movement/drift).
_FEATURE_BLOCK: dict[str, str] = {}
for _letter, _cls in [("a", BlockA), ("b", BlockB), ("c", BlockC),
                       ("d", BlockD), ("e", BlockE), ("f", BlockF)]:
    for _name in _cls.feature_names():
        _FEATURE_BLOCK[_name] = _letter


def _measure_snapshot(g, sample_budget: int) -> dict[str, float]:
    """Measure reduced Blocks A-F on a snapshot graph, returning its feature dict."""
    _block_e.MOTIF_COUNTER = HybridMotifCounter(n_samples=sample_budget, seed=1)
    a = BlockA().calculate(g)
    b = BlockB().calculate(g)
    c = BlockC().calculate(g)
    d = BlockD().calculate(g)
    e = BlockE().calculate(g, sample_budget=sample_budget)
    f = BlockF().calculate(g, skip_shortest_paths=True)
    names = a.feature_names() + b.feature_names() + c.feature_names() + \
        d.feature_names() + e.feature_names() + f.feature_names()
    values = a.as_vector() + b.as_vector() + c.as_vector() + \
        d.as_vector() + e.as_vector() + f.as_vector()
    return dict(zip(names, values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graph", help="Corpus graph name (e.g. 'wn18rr_v4').")
    parser.add_argument("--graphs-dir", default=None,
                        help="Corpus root holding <graph>/signature/. Default: "
                             "searches data/graphs/ then data/test_graphs/.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rewire-budget", type=int, default=20_000,
                        help="Stage 3 rewire budget (default 20000 — smaller than "
                             "signature_roundtrip.py's 50000 default since this script "
                             "runs 1 + num_checkpoints extra measurements).")
    parser.add_argument("--num-checkpoints", type=int, default=3,
                        help="Number of equally-spaced snapshots through Stage 3, in "
                             "addition to the post-Stage-2 (step 0) snapshot and the "
                             "final output (default 3, i.e. points at 25%%/50%%/75%%).")
    parser.add_argument("--sample-budget", type=int, default=_SNAPSHOT_SAMPLE_BUDGET,
                        help=f"Block E motif/path sample budget per snapshot "
                             f"(default {_SNAPSHOT_SAMPLE_BUDGET}).")
    parser.add_argument("--size-agnostic", action="store_true",
                        help="Fit PCA on scale-free structural features only (see "
                             "plot_signature_pca.py --size-agnostic).")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: "
                             "data/graph_population/signature_pca_trajectory_<graph>.png)")
    args = parser.parse_args()

    # ── Step 1: target signature + Stage-2 output (checkpoint 0) ─────────────
    search_dirs = [Path(args.graphs_dir)] if args.graphs_dir else DEFAULT_SEARCH_DIRS
    print(
        f"Loading   : cached target signature for '{args.graph}' "
        f"from {[str(d) for d in search_dirs]}"
    )
    target_sig, _tblocks, _graph_dir = load_target_from_corpus(args.graph, search_dirs)

    # Checkpoint step indices: 0 is the post-Stage-2 graph (no rewiring yet),
    # then num_checkpoints equally-spaced points through [0, rewire_budget],
    # and the budget itself (== the final output, for a consistent x-axis).
    fractions = [i / (args.num_checkpoints + 1) for i in range(1, args.num_checkpoints + 1)]
    mid_steps = sorted({max(1, round(f * args.rewire_budget)) for f in fractions})
    checkpoint_steps = [0] + mid_steps + [args.rewire_budget]

    snapshots: dict[int, "igraph.Graph"] = {}

    def _on_checkpoint(step, g):
        print(f"  Checkpoint step {step}/{args.rewire_budget}: "
              f"{g.vcount():,} nodes, {g.ecount():,} edges")
        snapshots[step] = g

    print(f"Generating: seed={args.seed}, rewire_budget={args.rewire_budget}, "
          f"checkpoints at steps {checkpoint_steps} …")
    g_final = Generator(target_sig).sample(
        seed=args.seed,
        rewire_budget=args.rewire_budget,
        checkpoint_steps=checkpoint_steps,
        checkpoint_callback=_on_checkpoint,
    )
    print(f"  final: {g_final.vcount():,} nodes  {g_final.ecount():,} edges  "
          f"best loss {g_final['stage3_best_loss']:.6f}")

    # ── Step 2: measure every snapshot ────────────────────────────────────────
    print(f"Measuring : {len(checkpoint_steps)} snapshots (sample_budget={args.sample_budget}) …")
    trajectory: list[tuple[int, dict[str, float]]] = []
    for step in checkpoint_steps:
        print(f"  step {step} …")
        feats = _measure_snapshot(snapshots[step], args.sample_budget)
        trajectory.append((step, feats))

    # ── Step 3: PCA basis from the corpus, project target + trajectory ───────
    corpus_signatures = _find_corpus_signatures()
    if not corpus_signatures:
        raise SystemExit("No corpus signatures found under data/graphs/ or data/test_graphs/")
    corpus_names = sorted(corpus_signatures)
    corpus_features = [_load_signature_json(corpus_signatures[name]) for name in corpus_names]
    target_features = _load_signature_json(corpus_signatures[args.graph])

    mat, feature_names = _build_matrix(corpus_features, size_agnostic=args.size_agnostic)
    coords, impute, mean, std, components = _fit_pca_2d(mat)

    mode = "size-agnostic (structural only)" if args.size_agnostic else "raw (all features)"
    print(f"Mode      : {mode}")
    print(f"Features  : {len(feature_names)} (union across corpus)")

    target_xy = _project(target_features, feature_names, impute, mean, std, components)
    traj_xy = np.array([
        _project(feats, feature_names, impute, mean, std, components)
        for _step, feats in trajectory
    ])

    # Full feature-space distance to target (standardized, all `len(feature_names)`
    # dims) — unlike the 2D PCA projection above, this isn't lossy, so it's the
    # right number to check "did the graph actually get closer to the target"
    # rather than reading positions off the (only ~60%-of-variance) PC1/PC2 plot.
    def _standardize(feats: dict) -> np.ndarray:
        vec = np.array([feats.get(name, np.nan) for name in feature_names], dtype=float)
        return (np.where(np.isnan(vec), impute, vec) - mean) / std

    target_z = _standardize(target_features)
    block_letters = sorted(set(_FEATURE_BLOCK.get(n) for n in feature_names) - {None})
    block_masks = {
        letter: np.array([_FEATURE_BLOCK.get(n) == letter for n in feature_names])
        for letter in block_letters
    }

    print()
    header = f"  {'step':>8}  {'full-space dist':>16}  {'PC1,PC2 dist':>14}"
    header += "".join(f"  {'block ' + letter.upper():>10}" for letter in block_letters)
    print(header)
    for (step, feats), (x, y) in zip(trajectory, traj_xy):
        diff = _standardize(feats) - target_z
        full_dist = float(np.linalg.norm(diff))
        pca_dist = float(np.hypot(x - target_xy[0], y - target_xy[1]))
        row = f"  {step:>8}  {full_dist:>16.3f}  {pca_dist:>14.3f}"
        for letter in block_letters:
            block_dist = float(np.linalg.norm(diff[block_masks[letter]]))
            row += f"  {block_dist:>10.3f}"
        print(row)
    print()
    print("  (per-block dist = Euclidean distance restricted to that block's "
          "standardized features; Stage 3 only steers blocks E and F)")
    print()

    # ── Step 4: plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(coords[:, 0], coords[:, 1], c="#B0B0B0", s=40, alpha=0.6,
               label="corpus (original graphs)", zorder=2)
    for name, (x, y) in zip(corpus_names, coords):
        ax.annotate(name, (x, y), fontsize=8, color="#808080",
                    xytext=(4, 4), textcoords="offset points")

    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "-", color="#4C72B0", lw=1.8, alpha=0.8, zorder=3)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(trajectory)))
    for (step, _feats), (x, y), colour in zip(trajectory, traj_xy, cmap):
        label = "post Stage 2" if step == 0 else (
            "final (Stage 3 done)" if step == args.rewire_budget else f"Stage 3 step {step}"
        )
        ax.scatter(x, y, color=colour, s=130, edgecolor="black", linewidth=1.0,
                   zorder=4, label=label)

    ax.scatter(*target_xy, c="#C44E52", s=180, marker="*", edgecolor="black",
               linewidth=1.2, label=f"{args.graph} — target (original)", zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    title_suffix = " — size-agnostic" if args.size_agnostic else ""
    ax.set_title(f"{args.graph}: signature trajectory through Stage 3{title_suffix}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)

    # Zoomed inset around the trajectory + target: Stage 3 preserves the degree
    # sequence (only rewires edges), so its movement in corpus-scale PC1/PC2 is
    # often tiny relative to the spread across distinct graphs — without a zoom
    # the whole trajectory can collapse to a single dot.
    cluster_pts = np.vstack([traj_xy, target_xy[None, :]])
    pad = 0.25 * max(np.ptp(cluster_pts[:, 0]), np.ptp(cluster_pts[:, 1]), 1e-6)
    x0, x1 = cluster_pts[:, 0].min() - pad, cluster_pts[:, 0].max() + pad
    y0, y1 = cluster_pts[:, 1].min() - pad, cluster_pts[:, 1].max() + pad

    axins = zoomed_inset_axes(ax, zoom=1, loc="upper right", borderpad=1.5)
    axins.plot(traj_xy[:, 0], traj_xy[:, 1], "-", color="#4C72B0", lw=1.8, alpha=0.8, zorder=3)
    for (_step, _feats), (x, y), colour in zip(trajectory, traj_xy, cmap):
        axins.scatter(x, y, color=colour, s=130, edgecolor="black", linewidth=1.0, zorder=4)
    axins.scatter(*target_xy, c="#C44E52", s=180, marker="*", edgecolor="black",
                  linewidth=1.2, zorder=5)
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("#555555")
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#555555", lw=0.8, alpha=0.6)

    fig.tight_layout()

    mode_suffix = "_size_agnostic" if args.size_agnostic else ""
    out_path = Path(args.out) if args.out else (
        _REPO / "data" / "graph_population"
        / f"signature_pca_trajectory_{args.graph}{mode_suffix}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved     : {out_path}")


if __name__ == "__main__":
    main()
