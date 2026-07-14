"""Run the full pipeline on one graph file and PCA-plot a synthetic population.

Given a path to any ``.ttl``/``.nt`` knowledge graph, this runs the complete
measure -> generate -> re-measure loop end to end, repeated across ``--num-graphs``
seeds, and places the whole result in signature space:

1. **measure** the input graph's reduced signature once (the target);
2. **generate** ``--num-graphs`` synthetic graphs that target it, one per seed
   (Stages 1/2/3) — the population the repeated generation loop produces;
3. **re-measure** every synthetic graph's signature;
4. **project** the target and all synthetic points into a 2D PCA basis fit on
   the *corpus* of real-KG signatures (the grey cloud), drawing the synthetic
   population as a cloud around the target with faint arrows.

The PCA basis is fit on the corpus only (never on the new points), so the axes
reflect real cross-graph variance; how tightly the cloud surrounds the target
vs. drifts to one side separates generator variance from systematic bias.
Unlike ``plot_signature_pca.py`` (which needs a cached ``signature_synth/``
sibling and a corpus *name*), this script works from a raw graph path and
produces every synthetic graph itself.

Runtime note: full-fidelity measurement runs ``1 + num_graphs`` times and is
dominated by Block E's colour-coding sampler, so a few-thousand-vertex graph at
the default ``--num-graphs`` takes a while. Use ``--sample-budget`` to trade
accuracy for speed.

Usage
-----
    python scripts/roundtrip_pca.py data/graphs/swdf/swdf.nt
    python scripts/roundtrip_pca.py mygraph.ttl --num-graphs 20 --rewire-budget 50000
    python scripts/roundtrip_pca.py mygraph.ttl --size-agnostic --out fig.png
    python scripts/roundtrip_pca.py mygraph.ttl --sample-budget 5000   # faster
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from kgsynth import Generator, Signature, load_kg
import kgsynth.signature.block_e as _block_e
from kgsynth.motif_counter import HybridMotifCounter

from plot_signature_pca import (
    _find_corpus_signatures,
    _load_signature_json,
    _build_matrix,
    _fit_pca_2d,
    _project,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("roundtrip_pca")


def _measure_features(g, sample_budget: int | None) -> dict[str, float]:
    """Measure all six reduced blocks on ``g`` and return the flat feature dict.

    When ``sample_budget`` is given, Block E's motif/path sampler is rebound to
    that budget (fewer walks -> faster, noisier); otherwise the 100k default is
    used. ``Signature.from_graph`` measures every block, and ``as_features``
    flattens to the same ``{name: value}`` mapping a ``signature.json`` stores.
    """
    if sample_budget is not None:
        _block_e.MOTIF_COUNTER = HybridMotifCounter(n_samples=sample_budget, seed=1)
    return Signature.from_graph(g).as_features()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kg_file", type=Path, help="Path to the input KG (.ttl or .nt).")
    parser.add_argument("-n", "--num-graphs", type=int, default=10,
                        help="Number of synthetic graphs to generate from the same "
                             "target, one per seed (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="First generator seed; subsequent graphs use seed+1, "
                             "seed+2, … (default: 42).")
    parser.add_argument("--rewire-budget", type=int, default=50_000,
                        help="Stage-3 rewiring attempts per graph (default: 50000).")
    parser.add_argument("--sample-budget", type=int, default=None,
                        help="Block E motif/path sample budget for BOTH measurements "
                             "(default: the 100k full-fidelity budget). Lower is faster.")
    parser.add_argument("--size-agnostic", action="store_true",
                        help="Fit PCA on scale-free structural features only (drops "
                             "size-dependent features; see plot_signature_pca.py).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output image path (default: <kg-stem>_roundtrip_pca.png "
                             "in the current directory).")
    args = parser.parse_args()

    if not args.kg_file.is_file():
        raise SystemExit(f"No such KG file: {args.kg_file}")

    n = args.num_graphs
    total_steps = 1 + 2 * n

    # ── Step 1: measure the input graph once (target) ─────────────────────────
    log.info("1/%d measuring target signature of %s", total_steps, args.kg_file.name)
    g_in = load_kg(args.kg_file)
    target_feats = _measure_features(g_in, args.sample_budget)
    target_sig = Signature.from_features(target_feats)

    # ── Steps 2..(1+2n): generate + re-measure n synthetic graphs ─────────────
    # Each graph is generated independently from the same target signature, one
    # seed apart, mirroring the repeated generation loop's population output.
    synth_feats_list: list[dict[str, float]] = []
    for i in range(n):
        seed = args.seed + i
        step = 2 + 2 * i
        log.info("%d/%d generating synthetic KG %d/%d (seed=%d, rewire_budget=%d)",
                 step, total_steps, i + 1, n, seed, args.rewire_budget)
        g_synth = Generator(target_sig).sample(seed=seed, rewire_budget=args.rewire_budget)
        log.info("    synthetic KG %d/%d: %d vertices, %d edges",
                 i + 1, n, g_synth.vcount(), g_synth.ecount())

        log.info("%d/%d re-measuring synthetic KG %d/%d", step + 1, total_steps, i + 1, n)
        synth_feats_list.append(_measure_features(g_synth, args.sample_budget))

    # ── Final step: PCA basis from the corpus, project target + population ────
    log.info("fitting corpus PCA basis and projecting")
    corpus_signatures = _find_corpus_signatures()
    if not corpus_signatures:
        raise SystemExit(
            "No corpus signatures found under data/graphs/ or data/test_graphs/ "
            "to fit the PCA basis on."
        )
    corpus_names = sorted(corpus_signatures)
    corpus_features = [_load_signature_json(corpus_signatures[n_]) for n_ in corpus_names]

    mat, feature_names = _build_matrix(corpus_features, size_agnostic=args.size_agnostic)
    coords, impute, mean, std, components = _fit_pca_2d(mat)

    t_xy = _project(target_feats, feature_names, impute, mean, std, components)
    synth_xy = np.array([
        _project(feats, feature_names, impute, mean, std, components)
        for feats in synth_feats_list
    ])

    mode = "size-agnostic (structural only)" if args.size_agnostic else "raw (all features)"
    drifts = np.linalg.norm(synth_xy - t_xy, axis=1)
    log.info("    corpus: %d graphs, %d features (%s); %d synthetic graphs; "
             "PC drift from target: mean=%.3f std=%.3f",
             len(corpus_names), len(feature_names), mode, n, drifts.mean(), drifts.std())

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c="#B0B0B0", s=40, alpha=0.7,
               label="corpus (real graphs)", zorder=2)
    for name, (x, y) in zip(corpus_names, coords):
        ax.annotate(name, (x, y), fontsize=8, color="#808080",
                    xytext=(4, 4), textcoords="offset points")

    label = args.kg_file.stem
    for x, y in synth_xy:
        ax.annotate("", xy=(x, y), xytext=t_xy,
                    arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.0, alpha=0.35),
                    zorder=3)
    ax.scatter(synth_xy[:, 0], synth_xy[:, 1], c="#4C72B0", s=90, marker="^",
               edgecolor="black", linewidth=1.0, alpha=0.85, zorder=4,
               label=f"{label} — synthetic (n={n})")
    ax.scatter(*t_xy, c="#C44E52", s=180, marker="*", edgecolor="black", linewidth=1.2,
               label=f"{label} — target (measured)", zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    suffix = " — size-agnostic" if args.size_agnostic else ""
    ax.set_title(f"Synthetic population in corpus PCA space: {label} (n={n}){suffix}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = args.out or Path(f"{label}_roundtrip_pca.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("saved %s", out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
