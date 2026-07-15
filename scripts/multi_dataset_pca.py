"""Generate synthetic populations for multiple corpus graphs and PCA-plot them together.

Given two or more corpus graph names, each with a cached signature
(``data/graphs/<name>/`` or ``data/test_graphs/<name>/``), this drives
``kgsynth.dataset`` — the same ``WorkUnit`` / ``run_unit`` worker machinery
``kgsynth dataset`` uses, run through one shared process pool — to generate
``--num-graphs`` independent synthetic replicas of *each* target (an
``Identity`` transform: no perturbation, only the seed differs per replica).

All targets and all synthetic replicas, across every base graph, are then
projected into a single 2D PCA basis fit on the corpus of real-KG signatures,
and plotted together in one figure: every target in **red**, every synthetic
replica in **blue** (marker shape distinguishes which base graph a point
belongs to; a legend maps shapes to names).

The target must already be measured (``kgsynth measure <file>``) so its
``signature/block_e.json`` is cached — ``kgsynth.dataset.worker`` refuses to
measure Block E inside a worker process, since ``load_kg``'s hash-ordered
vertex numbering makes a seeded colour-coding estimate non-reproducible across
processes (see ``kgsynth.dataset.config._require_cached_block_e``).

Usage
-----
    kgsynth measure data/test_graphs/wn18rr_v4/wn18rr_v4.nt   # once, if uncached
    kgsynth measure data/test_graphs/fb237_v4/fb237_v4.nt     # once, if uncached
    python scripts/multi_dataset_pca.py wn18rr_v4 fb237_v4
    python scripts/multi_dataset_pca.py wn18rr_v4 fb237_v4 --num-graphs 10 --workers 20
    python scripts/multi_dataset_pca.py wn18rr_v4 fb237_v4 --out both.png
    # Stage 1+2 only for fb237_v4 (Stage 3 effectively skipped via budget=1):
    python scripts/multi_dataset_pca.py wn18rr_v4 fb237_v4 --num-graphs 10 --workers 20 \\
        --rewire-budget-for fb237_v4=1
    # wn18rr_v4 already generated (e.g. an earlier --keep-graphs run) — only
    # generate fb237_v4, reusing wn18rr_v4's replicas from disk:
    python scripts/multi_dataset_pca.py wn18rr_v4 fb237_v4 --num-graphs 10 --workers 20 \\
        --load-existing wn18rr_v4=generated/wn18rr_v4
"""

import argparse
import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from kgsynth.corpus import DEFAULT_SEARCH_DIRS
from kgsynth.dataset.plan import WorkUnit
from kgsynth.dataset.worker import run_unit
from kgsynth.transform import Identity

from plot_signature_pca import (
    _find_corpus_signatures,
    _load_signature_json,
    _build_matrix,
    _fit_pca_2d,
    _project,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("multi_dataset_pca")

# Each worker is single-threaded Python; letting numpy/BLAS spawn its own threads
# inside every one of them just oversubscribes the cores (mirrors
# kgsynth.dataset.runner._pin_blas_threads). Set before the pool is created so
# spawned children inherit it.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# Distinct marker shapes for each base graph's synthetic population (colour
# alone is shared — red targets, blue synthetics — per the plot spec).
_MARKERS = ("^", "s", "D", "v", "P", "X", "o")


def _require_cached_target(graph: str) -> Path:
    """Return *graph*'s cached ``signature/`` dir, or raise with a fix-it message.

    :param graph: Corpus graph name.
    :raises SystemExit: If the graph isn't in the corpus, or has no cached
        ``block_e.json`` (the same precondition ``kgsynth dataset`` enforces).
    """
    for root in DEFAULT_SEARCH_DIRS:
        sig_dir = root / graph / "signature"
        if sig_dir.is_dir():
            if (sig_dir / "block_e.json").exists():
                return sig_dir
            raise SystemExit(
                f"'{graph}' has no cached block_e.json ({sig_dir}). Run "
                f"`kgsynth measure <path-to-{graph}>` first — this script (like "
                f"`kgsynth dataset`) never measures Block E inside a worker "
                f"process, since that would not be reproducible across workers."
            )
    available = sorted(
        p.name for root in DEFAULT_SEARCH_DIRS if root.is_dir()
        for p in root.iterdir() if (p / "signature").is_dir()
    )
    raise SystemExit(f"'{graph}' not found in the corpus. Available: {available}")


def _build_units(graph: str, n: int, seed: int, out_dir: Path,
                  generator_opts: dict) -> list[WorkUnit]:
    """``n`` unperturbed replicas of *graph*'s cached target — one seed each.

    Mirrors ``kgsynth.dataset.plan.build_units``'s seeding: one
    :class:`~numpy.random.SeedSequence` spawned per unit, so a unit's seeds
    depend only on its index, never on worker count or completion order. Each
    base graph gets its own seed sequence (seeded from the same master seed),
    so two base graphs never collide on the same generate/perturb seed pair.
    """
    children = np.random.SeedSequence(seed).spawn(n)
    width = max(4, len(str(n - 1)))
    return [
        WorkUnit(
            index=i,
            out_dir=out_dir / f"graph_{i:0{width}d}",
            base=graph,
            transform=Identity(),
            label="replica",
            perturb_seed=int(seeds[0]),
            generate_seed=int(seeds[1]),
            measure=True,  # writes achieved.json — the PCA projection reads it
            generator_opts=tuple(sorted(generator_opts.items())),
        )
        for i, seeds in enumerate(c.generate_state(2) for c in children)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graphs", nargs="+",
                        help="Two or more corpus graph names (e.g. 'wn18rr_v4 fb237_v4'); "
                             "each must already have a cached signature/block_e.json.")
    parser.add_argument("-n", "--num-graphs", type=int, default=10,
                        help="Number of synthetic replicas per base graph (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed; each base graph's replica seeds are spawned "
                             "from it independently via numpy.random.SeedSequence "
                             "(default: 42).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes, shared across all base graphs' "
                             "units (default: CPU count).")
    parser.add_argument("--rewire-budget", type=int, default=50_000,
                        help="Stage-3 rewiring attempts per graph, unless overridden for "
                             "a specific graph via --rewire-budget-for (default: 50000).")
    parser.add_argument("--rewire-budget-for", action="append", default=[],
                        metavar="GRAPH=BUDGET",
                        help="Per-graph override, e.g. 'fb237_v4=1' to effectively skip "
                             "Stage 3 for that graph only (Stage 1+2 only). Repeatable.")
    parser.add_argument("--size-agnostic", action="store_true",
                        help="Fit PCA on scale-free structural features only (drops "
                             "size-dependent features; see plot_signature_pca.py).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output image path (default: signature_pca_multi_<graphs>.png "
                             "in the current directory).")
    parser.add_argument("--keep-graphs", type=Path, default=None,
                        help="Copy the generated .ttl graphs + metadata here instead of "
                             "discarding them after the plot is made.")
    parser.add_argument("--load-existing", action="append", default=[],
                        metavar="GRAPH=DIR",
                        help="Skip generation for GRAPH and instead load its replicas from "
                             "DIR/graph_*/achieved.json (e.g. a --keep-graphs directory from "
                             "an earlier run). Repeatable, one per graph.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Crop the y-axis (PC2) to this fixed range, e.g. "
                             "'--ylim -2.5 4.5'. Points outside the range are simply not "
                             "shown (no effect on the PCA fit itself). Default: full range.")
    args = parser.parse_args()

    if len(args.graphs) < 2:
        log.warning("only one graph given (%s); the combined plot works with any "
                     "number, but you asked for a multi-graph comparison.", args.graphs[0])

    # ── Parse per-graph rewire-budget overrides (e.g. "fb237_v4=1") ───────────
    rewire_budgets = {g: args.rewire_budget for g in args.graphs}
    for spec in args.rewire_budget_for:
        graph, _, budget = spec.partition("=")
        if graph not in rewire_budgets:
            raise SystemExit(f"--rewire-budget-for '{spec}': '{graph}' is not one of {args.graphs}")
        rewire_budgets[graph] = int(budget)

    # ── Validate every base graph up front — a typo on graph 2 must not cost a
    # completed run of graph 1's (possibly slow) generation. ──────────────────
    sig_dirs = {g: _require_cached_target(g) for g in args.graphs}
    target_feats = {g: _load_signature_json(sig_dirs[g] / "signature.json") for g in args.graphs}

    # ── --load-existing GRAPH=DIR: skip generation for GRAPH entirely and read
    # its replicas' achieved.json straight from an earlier run's directory. ───
    existing_dirs: dict[str, Path] = {}
    for spec in args.load_existing:
        graph, _, dir_str = spec.partition("=")
        if graph not in args.graphs:
            raise SystemExit(f"--load-existing '{spec}': '{graph}' is not one of {args.graphs}")
        existing_dirs[graph] = Path(dir_str)

    preloaded_feats: dict[str, list[dict]] = {}
    for g, d in existing_dirs.items():
        replica_dirs = sorted(p for p in d.glob("graph_*") if (p / "achieved.json").exists())
        if not replica_dirs:
            raise SystemExit(f"--load-existing {g}={d}: no graph_*/achieved.json found.")
        preloaded_feats[g] = [_load_signature_json(p / "achieved.json") for p in replica_dirs]
        log.info("%-12s loaded %d existing replicas from %s (skipping generation)",
                 g, len(replica_dirs), d)

    graphs_to_generate = [g for g in args.graphs if g not in existing_dirs]

    n = args.num_graphs
    work_dir = Path(args.keep_graphs) if args.keep_graphs else Path(tempfile.mkdtemp(
        prefix="multi_dataset_pca_"
    ))
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Build units only for graphs that still need generating. Each unit's
    # out_dir is work_dir/<base_graph>/graph_NNNN, so the base graph a result
    # belongs to is read straight off out_dir's parent — no separate lookup
    # table needed. ─────────────────────────────────────────────────────────
    all_units: list[WorkUnit] = []
    for g in graphs_to_generate:
        all_units.extend(_build_units(
            g, n, args.seed, work_dir / g,
            generator_opts={"rewire_budget": rewire_budgets[g]},
        ))

    results = []
    if all_units:
        workers = args.workers or os.cpu_count() or 1
        workers = min(workers, len(all_units))
        log.info("generating %d synthetic replicas each for %s (%d total) across %d worker "
                 "process(es) → %s", n, graphs_to_generate, len(all_units), workers, work_dir)

        for var in _THREAD_VARS:
            os.environ.setdefault(var, "1")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_unit, u): u for u in all_units}
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()  # run_unit never raises; failures come back as UnitResult
                g = Path(result.out_dir).parent.name
                results.append((g, result))
                status = "ok" if result.ok else "FAILED"
                log.info("[%d/%d] %-12s replica %-4d %-6s V=%-7d E=%-8d %5.1fs%s",
                         done, len(all_units), g, result.index, status,
                         result.num_entities, result.num_edges, result.elapsed,
                         f"  ({result.error})" if not result.ok else "")
    else:
        log.info("all graphs loaded via --load-existing; nothing to generate")

    failed = [(g, r) for g, r in results if not r.ok]
    if failed:
        for g, r in failed:
            log.error("  %s unit %d failed: %s", g, r.index, r.error)

    # ── Per-graph: gather achieved.json for the units that succeeded, plus
    # whatever was loaded via --load-existing. ────────────────────────────────
    synth_feats: dict[str, list[dict]] = {g: list(preloaded_feats.get(g, [])) for g in args.graphs}
    for g, r in results:
        if r.ok:
            synth_feats[g].append(_load_signature_json(Path(r.out_dir) / "achieved.json"))
    for g in args.graphs:
        if not synth_feats[g]:
            raise SystemExit(f"All replicas of '{g}' failed; see errors above.")
        ok_n = len(synth_feats[g])
        if g in graphs_to_generate and ok_n < n:
            log.warning("%s: %d/%d replicas failed; continuing with %d.", g, n - ok_n, n, ok_n)

    # ── One PCA basis from the corpus, project every target + every replica ──
    log.info("fitting corpus PCA basis and projecting")
    corpus_signatures = _find_corpus_signatures()
    if not corpus_signatures:
        raise SystemExit(
            "No corpus signatures found under data/graphs/ or data/test_graphs/ "
            "to fit the PCA basis on."
        )
    corpus_names = sorted(corpus_signatures)
    corpus_features = [_load_signature_json(corpus_signatures[name]) for name in corpus_names]

    mat, feature_names = _build_matrix(corpus_features, size_agnostic=args.size_agnostic)
    coords, impute, mean, std, components = _fit_pca_2d(mat)

    def proj(feats):
        return _project(feats, feature_names, impute, mean, std, components)

    t_xy = {g: proj(target_feats[g]) for g in args.graphs}
    synth_xy = {g: np.array([proj(f) for f in synth_feats[g]]) for g in args.graphs}

    mode = "size-agnostic (structural only)" if args.size_agnostic else "raw (all features)"
    log.info("    corpus: %d graphs, %d features (%s)", len(corpus_names), len(feature_names), mode)
    for g in args.graphs:
        drifts = np.linalg.norm(synth_xy[g] - t_xy[g], axis=1)
        log.info("    %-12s %d/%d replicas ok; PC drift from target: mean=%.3f std=%.3f",
                 g, len(synth_feats[g]), n, drifts.mean(), drifts.std())

    # ── Plot: every target red, every synthetic blue; marker shape per graph ──
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c="#B0B0B0", s=40, alpha=0.7,
               label="corpus (real graphs)", zorder=2)
    for name, (x, y) in zip(corpus_names, coords):
        ax.annotate(name, (x, y), fontsize=8, color="#808080",
                    xytext=(4, 4), textcoords="offset points")

    for i, g in enumerate(args.graphs):
        marker = _MARKERS[i % len(_MARKERS)]
        for x, y in synth_xy[g]:
            ax.annotate("", xy=(x, y), xytext=t_xy[g],
                        arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.0, alpha=0.3),
                        zorder=3)
        ax.scatter(synth_xy[g][:, 0], synth_xy[g][:, 1], c="#4C72B0", s=90, marker=marker,
                   edgecolor="black", linewidth=1.0, alpha=0.85, zorder=4,
                   label=f"{g} — synthetic (n={len(synth_feats[g])})")
        ax.scatter(*t_xy[g], c="#C44E52", s=200, marker=marker, edgecolor="black",
                   linewidth=1.3, zorder=5, label=f"{g} — target (measured)")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    suffix = " — size-agnostic" if args.size_agnostic else ""
    graphs_label = ", ".join(args.graphs)
    ax.set_title(f"Synthetic populations in corpus PCA space: {graphs_label}{suffix}")
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    graphs_suffix = "_".join(args.graphs)
    out_path = args.out or Path(f"signature_pca_multi_{graphs_suffix}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("saved %s", out_path)
    print(f"Saved: {out_path}")

    if not args.keep_graphs:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"Generated graphs kept in: {work_dir}")


if __name__ == "__main__":
    main()
