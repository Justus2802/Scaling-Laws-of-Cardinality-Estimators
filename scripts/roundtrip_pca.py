"""Generate a synthetic population from a corpus graph and PCA-plot it.

Given the name of a corpus graph with a cached signature (``data/graphs/<name>/``
or ``data/test_graphs/<name>/``), this drives ``kgsynth.dataset`` — the same
parallel worker pool ``kgsynth dataset`` uses — to generate ``--num-graphs``
independent synthetic replicas of that target (an ``Identity`` transform: no
perturbation, only the seed differs per replica), one process per graph. It then
projects the target and every synthetic graph into a 2D PCA basis fit on the
*corpus* of real-KG signatures (the grey cloud), drawing the synthetic
population as a cloud around the target with faint arrows.

Unlike the earlier version of this script, the target is **not** measured here:
``kgsynth.dataset.worker`` refuses to measure Block E inside a worker process,
because ``load_kg``'s vertex numbering is hash-ordered and a seeded colour-coding
estimate is therefore not reproducible across processes (see
``kgsynth.dataset.config._require_cached_block_e``). Measure the graph once,
up front, with ``kgsynth measure`` (or ``scripts/measure_signature.py``) so its
``signature/block_e.json`` is cached before running this script.

Usage
-----
    kgsynth measure data/graphs/swdf/swdf.nt      # once, if not already cached
    python scripts/roundtrip_pca.py swdf
    python scripts/roundtrip_pca.py wn18rr_v4 --num-graphs 20 --workers 8
    python scripts/roundtrip_pca.py wn18rr_v4 --size-agnostic --out fig.png
    python scripts/roundtrip_pca.py wn18rr_v4 --rewire-budget 20000   # faster
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
log = logging.getLogger("roundtrip_pca")

# Each worker is single-threaded Python; letting numpy/BLAS spawn its own threads
# inside every one of them just oversubscribes the cores (mirrors
# kgsynth.dataset.runner._pin_blas_threads). Set before the pool is created so
# spawned children inherit it.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


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
    depend only on its index, never on worker count or completion order.
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
    parser.add_argument("graph", help="Corpus graph name (e.g. 'swdf'); must already have "
                        "a cached signature/block_e.json (see `kgsynth measure`).")
    parser.add_argument("-n", "--num-graphs", type=int, default=10,
                        help="Number of synthetic replicas to generate from the same "
                             "target, one per seed (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed; each replica's seeds are spawned from it "
                             "via numpy.random.SeedSequence (default: 42).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes (default: CPU count).")
    parser.add_argument("--rewire-budget", type=int, default=50_000,
                        help="Stage-3 rewiring attempts per graph (default: 50000).")
    parser.add_argument("--size-agnostic", action="store_true",
                        help="Fit PCA on scale-free structural features only (drops "
                             "size-dependent features; see plot_signature_pca.py).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output image path (default: <graph>_population_pca.png "
                             "in the current directory).")
    parser.add_argument("--keep-graphs", type=Path, default=None,
                        help="Copy the generated .ttl graphs + metadata here instead of "
                             "discarding them after the plot is made.")
    args = parser.parse_args()

    sig_dir = _require_cached_target(args.graph)
    target_feats = _load_signature_json(sig_dir / "signature.json")

    n = args.num_graphs
    work_dir = Path(args.keep_graphs) if args.keep_graphs else Path(tempfile.mkdtemp(
        prefix=f"roundtrip_pca_{args.graph}_"
    ))
    work_dir.mkdir(parents=True, exist_ok=True)
    units = _build_units(
        args.graph, n, args.seed, work_dir,
        generator_opts={"rewire_budget": args.rewire_budget},
    )

    workers = args.workers or os.cpu_count() or 1
    workers = min(workers, n)
    log.info("generating %d synthetic replicas of '%s' across %d worker process(es) → %s",
             n, args.graph, workers, work_dir)

    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_unit, u): u for u in units}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()  # run_unit never raises; failures come back as UnitResult
            results.append(result)
            status = "ok" if result.ok else "FAILED"
            log.info("[%d/%d] replica %-4d %-6s V=%-7d E=%-8d %5.1fs%s",
                     done, n, result.index, status, result.num_entities, result.num_edges,
                     result.elapsed, f"  ({result.error})" if not result.ok else "")

    failed = [r for r in results if not r.ok]
    if failed:
        for r in failed:
            log.error("  unit %d failed: %s", r.index, r.error)
        if len(failed) == len(results):
            raise SystemExit(f"All {len(results)} replicas failed; see errors above.")
        log.warning("%d/%d replicas failed; continuing with the %d that succeeded.",
                    len(failed), len(results), len(results) - len(failed))

    ok_results = sorted((r for r in results if r.ok), key=lambda r: r.index)
    synth_feats_list = [
        _load_signature_json(Path(r.out_dir) / "achieved.json") for r in ok_results
    ]

    # ── PCA basis from the corpus, project target + population ────────────────
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

    t_xy = _project(target_feats, feature_names, impute, mean, std, components)
    synth_xy = np.array([
        _project(feats, feature_names, impute, mean, std, components)
        for feats in synth_feats_list
    ])

    mode = "size-agnostic (structural only)" if args.size_agnostic else "raw (all features)"
    drifts = np.linalg.norm(synth_xy - t_xy, axis=1)
    log.info("    corpus: %d graphs, %d features (%s); %d/%d synthetic replicas ok; "
             "PC drift from target: mean=%.3f std=%.3f",
             len(corpus_names), len(feature_names), mode, len(ok_results), n,
             drifts.mean(), drifts.std())

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c="#B0B0B0", s=40, alpha=0.7,
               label="corpus (real graphs)", zorder=2)
    for name, (x, y) in zip(corpus_names, coords):
        ax.annotate(name, (x, y), fontsize=8, color="#808080",
                    xytext=(4, 4), textcoords="offset points")

    label = args.graph
    for x, y in synth_xy:
        ax.annotate("", xy=(x, y), xytext=t_xy,
                    arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.0, alpha=0.35),
                    zorder=3)
    ax.scatter(synth_xy[:, 0], synth_xy[:, 1], c="#4C72B0", s=90, marker="^",
               edgecolor="black", linewidth=1.0, alpha=0.85, zorder=4,
               label=f"{label} — synthetic (n={len(ok_results)})")
    ax.scatter(*t_xy, c="#C44E52", s=180, marker="*", edgecolor="black", linewidth=1.2,
               label=f"{label} — target (measured)", zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    suffix = " — size-agnostic" if args.size_agnostic else ""
    ax.set_title(f"Synthetic population in corpus PCA space: {label} (n={len(ok_results)}){suffix}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = args.out or Path(f"{label}_population_pca.png")
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
