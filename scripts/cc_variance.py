"""Collect exact-vs-CC counter benchmark data (accuracy + runtime) per motif size.

Runs the colour-coding estimator (``CCMotifCounter``) with N different seeds on
the same graph and records, per seed, the estimated 4-/5-/6-node motif counts and
induced k-star counts (k=2..10) **and the wall-clock time of each CC family call**.
The exact ground-truth count and the exact per-family runtime are computed once
via ``ExactMotifCounter`` and stored in the meta sidecar so the estimator's bias,
spread, and speed-up can be read off later.

Per-motif-size comparison covers: triangle (k=3), 4-node motifs (k=4), 5-cycle
(k=5), 6-cycle (k=6, exact via the ESCAPE enumerator), and stars k=2..10.  Two
caveats: triangle uses ``list_triangles`` in *both* counters (exact in each, so
not a real sampler race, variance 0); and stars are counted jointly (one
``count_stars`` call yields all k=2..10), so ``runtime_stars_s`` is a single value
for the whole family while accuracy stays per-k.

Stars use the same colour-coding machinery (``cc_run_stars``), which now also
averages over ``n_colorings`` colourings — so their spread tightens along the
n_colorings axis just like the motif estimators, and the all-zero collapse at
high k (single-colouring failure) is visibly mitigated by larger n_colorings.

Sweeps both ``--n-colorings`` and ``--n-samples`` (a 2-D grid) so the variance
reduction from averaging more independent colourings (Alon–Yuster–Zwick 1995;
Motivo / Bressan et al. 2021) and from drawing more path samples can be read
off later.  This script only *collects* — plot the result with
``scripts/cc_variance_viz.py``.

Output (single graph → prefix experiments/cc_variance_sweeps/<graph>_sweep;
        --graphs → experiments/cc_variance_sweeps/<sweep-options>_<timestamp>/<graph>)
------
  <out>.csv         — one row per (n_samples, n_colorings, seed); columns:
                      n_samples, n_colorings, seed, triangle_count,
                      four_cycle_count, diamond_count, k4_count,
                      tailed_triangle_count, five_cycle_count, six_cycle_count,
                      star_count_k2..k10, and per-family CC runtimes
                      runtime_triangle_s, runtime_motif4_s, runtime_motif5_s,
                      runtime_motif6_s, runtime_stars_s
  <out>_meta.json   — sweep metadata for plotting: graph name, n_runs, the swept
                      n_samples / n_colorings axes, exact ground-truth counts
                      (None where exact enumeration was infeasible), exact
                      per-family runtimes (exact_runtime) and the pre-measured
                      target signature values.

Multiple graphs can be swept in one invocation via ``--graphs``; each graph then
gets its own ``<graph>.csv`` / ``<graph>_meta.json`` inside a directory named by
the other sweep options (n_samples / n_colorings / n_runs) plus a timestamp, so
repeated runs never overwrite one another.

Usage
-----
    python scripts/cc_variance.py wn18rr_v4
    python scripts/cc_variance.py fb237_v4_ind --n-runs 100 --n-samples 10000 50000
    python scripts/cc_variance.py wn18rr_v4 --n-colorings 1 4 16 64 --n-samples 1000 10000 100000
    python scripts/cc_variance.py wn18rr_v4 --n-timings 5   # average exact runtime over 5 repeats
    python scripts/cc_variance.py wn18rr_v4 --skip-exact    # CC sweep only, no exact ground truth
    python scripts/cc_variance.py --graphs wn18rr_v4 fb237_v4_ind  # sweep several graphs
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from kgsynth.motif_counter import CCMotifCounter, ExactMotifCounter  # noqa: E402
from kgsynth.corpus import DEFAULT_SEARCH_DIRS, REPO_ROOT, load_target_from_corpus

# (feature name, sorted degree sequence) for the CC-estimated 4-node motifs.
_MOTIF4_FEATURES = [
    ("four_cycle_count",      (2, 2, 2, 2)),
    ("diamond_count",         (2, 2, 3, 3)),
    ("k4_count",              (3, 3, 3, 3)),
    ("tailed_triangle_count", (1, 2, 2, 3)),
]
# (feature name, sorted degree sequence) for the CC-estimated 5-node motifs.
_MOTIF5_FEATURES = [
    ("five_cycle_count",      (2, 2, 2, 2, 2)),
]
# (feature name, sorted degree sequence) for the CC-estimated 6-node motifs.
_MOTIF6_FEATURES = [
    ("six_cycle_count",       (2, 2, 2, 2, 2, 2)),
]
# (feature name, k) for the CC-estimated induced k-stars (k=2..10).
_STAR_FEATURES = [(f"star_count_k{k}", k) for k in range(2, 11)]
# Estimated features (triangle excluded — it is exact, variance 0).
_PLOT_FEATURES = [
    name for name, _ in
    _MOTIF4_FEATURES + _MOTIF5_FEATURES + _MOTIF6_FEATURES + _STAR_FEATURES
]
_ALL_FEATURES = ["triangle_count"] + _PLOT_FEATURES

# Per-family runtime columns recorded per row (CC) and in meta (exact).
# Stars are counted jointly (one call yields k=2..10), so star timing is a single
# value for the whole family; triangle uses list_triangles in *both* counters.
_RUNTIME_COLS = [
    "runtime_triangle_s", "runtime_motif4_s", "runtime_motif5_s",
    "runtime_motif6_s", "runtime_stars_s",
]

# Degree guard for the *exact* star ground truth. A hub of degree d that sits in
# a triangle drives ExactMotifCounter.count_stars into C(d, k) subset enumeration
# (k up to 10), which is intractable for large hubs (e.g. d=1050 → C(d,10) ≈
# 4e23). This matches the counter's internal _HUB_THRESH, the degree at which
# that branch engages. Above it the exact star truth is left None (like the c5/c6
# cycles) while CC stars are still swept; raise it to admit denser hubs.
_STAR_EXACT_MAX_DEGREE = 50


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graph", nargs="?", default=None,
                        help="Graph name in the corpus (e.g. fb237_v4_ind)")
    parser.add_argument("--graphs", nargs="+", default=None,
                        help="Sweep across multiple graphs. Each graph gets its own "
                             "<graph>.csv / <graph>_meta.json inside a directory named by "
                             "the other sweep options plus a timestamp (so repeated runs "
                             "don't overwrite each other).")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Number of CC estimator runs with different seeds (default: 50)")
    parser.add_argument("--n-samples", type=int, nargs="+", default=[10_000, 100_000],
                        help="Path-sample counts to sweep; variance is collected for each "
                             "(default: 10000 100000).")
    parser.add_argument("--n-colorings", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="Colouring counts to sweep; variance is collected for each "
                             "(default: 1 2 4 8 16 32). Each estimate averages that many "
                             "independent colourings.")
    parser.add_argument("--n-timings", type=int, default=1,
                        help="Repeat the exact counter this many times and record the mean "
                             "wall-clock (default: 1). The exact counter has no seed axis to "
                             "average over, so repeats smooth its timing noise; CC runtime is "
                             "averaged over the n_runs seeds instead.")
    parser.add_argument("--skip-exact", action="store_true",
                        help="Skip the exact ground-truth phase for all motifs. Truth counts "
                             "and exact per-family runtimes are all recorded as None; only the "
                             "CC sweep runs. Useful when exact enumeration is intractable or "
                             "only the CC variance/runtime is of interest.")
    parser.add_argument("--exact-max-degree", type=int, default=100,
                        help="Degree guard for exact ESCAPE c5/c6 enumeration (default: 100). "
                             "wn18rr_v4 has a single degree-68 hub that the library default of "
                             "50 would reject; raise this to admit such hubs (slower) or lower "
                             "it to fall back to a CC-only c5/c6 baseline.")
    parser.add_argument("--graphs-dir", type=Path, default=None,
                        help="Corpus root directory")
    parser.add_argument("--out", type=Path, default=None,
                        help="Single graph: output path prefix (no extension), default "
                             "experiments/cc_variance_sweeps/<graph>_sweep. With --graphs: "
                             "the base directory that holds the timestamped sweep directory "
                             "(default experiments/cc_variance_sweeps).")
    args = parser.parse_args()

    # Collect the graphs to run: either the multi-graph sweep or the single positional.
    graphs = args.graphs if args.graphs else ([args.graph] if args.graph else [])
    if not graphs:
        parser.error("provide a graph name positionally or via --graphs")

    search_dirs = [args.graphs_dir] if args.graphs_dir else DEFAULT_SEARCH_DIRS
    base_dir = args.out or (REPO_ROOT / "experiments" / "cc_variance_sweeps")

    if args.graphs:
        # Multi-graph sweep: one directory named by the sweep options + a timestamp,
        # holding one <graph>.csv / <graph>_meta.json per graph.
        sweep_dir = base_dir / _sweep_slug(args)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        print(f"Sweeping {len(graphs)} graph(s) into {sweep_dir}")
        for graph in graphs:
            print(f"\n=== {graph} ===")
            _run_graph(graph, sweep_dir / graph, args, search_dirs)
        print(f"\nAll graphs done → {sweep_dir}")
    else:
        # Single-graph run (backward-compatible naming).
        out_prefix = base_dir if args.out else base_dir / f"{graphs[0]}_sweep"
        _run_graph(graphs[0], out_prefix, args, search_dirs)


def _sweep_slug(args) -> str:
    """Build a directory-name slug encoding the swept axes plus a timestamp.

    The slug mirrors how single-graph runs distinguish output by graph name: here
    the directory instead encodes n_samples / n_colorings / n_runs and a run
    timestamp, so each ``--graphs`` invocation lands in its own directory.

    :param args: parsed argparse namespace.
    :returns: slug like ``ns10000-100000_nc1-2-4_runs50_20260701-142530``.
    """
    ns = "-".join(str(x) for x in sorted(set(args.n_samples)))
    nc = "-".join(str(x) for x in sorted(set(args.n_colorings)))
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"ns{ns}_nc{nc}_runs{args.n_runs}_{ts}"


def _run_graph(graph: str, out_prefix: Path, args, search_dirs) -> None:
    """Run the exact + CC sweep for one graph and write its CSV and meta sidecar.

    :param graph: graph name in the corpus.
    :param out_prefix: output path prefix (no extension); ``.csv`` and
        ``_meta.json`` are appended.
    :param args: parsed argparse namespace (sweep axes and guards).
    :param search_dirs: corpus search directories.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.json")

    print(f"Loading '{graph}' …")
    _, tblocks, graph_dir = load_target_from_corpus(graph, search_dirs)
    assert graph_dir is not None

    from kgsynth.kg_io import load_kg

    kg_files = sorted(
        p for p in graph_dir.iterdir()
        if p.suffix in {".nt", ".ttl"} and not p.stem.endswith("_synth")
    )
    if not kg_files:
        sys.exit(f"No .ttl/.nt file found in {graph_dir}")
    g = load_kg(kg_files[0])

    # Build simple undirected graph (same as BlockE.calculate() does internally)
    g_und = g.as_undirected(combine_edges="first").simplify()
    n = g_und.vcount()
    print(f"  Graph: {n:,} nodes, {g_und.ecount():,} edges")

    # Exact ground truth + per-family exact runtimes (deterministic reference).
    if args.skip_exact:
        print("  Skipping exact ground truth (--skip-exact); truth/runtime → None.")
        truth = {feat: None for feat in _ALL_FEATURES}
        exact_runtime = {key: None for key in
                         ("triangle", "motif4", "motif5", "motif6", "stars")}
    else:
        truth, exact_runtime = _exact_ground_truth(
            g_und, n_timings=args.n_timings, exact_max_degree=args.exact_max_degree)
    print("  Exact ground-truth counts:")
    for feat in _ALL_FEATURES:
        val = truth.get(feat)
        print(f"    {feat:<28} {'n/a' if val is None else f'{val:,}'}")
    print("  Exact wall-clock per family (s):")
    for key, secs in exact_runtime.items():
        print(f"    {key:<12} {'n/a' if secs is None else f'{secs:.3f}'}")

    n_colorings_list = sorted(set(args.n_colorings))
    n_samples_list = sorted(set(args.n_samples))
    print(f"Sweeping n_samples={n_samples_list} × n_colorings={n_colorings_list}; "
          f"{args.n_runs} CC seeds each …")
    rows: list[dict] = []
    _t_all = time.perf_counter()
    for ns in n_samples_list:
        for nc in n_colorings_list:
            print(f"  n_samples={ns:,}, n_colorings={nc}: running {args.n_runs} seeds …",
                  flush=True)
            _t_nc = time.perf_counter()
            for seed in range(args.n_runs):
                # Fresh CCMotifCounter per seed so each run is an independent estimate.
                cc = CCMotifCounter(n_samples=ns, seed=seed, n_colorings=nc)
                # Time each motif family separately (wall-clock per call).
                _t = time.perf_counter()
                tri = cc.count_triangles(g_und)
                rt_tri = time.perf_counter() - _t
                _t = time.perf_counter()
                motifs4 = cc.count_motifsk(g_und, 4)
                rt_m4 = time.perf_counter() - _t
                _t = time.perf_counter()
                motifs5 = cc.count_motifsk(g_und, 5)
                rt_m5 = time.perf_counter() - _t
                _t = time.perf_counter()
                motifs6 = cc.count_motifsk(g_und, 6)
                rt_m6 = time.perf_counter() - _t
                _t = time.perf_counter()
                stars = cc.count_stars(g_und)
                rt_stars = time.perf_counter() - _t
                row = {"n_samples": ns, "n_colorings": nc, "seed": seed,
                       "triangle_count": tri}
                for name, ds in _MOTIF4_FEATURES:
                    row[name] = motifs4.get(ds, 0)
                for name, ds in _MOTIF5_FEATURES:
                    row[name] = motifs5.get(ds, 0)
                for name, ds in _MOTIF6_FEATURES:
                    row[name] = motifs6.get(ds, 0)
                for name, k in _STAR_FEATURES:
                    row[name] = stars.get(k, 0)
                row["runtime_triangle_s"] = rt_tri
                row["runtime_motif4_s"] = rt_m4
                row["runtime_motif5_s"] = rt_m5
                row["runtime_motif6_s"] = rt_m6
                row["runtime_stars_s"] = rt_stars
                rows.append(row)
                if (seed + 1) % 10 == 0 or seed + 1 == args.n_runs:
                    _el = time.perf_counter() - _t_nc
                    print(f"    seed {seed + 1}/{args.n_runs}  "
                          f"({_el:.1f}s, {_el / (seed + 1) * 1e3:.0f} ms/seed)  "
                          f"c5~{row['five_cycle_count']}", flush=True)
            print(f"  n_samples={ns:,}, n_colorings={nc} done in "
                  f"{time.perf_counter() - _t_nc:.1f}s", flush=True)
    print(f"All sweeps done in {time.perf_counter() - _t_all:.1f}s", flush=True)

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["n_samples", "n_colorings", "seed"]
            + _ALL_FEATURES + _RUNTIME_COLS,
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV → {csv_path}")

    # Pull the target values from the pre-measured signature for reference.
    te = tblocks.get("e")
    target = {}
    if te is not None:
        for feat in _ALL_FEATURES:
            val = getattr(te, feat, None)
            target[feat] = None if val is None else float(val)

    meta = {
        "graph": graph,
        "n_runs": args.n_runs,
        "n_samples_list": n_samples_list,
        "n_colorings_list": n_colorings_list,
        "n_timings": args.n_timings,
        "exact_max_degree": args.exact_max_degree,
        "truth": truth,
        "exact_runtime": exact_runtime,
        "target": target,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Meta → {meta_path}")
    print(f"\nPlot with:  python scripts/cc_variance_viz.py {csv_path}")


def _exact_ground_truth(
    g_und, n_timings: int = 1, exact_max_degree: int = 100
) -> tuple[dict[str, int | None], dict[str, float | None]]:
    """Compute exact counts and per-family wall-clock times via ExactMotifCounter.

    Returns ``(truth, runtime)`` where ``truth`` maps feature name → exact count
    and ``runtime`` maps a family key (``triangle``/``motif4``/``motif5``/
    ``motif6``/``stars``) → mean seconds for that exact call over ``n_timings``
    repeats.  A cycle entry (and its runtime) is ``None`` when the exact ESCAPE
    enumeration is infeasible (high-degree hub) or unsupported, so callers can
    skip its ground-truth line.  The exact counter is deterministic, so repeats
    only smooth timing noise (the count is identical each pass).

    :param g_und: undirected simple graph.
    :param n_timings: number of repeated timing passes per family (>= 1).
    :param exact_max_degree: degree guard for the exact ESCAPE c5/c6 pass; a
        graph whose max degree exceeds it yields ``None`` for those cycles.
    """
    # Raise the ESCAPE degree guard so an isolated hub (e.g. wn18rr_v4's deg-68
    # node) doesn't suppress the exact c5/c6 baseline.
    exact = ExactMotifCounter(max_degree=exact_max_degree)
    truth: dict[str, int | None] = {}
    runtime: dict[str, float | None] = {}
    reps = max(1, n_timings)

    def _mean_time(call) -> float:
        """Mean wall-clock (s) of ``call`` over ``reps`` passes; result discarded."""
        total = 0.0
        for _ in range(reps):
            _t = time.perf_counter()
            call()
            total += time.perf_counter() - _t
        return total / reps

    print(f"  Exact ground truth: {reps} timing pass(es) per family "
          "(this phase is single-threaded and can dominate on dense/hub graphs) …",
          flush=True)

    print("    [1/4] triangle …", flush=True)
    truth["triangle_count"] = exact.count_triangles(g_und)
    runtime["triangle"] = _mean_time(lambda: exact.count_triangles(g_und))
    print(f"    [1/4] triangle done ({truth['triangle_count']:,}, "
          f"{runtime['triangle']:.3f}s)", flush=True)

    print("    [2/4] motif4 (k=4; cost O(m·Δ²)) …", flush=True)
    motifs4 = exact.count_motifs4(g_und)
    runtime["motif4"] = _mean_time(lambda: exact.count_motifs4(g_und))
    for name, ds in _MOTIF4_FEATURES:
        truth[name] = motifs4.get(ds, 0)
    print(f"    [2/4] motif4 done (c4={truth['four_cycle_count']:,}, "
          f"{runtime['motif4']:.3f}s)", flush=True)

    # 5- and 6-node cycles share the ESCAPE enumerator; the exact counter's
    # max_degree (set above) governs whether a hub graph is admitted.
    for key, features, k in (("motif5", _MOTIF5_FEATURES, 5),
                             ("motif6", _MOTIF6_FEATURES, 6)):
        print(f"    [3/4] motif{k} (k={k}, ESCAPE) …", flush=True)
        try:
            motifs = exact.count_motifsk(g_und, k)
            runtime[key] = _mean_time(lambda k=k: exact.count_motifsk(g_und, k))
            for name, ds in features:
                truth[name] = motifs.get(ds, 0)
            print(f"    [3/4] motif{k} done ({runtime[key]:.3f}s)", flush=True)
        except (RuntimeError, NotImplementedError) as exc:
            print(f"    [3/4] motif{k} unavailable ({exc}); "
                  "skipping its ground-truth line", flush=True)
            runtime[key] = None
            for name, _ in features:
                truth[name] = None

    max_deg = max(g_und.degree()) if g_und.vcount() else 0
    if max_deg > _STAR_EXACT_MAX_DEGREE:
        print(f"    [4/4] stars unavailable (max degree {max_deg} > "
              f"{_STAR_EXACT_MAX_DEGREE}; C(d,k) hub enumeration intractable); "
              "skipping exact star ground truth", flush=True)
        runtime["stars"] = None
        for name, _ in _STAR_FEATURES:
            truth[name] = None
    else:
        print("    [4/4] stars (k=2..10) …", flush=True)
        stars = exact.count_stars(g_und)
        runtime["stars"] = _mean_time(lambda: exact.count_stars(g_und))
        for name, k in _STAR_FEATURES:
            truth[name] = stars.get(k, 0)
        print(f"    [4/4] stars done ({runtime['stars']:.3f}s)", flush=True)

    return truth, runtime


if __name__ == "__main__":
    main()
