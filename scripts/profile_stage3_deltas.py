"""Profile Stage-3 per-swap delta costs on the Stage-2 synthetic graph.

Rebuilds the exact Stage-2 graph a ``signature_roundtrip.py <graph>`` run feeds
into ``refine()`` (same derived seeds: master ``--seed`` → stage1 seed, +1 →
stage2, +2 → stage3 RNG), replays the Stage-3 uniform swap-proposal sampling,
and times each incremental delta helper separately per proposal:

* ``_triangle_node_delta``  — O(Δ)
* ``_motif4_delta``         — O(Δ²)
* ``_cycle_delta`` k=5      — O(Δ³)
* ``_cycle_delta`` k=6      — O(Δ⁴)

Each timed call is bounded by ``--timeout`` seconds via SIGALRM; on timeout the
adjacency entries touched by the aborted call are restored and the proposal is
recorded as censored (its true cost is ≥ the cap).  Results are written to
``experiments/stage3_delta_profiling/``:

* ``proposals_<graph>_seed<seed>.csv`` — per-proposal endpoint degrees + timings
* ``degree_stats_<graph>_seed<seed>.csv`` — Stage-2 simple-degree distribution
* ``summary.md`` — aggregated findings across all profiled graphs (regenerated
  from every proposals CSV present in the output directory)

Usage
-----
    python scripts/profile_stage3_deltas.py fb237_v4 wn18rr_v4
    python scripts/profile_stage3_deltas.py fb237_v4 --proposals 300 --timeout 5
"""

import argparse
import csv
import json
import signal
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
from kgsynth.generator.stage1 import sample_schema
from kgsynth.generator.stage2 import instantiate
from kgsynth.generator.stage3 import CYCLE_DELTA_MAX_DEGREE
from kgsynth.generator.local_updates import (
    _adj_inc, _triangle_node_delta, _motif4_delta, _cycle_delta,
)
from kgsynth.generator._constants import _RDF_TYPE
from kgsynth.signature import BlockA, BlockB, BlockC, BlockD, BlockF

_OUT_DIR = _REPO / "experiments" / "stage3_delta_profiling"
_SEARCH_DIRS = [_REPO / "data" / "graphs", _REPO / "data" / "test_graphs"]

# 4-node motif types tracked by refine() when all four targets are active.
_M4_ALL = frozenset({(2, 2, 2, 2), (1, 2, 2, 3), (2, 2, 3, 3), (3, 3, 3, 3)})


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout


def _timed(fn, adj, s1, o1, s2, o2, timeout):
    """Run ``fn()`` under a SIGALRM cap; restore swap-node adjacency on timeout.

    All four delta helpers only mutate adjacency entries between the four swap
    endpoints (via _adj_inc/_adj_dec on the four changed pairs), so snapshotting
    those 16 directed entries suffices to undo a mid-call abort.

    :returns: (elapsed_seconds, timed_out)
    """
    nodes = (s1, o1, s2, o2)
    snap = {(u, v): adj[u].get(v) for u in nodes for v in nodes if u != v}
    signal.setitimer(signal.ITIMER_REAL, timeout)
    t0 = time.perf_counter()
    try:
        fn()
        return time.perf_counter() - t0, False
    except _Timeout:
        for (u, v), cnt in snap.items():
            if cnt is None:
                adj[u].pop(v, None)
            else:
                adj[u][v] = cnt
        return time.perf_counter() - t0, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _load_blocks(graph_name: str):
    """Load cached reduced blocks A/B/C/D/F for ``graph_name`` (E not needed)."""
    for root in _SEARCH_DIRS:
        sig_dir = root / graph_name / "signature"
        if sig_dir.is_dir():
            classes = {"a": BlockA, "b": BlockB, "c": BlockC, "d": BlockD, "f": BlockF}
            return {
                letter: cls.from_serializable(
                    json.loads((sig_dir / f"block_{letter}.json").read_text())
                )
                for letter, cls in classes.items()
            }
    raise SystemExit(f"'{graph_name}' not found in {[str(d) for d in _SEARCH_DIRS]}")


def _build_stage2_graph(graph_name: str, seed: int):
    """Stage 1 + Stage 2 with the same derived seeds as Generator.sample(seed)."""
    blocks = _load_blocks(graph_name)
    schema = sample_schema(
        blocks["a"], blocks["c"], d=blocks["d"], b=blocks["b"], f=blocks["f"], seed=seed
    )
    return instantiate(schema, seed=seed + 1)


def profile_graph(graph_name: str, seed: int, n_proposals: int, timeout: float):
    """Profile ``n_proposals`` Stage-3 swap proposals on the Stage-2 graph.

    Returns (rows, degree_stats) where rows are per-proposal dicts.
    """
    print(f"[{graph_name}] building Stage-2 graph (seed={seed}) …")
    g = _build_stage2_graph(graph_name, seed)

    # Replicate refine()'s content-edge extraction and adjacency build.
    content = [
        (e.source, e.target, e["predicate"]) for e in g.es if e["predicate"] != _RDF_TYPE
    ]
    n = g.vcount()
    adj: list[dict] = [{} for _ in range(n)]
    for s, o, _ in content:
        _adj_inc(adj, s, o)

    sim_deg = np.array([len(adj[v]) for v in range(n)])
    deg_stats = {
        "graph": graph_name, "seed": seed,
        "nodes": n, "content_edges": len(content),
        "deg_mean": round(float(sim_deg.mean()), 3),
        "deg_p50": int(np.percentile(sim_deg, 50)),
        "deg_p90": int(np.percentile(sim_deg, 90)),
        "deg_p99": int(np.percentile(sim_deg, 99)),
        "deg_max": int(sim_deg.max()),
        "top10_degrees": " ".join(map(str, sorted(sim_deg, reverse=True)[:10])),
    }
    print(f"[{graph_name}] Stage-2: {n} nodes, {len(content)} content edges, "
          f"deg p90={deg_stats['deg_p90']} p99={deg_stats['deg_p99']} max={deg_stats['deg_max']}")

    # Stage-3 uniform proposal sampling (non-targeted path), same RNG stream shape.
    rel_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, p) in enumerate(content):
        rel_to_idxs[p].append(i)
    swappable = [r for r, lst in rel_to_idxs.items() if len(lst) >= 2]
    rng = np.random.default_rng(seed + 2)

    prev = signal.signal(signal.SIGALRM, _alarm_handler)
    rows = []
    try:
        while len(rows) < n_proposals:
            rel = swappable[int(rng.integers(len(swappable)))]
            pool = rel_to_idxs[rel]
            pi1, pi2 = rng.choice(len(pool), size=2, replace=False)
            s1, o1, _ = content[pool[pi1]]
            s2, o2, _ = content[pool[pi2]]
            if s1 == o2 or s2 == o1:
                continue  # refine() rejects these before any delta work

            row = {
                "graph": graph_name, "proposal": len(rows),
                "deg_s1": len(adj[s1]), "deg_o1": len(adj[o1]),
                "deg_s2": len(adj[s2]), "deg_o2": len(adj[o2]),
            }
            row["deg_max4"] = max(row["deg_s1"], row["deg_o1"], row["deg_s2"], row["deg_o2"])

            t, to = _timed(lambda: _triangle_node_delta(adj, s1, o1, s2, o2),
                           adj, s1, o1, s2, o2, timeout)
            row["t_triangle"], row["to_triangle"] = round(t, 6), int(to)
            t, to = _timed(lambda: _motif4_delta(adj, s1, o1, s2, o2, types=_M4_ALL),
                           adj, s1, o1, s2, o2, timeout)
            row["t_motif4"], row["to_motif4"] = round(t, 6), int(to)
            t, to = _timed(lambda: _cycle_delta(adj, s1, o1, s2, o2, k5=True, k6=False),
                           adj, s1, o1, s2, o2, timeout)
            row["t_c5"], row["to_c5"] = round(t, 6), int(to)
            t, to = _timed(lambda: _cycle_delta(adj, s1, o1, s2, o2, k5=False, k6=True),
                           adj, s1, o1, s2, o2, timeout)
            row["t_c6"], row["to_c6"] = round(t, 6), int(to)

            # Guarded variant: what Stage 3 actually runs — every node the path
            # DFS expands is checked against CYCLE_DELTA_MAX_DEGREE; on the first
            # hub the whole delta is dropped (returns None).
            dropped = []
            t, to = _timed(
                lambda: dropped.append(
                    _cycle_delta(adj, s1, o1, s2, o2, k5=True, k6=True,
                                 max_degree=CYCLE_DELTA_MAX_DEGREE) is None
                ),
                adj, s1, o1, s2, o2, timeout,
            )
            row["guard"] = CYCLE_DELTA_MAX_DEGREE
            row["t_c56_guarded"] = round(t, 6)
            row["guard_dropped"] = int(dropped[0]) if dropped else int(to)

            rows.append(row)
            if len(rows) % 25 == 0:
                spent = sum(r["t_triangle"] + r["t_motif4"] + r["t_c5"] + r["t_c6"] for r in rows)
                print(f"[{graph_name}] {len(rows)}/{n_proposals} proposals "
                      f"({spent:.1f}s timing so far)")
    finally:
        signal.signal(signal.SIGALRM, prev)
    return rows, deg_stats


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def _summarise(out_dir: Path) -> None:
    """Regenerate summary.md from every proposals CSV in the output directory."""
    lines = [
        "# Stage-3 per-swap delta profiling",
        "",
        "Per-proposal wall-clock cost of each incremental delta in `refine()`, measured on",
        "the Stage-2 synthetic graph with the same swap-proposal distribution Stage 3 uses.",
        "Timed-out calls are censored at the cap — their reported time is a **lower bound**,",
        "so all aggregates for columns with timeouts are lower bounds too.",
        "Generated by `scripts/profile_stage3_deltas.py`.",
        "",
    ]
    for deg_csv in sorted(out_dir.glob("degree_stats_*.csv")):
        with open(deg_csv) as fh:
            for row in csv.DictReader(fh):
                lines += [
                    f"## {row['graph']} (seed {row['seed']})",
                    "",
                    f"Stage-2 graph: {row['nodes']} nodes, {row['content_edges']} content edges; "
                    f"simple degree mean {row['deg_mean']}, p90 {row['deg_p90']}, "
                    f"p99 {row['deg_p99']}, max {row['deg_max']}.",
                    f"Top-10 degrees: {row['top10_degrees']}",
                    "",
                ]
        prop_csv = deg_csv.with_name(deg_csv.name.replace("degree_stats_", "proposals_"))
        if not prop_csv.exists():
            continue
        with open(prop_csv) as fh:
            rows = list(csv.DictReader(fh))
        n = len(rows)
        lines += [
            f"{n} swap proposals. Per-delta timing (seconds; censored at the cap):",
            "",
            "| delta | mean | median | p90 | max | total | timeouts |",
            "|---|---|---|---|---|---|---|",
        ]
        totals = {}
        for name, col, tocol in [("triangle Δ", "t_triangle", "to_triangle"),
                                 ("4-motif Δ", "t_motif4", "to_motif4"),
                                 ("5-cycle Δ", "t_c5", "to_c5"),
                                 ("6-cycle Δ", "t_c6", "to_c6")]:
            t = np.array([float(r[col]) for r in rows])
            k = sum(int(r[tocol]) for r in rows)
            totals[name] = t.sum()
            lines.append(
                f"| {name} | {t.mean():.4f} | {np.median(t):.4f} | "
                f"{np.percentile(t, 90):.4f} | {t.max():.3f} | {t.sum():.2f} | {k}/{n} |"
            )
        grand = sum(totals.values())
        share_c56 = (totals["5-cycle Δ"] + totals["6-cycle Δ"]) / grand * 100 if grand else 0
        per_attempt = grand / n if n else 0
        lines.append("")
        # Node-level guarded run (columns present when the profiler measured it):
        # the delta Stage 3 actually computes, dropped on the first hub the DFS expands.
        if rows and "t_c56_guarded" in rows[0]:
            tg = np.array([float(r["t_c56_guarded"]) for r in rows])
            drop = sum(int(r["guard_dropped"]) for r in rows)
            lines.append(
                f"- **Node-level guard** (`CYCLE_DELTA_MAX_DEGREE = {rows[0]['guard']}`, every "
                f"DFS-expanded node checked): mean {tg.mean() * 1000:.2f} ms per proposal "
                f"(max {tg.max():.3f} s), delta dropped on {drop}/{n} proposals "
                f"({drop / n * 100:.0f}%) → {tg.mean() * 5000 / 60:.1f} min of cycle work per "
                f"5 000 attempts."
            )
        lines += [
            f"- 5+6-cycle share of total delta time: **{share_c56:.1f}%**"
            + (" (lower bound — timeouts censored)" if any(
                int(r["to_c5"]) or int(r["to_c6"]) for r in rows) else ""),
            f"- Mean cost per proposal: **{per_attempt:.4f} s** → a 5 000-attempt budget "
            f"projects to ≥ **{per_attempt * 5000 / 60:.1f} min** of delta work alone.",
            "",
            "Cycle-delta cost by max endpoint degree of the proposal:",
            "",
            "| max endpoint degree | proposals | mean t_c5 (s) | mean t_c6 (s) | c5/c6 timeouts |",
            "|---|---|---|---|---|",
        ]
        bins = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 300), (300, 10**9)]
        for lo, hi in bins:
            sel = [r for r in rows if lo <= int(r["deg_max4"]) < hi]
            if not sel:
                continue
            c5 = np.mean([float(r["t_c5"]) for r in sel])
            c6 = np.mean([float(r["t_c6"]) for r in sel])
            k = sum(int(r["to_c5"]) for r in sel), sum(int(r["to_c6"]) for r in sel)
            label = f"{lo}–{hi - 1}" if hi < 10**9 else f"≥{lo}"
            lines.append(f"| {label} | {len(sel)} | {c5:.4f} | {c6:.4f} | {k[0]}/{k[1]} |")
        # Projection: expected per-attempt cycle cost under an endpoint-only guard
        # (swaps whose max endpoint degree exceeds the guard skip the cycle delta).
        # Kept for comparison with the implemented node-level guard above: it shows
        # endpoint filtering alone leaves the interior-hub cost in place.
        lines += [
            "",
            "Projected effect of an **endpoint-only** degree guard per delta family",
            "(skipped swaps carry their counts over unchanged, so their steering is frozen).",
            "For motif4 the endpoint guard is exact — its candidates come from the endpoint",
            "neighbourhoods (`MOTIF4_DELTA_MAX_DEGREE`). For cycles it is shown for comparison",
            "only: the implemented cycle guard checks every DFS-expanded node:",
            "",
            "| guard | proposals passing | mean c5+c6 of passing (s) | c5+c6 s/attempt "
            "| mean motif4 of passing (s) | motif4 s/attempt |",
            "|---|---|---|---|---|---|",
        ]
        for guard in (20, 50, 100, 200, None):
            sel = rows if guard is None else [r for r in rows if int(r["deg_max4"]) <= guard]
            frac = len(sel) / n
            mean_c56 = (sum(float(r["t_c5"]) + float(r["t_c6"]) for r in sel) / len(sel)
                        if sel else 0.0)
            mean_m4 = (sum(float(r["t_motif4"]) for r in sel) / len(sel) if sel else 0.0)
            lines.append(
                f"| {'off (inf)' if guard is None else guard} | {frac * 100:.0f}% | "
                f"{mean_c56:.3f} | {frac * mean_c56:.3f} | "
                f"{mean_m4:.4f} | {frac * mean_m4:.4f} |"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"  wrote {out_dir / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graphs", nargs="+", help="corpus graph names, e.g. fb237_v4 wn18rr_v4")
    ap.add_argument("--seed", type=int, default=42, help="master seed (as in signature_roundtrip)")
    ap.add_argument("--proposals", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="per-delta-call cap in seconds (timed-out cost is censored)")
    args = ap.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.graphs:
        rows, deg_stats = profile_graph(name, args.seed, args.proposals, args.timeout)
        _write_csv(_OUT_DIR / f"proposals_{name}_seed{args.seed}.csv", rows)
        _write_csv(_OUT_DIR / f"degree_stats_{name}_seed{args.seed}.csv", [deg_stats])
    _summarise(_OUT_DIR)


if __name__ == "__main__":
    main()
