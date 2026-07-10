"""Characterise the variance of the Horvitz–Thompson neighbour-subsampling
estimator for the induced 5-/6-cycle count, as a function of endpoint node degree,
for several sample counts K — and fit a power law per K.

Motivation
----------
An "approximate hub delta" would keep hub swaps in Stage-3's steering signal at
bounded cost by subsampling each swap-endpoint hub's neighbourhood. This script
measures how noisy that estimate is: for each hub swap it computes the **exact**
set of induced cycles through the four changed pairs, then Monte-Carlo simulates
the estimator over that exact set and records the estimate's relative standard
deviation. The estimator is unbiased by construction, so relative std vs degree
(one curve per K) is the quantity that decides whether it is usable.

Estimator
---------
For each endpoint hub ``h`` with ``deg_h > K``, sample ``K`` of its neighbours
``S_h`` uniformly; an induced cycle through the changed pairs is *observed* iff
both of ``h``'s two in-cycle neighbours are in ``S_h``; each observed cycle is
reweighted by ``deg_h(deg_h-1)/(K(K-1))`` per subsampled hub. Observing a cycle
needs both neighbours sampled (prob ~ ``(K/deg)^2`` per hub), so relative std is
expected to grow roughly ``∝ deg/K``; the fitted exponent reports the real scaling.

Metric
------
* ``--metric count`` (default): relative std of the estimated cycle **count**
  through the changed pairs (the clean, fundamental quantity).
* ``--metric delta``: relative std of the estimated **delta** (after − before);
  far noisier — it is a difference of two large near-equal noisy counts.

Outputs (to ``experiments/estimator_variance/``)
------------------------------------------------
* ``<graph>_estimator_variance_<metric>_k<k>.csv`` — per (proposal, K) rows.
* ``<graph>_estimator_variance_<metric>_k<k>.png`` — relative-std vs degree
  scatter with a fitted power law per K (log-log), one figure per cycle size.
* fitted ``rel_std = a · deg^b`` parameters (a, b, R²) printed per K.

Usage
-----
    python scripts/estimator_variance.py fb237_v4
    python scripts/estimator_variance.py fb237_v4 --k 5 6 --samples 8 16 32 64 \
        --per-bin 15 --repeats 100 --metric count
    python scripts/estimator_variance.py fb237_v4 --metric delta --bins 20 50 100 200 600
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
from profile_stage3_deltas import _build_stage2_graph  # noqa: E402
from kgsynth.generator._constants import _RDF_TYPE  # noqa: E402
from kgsynth.generator.local_updates import (  # noqa: E402
    _induced_cycles_through_pair_mitm as _cyc, _adj_inc, _adj_dec,
)

_OUT_DIR = _REPO / "experiments" / "estimator_variance"
# Fixed categorical order (Paul Tol "bright", colourblind-safe) — one hue per K.
_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]


def _build(graph: str, seed: int):
    """Stage-2 content-edge adjacency for ``graph`` (same derived seeds as refine)."""
    g = _build_stage2_graph(graph, seed)
    n = g.vcount()
    content = [(e.source, e.target, e["predicate"])
               for e in g.es if e["predicate"] != _RDF_TYPE]
    adj = [dict() for _ in range(n)]
    for s, o, _ in content:
        _adj_inc(adj, s, o)
    from collections import defaultdict
    rel2idx = defaultdict(list)
    for i, (_, _, p) in enumerate(content):
        rel2idx[p].append(i)
    swappable = [r for r, l in rel2idx.items() if len(l) >= 2]
    return adj, content, rel2idx, swappable


def _union_cycles(adj, pairs, k):
    """Union of induced k-cycles through the given changed pairs (current adj)."""
    s = set()
    for x, y in pairs:
        s |= _cyc(adj, x, y, k)
    return s


def _ht_draw(adj, cyc_set, hubdeg, K, rng):
    """One Monte-Carlo draw of the H-T estimate of ``|cyc_set|``, subsampling each
    hub in ``hubdeg={h: deg}`` to ``K`` neighbours (uses ``adj`` to find each
    cycle's in-cycle neighbours of ``h``)."""
    if not hubdeg:
        return float(len(cyc_set))
    S = {h: set(rng.sample(list(adj[h]), K)) for h in hubdeg}
    wf = {h: d * (d - 1) / (K * (K - 1)) for h, d in hubdeg.items()}
    est = 0.0
    for V in cyc_set:
        w = 1.0
        ok = True
        for h in hubdeg:
            if h in V:
                if not all((x in S[h]) for x in V if x in adj[h]):
                    ok = False
                    break
                w *= wf[h]
        if ok:
            est += w
    return est


def _sample_proposals(adj, content, rel2idx, swappable, bins, per_bin, rng, max_tries):
    """Draw same-relation swap proposals bucketed by max endpoint degree."""
    edges = [(lo, hi) for lo, hi in zip(bins[:-1], bins[1:])]
    picked = {b: [] for b in edges}
    tries = 0
    while any(len(v) < per_bin for v in picked.values()) and tries < max_tries:
        tries += 1
        pool = rel2idx[swappable[rng.randrange(len(swappable))]]
        i1, i2 = rng.sample(pool, 2)
        s1, o1, _ = content[i1]
        s2, o2, _ = content[i2]
        if s1 == o2 or s2 == o1:
            continue
        dm = max(len(adj[s1]), len(adj[o1]), len(adj[s2]), len(adj[o2]))
        for lo, hi in edges:
            if lo <= dm < hi and len(picked[(lo, hi)]) < per_bin:
                picked[(lo, hi)].append((s1, o1, s2, o2))
                break
    return picked


def _rel_std_count(adj, s1, o1, s2, o2, k, Ks, R, rng, max_set):
    """Per-K relative std of the estimated cycle COUNT through the changed pairs.

    Returns (deg_max4, {K: (exact_count, rel_std)}) or None if the exact set is
    too large to materialise."""
    pairs = ((s1, o1), (s2, o2), (s1, o2), (s2, o1))
    cyc_set = _union_cycles(adj, pairs, k)
    if len(cyc_set) > max_set:
        return None
    exact = len(cyc_set)
    dm = max(len(adj[s1]), len(adj[o1]), len(adj[s2]), len(adj[o2]))
    out = {}
    for K in Ks:
        hubdeg = {v: len(adj[v]) for v in (s1, o1, s2, o2) if len(adj[v]) > K}
        if not hubdeg or exact == 0:
            out[K] = (exact, 0.0)
            continue
        est = np.array([_ht_draw(adj, cyc_set, hubdeg, K, rng) for _ in range(R)])
        out[K] = (exact, float(est.std() / exact) if exact else float("nan"))
    return dm, out


def _rel_std_delta(adj, s1, o1, s2, o2, k, Ks, R, rng, max_set):
    """Per-K relative std of the estimated cycle DELTA (after − before)."""
    pairs = ((s1, o1), (s2, o2), (s1, o2), (s2, o1))
    before = _union_cycles(adj, pairs, k)
    if len(before) > max_set:
        return None
    _adj_dec(adj, s1, o1); _adj_dec(adj, s2, o2)
    _adj_inc(adj, s1, o2); _adj_inc(adj, s2, o1)
    after = _union_cycles(adj, pairs, k)
    too_big = len(after) > max_set
    dm = max(len(adj[s1]), len(adj[o1]), len(adj[s2]), len(adj[o2]))
    res = None
    if not too_big:
        exact = len(after) - len(before)
        out = {}
        for K in Ks:
            hb_b = {v: len(adj[v]) for v in (s1, o1, s2, o2) if len(adj[v]) > K}
            # after-graph adj is current (swap applied); recompute hub degrees on it
            # (degrees are swap-invariant, but membership sets differ)
            eb = np.array([_ht_draw_before(adj, before, hb_b, K, rng, s1, o1, s2, o2)
                           for _ in range(R)])
            ea = np.array([_ht_draw(adj, after, hb_b, K, rng) for _ in range(R)])
            ed = ea - eb
            denom = abs(exact) if exact else float("nan")
            out[K] = (exact, float(ed.std() / denom) if exact else float("nan"))
        res = (dm, out)
    _adj_dec(adj, s1, o2); _adj_dec(adj, s2, o1)
    _adj_inc(adj, s1, o1); _adj_inc(adj, s2, o2)
    return res


def _ht_draw_before(adj_after, before_set, hubdeg, K, rng, s1, o1, s2, o2):
    """H-T draw for the BEFORE set while ``adj_after`` currently holds the swapped
    graph: temporarily revert the swap so in-cycle neighbours are looked up on the
    before-graph, then re-apply."""
    _adj_dec(adj_after, s1, o2); _adj_dec(adj_after, s2, o1)
    _adj_inc(adj_after, s1, o1); _adj_inc(adj_after, s2, o2)
    val = _ht_draw(adj_after, before_set, hubdeg, K, rng)
    _adj_dec(adj_after, s1, o1); _adj_dec(adj_after, s2, o2)
    _adj_inc(adj_after, s1, o2); _adj_inc(adj_after, s2, o1)
    return val


def _power_fit(deg, rel_std):
    """Fit rel_std = a · deg^b in log-log space. Returns (a, b, r2) or None."""
    deg = np.asarray(deg, float)
    rel_std = np.asarray(rel_std, float)
    m = (deg > 0) & (rel_std > 0)
    if m.sum() < 3:
        return None
    lx, ly = np.log(deg[m]), np.log(rel_std[m])
    b, la = np.polyfit(lx, ly, 1)
    pred = la + b * lx
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(la)), float(b), float(r2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", help="corpus graph name (e.g. fb237_v4)")
    ap.add_argument("--seed", type=int, default=42, help="master seed (as in roundtrip)")
    ap.add_argument("--k", type=int, nargs="+", default=[6], help="cycle sizes (5 and/or 6)")
    ap.add_argument("--samples", type=int, nargs="+", default=[8, 16, 32, 64],
                    metavar="K", help="neighbour sample counts to sweep")
    ap.add_argument("--bins", type=int, nargs="+", default=[20, 50, 100, 200, 600],
                    help="degree-bin edges for proposal selection")
    ap.add_argument("--per-bin", type=int, default=15, help="proposals per degree bin")
    ap.add_argument("--repeats", type=int, default=100, help="MC repeats per (proposal, K)")
    ap.add_argument("--metric", choices=["count", "delta"], default="count")
    ap.add_argument("--max-set", type=int, default=400_000,
                    help="skip a proposal if any exact cycle set exceeds this")
    ap.add_argument("--max-tries", type=int, default=400_000,
                    help="cap on random draws while filling degree bins")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"Building {args.graph} Stage-2 graph (seed {args.seed}) …")
    adj, content, rel2idx, swappable = _build(args.graph, args.seed)

    picked = _sample_proposals(adj, content, rel2idx, swappable,
                               args.bins, args.per_bin, rng, args.max_tries)
    print("proposals per bin:",
          {f"{lo}-{hi}": len(v) for (lo, hi), v in picked.items()})

    measure = _rel_std_count if args.metric == "count" else _rel_std_delta
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    for k in args.k:
        rows = []
        for props in picked.values():
            for (s1, o1, s2, o2) in props:
                r = measure(adj, s1, o1, s2, o2, k, args.samples, args.repeats,
                            rng, args.max_set)
                if r is None:
                    continue
                dm, per_k = r
                for K, (exact, rstd) in per_k.items():
                    rows.append({"deg_max4": dm, "K": K, "exact": exact,
                                 "rel_std": rstd})
        if not rows:
            print(f"k={k}: no usable proposals (all exact sets exceeded --max-set)")
            continue

        # CSV
        csv_path = _OUT_DIR / f"{args.graph}_estimator_variance_{args.metric}_k{k}.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["deg_max4", "K", "exact", "rel_std"])
            w.writeheader()
            w.writerows(rows)

        # Fit + plot: rel_std vs degree, one power-law curve per K.
        fig, ax = plt.subplots(figsize=(7, 5))
        print(f"\nk={k} ({args.metric}) — power-law fit  rel_std = a · deg^b  per K")
        print(f"  {'K':>4}  {'n':>3}  {'a':>10}  {'b':>7}  {'R²':>6}")
        for i, K in enumerate(args.samples):
            sub = [r for r in rows if r["K"] == K and r["rel_std"] > 0]
            if not sub:
                continue
            deg = np.array([r["deg_max4"] for r in sub], float)
            rstd = np.array([r["rel_std"] for r in sub], float) * 100  # percent
            color = _COLORS[i % len(_COLORS)]
            ax.scatter(deg, rstd, s=24, alpha=0.55, color=color,
                       edgecolors="none", label=f"K={K}")
            fit = _power_fit(deg, rstd)
            if fit:
                a, b, r2 = fit
                xs = np.linspace(deg.min(), deg.max(), 100)
                ax.plot(xs, a * xs ** b, color=color, lw=2)
                print(f"  {K:>4}  {len(sub):>3}  {a:>10.3g}  {b:>+7.2f}  {r2:>6.2f}")
            else:
                print(f"  {K:>4}  {len(sub):>3}  {'(too few pts)':>10}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("max endpoint degree")
        ax.set_ylabel(f"relative std of estimated {args.metric} (%)")
        ax.set_title(f"HT cycle-{args.metric} estimator variance vs degree "
                     f"— {args.graph}, k={k}", fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(title="samples", fontsize=8)
        fig.tight_layout()
        png_path = csv_path.with_suffix(".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"  → {csv_path}\n  → {png_path}")


if __name__ == "__main__":
    main()
