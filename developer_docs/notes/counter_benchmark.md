# Motif counters: exact vs CC, adaptive sampling, and the benchmark

Covers (1) the exact-vs-colour-coding (CC) counter comparison per motif size and
(2) the optional adaptive sample-size feature on the CC counter.

## Counters

`src/kgsynth/motif_counter/` provides three `MotifCounter` implementations:

| Counter | Triangle | k=4 | k=5 | k=6 | stars k=2..10 |
|---|---|---|---|---|---|
| `ExactMotifCounter` | exact | exact | ESCAPE | ESCAPE | exact |
| `CCMotifCounter` | exact | CC | CC | CC | CC |
| `HybridMotifCounter` | exact | CC | CC | CC | CC |

- Exact k=5/6 use the ESCAPE BFS enumerator (`count_motifsk_escape`), which
  raises `RuntimeError` when the max degree exceeds the guard (`max_degree`
  constructor arg, default 50). It is validated against an independent
  brute-force induced-cycle oracle in `tests/test_generator_motif_counter.py`
  (`TestEscapeExactCyclesVsBrute`).
- The CC sampler functions (`cc_run`, `cc_run_stars`, `cc_run_stars_loop`) live in
  `cc_motif_counter.py` next to `CCMotifCounter`.

### Fuzz-test coverage

Brute-force oracles shared by the counter tests live in
[`tests/_brute_motifs.py`](../../tests/_brute_motifs.py) (`brute_tri_counts`,
`brute_induced_cycles`, `brute_motifsk` — connected induced graphlets by sorted
degree sequence — and `brute_stars`). They enumerate subgraphs directly, so they
are independent ground truth. `brute_motifsk`/`brute_stars` are verified to match
`ExactMotifCounter` exactly.

[`tests/test_hybrid_motif_counter.py`](../../tests/test_hybrid_motif_counter.py)
fuzzes `HybridMotifCounter` across **every** family it reports:

- **Exact paths** (triangles, k=2 edges, k=3 wedge/triangle) — asserted for exact
  equality against the oracle over 200 random graphs.
- **CC paths** (k=4 graphlets, C5, C6, stars) — the estimate is averaged over
  independent hybrid seeds and compared to the oracle only where the true count is
  abundant, with a relative tolerance (the same statistical strategy the CC star
  tests in `test_generator_motif_counter.py` use).

## Adaptive sampling (CC counter)

`CCMotifCounter(n_samples=…, n_colorings=…, adaptive=False)`. The `adaptive`
boolean changes how `n_samples` is interpreted, resolved per call in
`_resolve_samples(g)`:

- **`adaptive=False` (default):** every call uses exactly `n_samples` path /
  centre samples, independent of graph size.
- **`adaptive=True`:** `n_samples` becomes the *base budget*; the effective count
  scales with the node count `n`:
  `effective = max(500, min(n·20, n_samples·5))`.

  | regime | effective samples | with `n_samples=5000` |
  |---|---|---|
  | floor | `500` | `n ≤ 25` |
  | linear | `n·20` | `25 < n < 1250` |
  | cap | `n_samples·5` | `n ≥ 1250` → 25 000 |

**Gotcha:** under `adaptive=True` the effective count is *not* `n_samples` — it
can be up to `5·n_samples`. So `CCMotifCounter(n_samples=5000, adaptive=True)` on
a large graph draws 25 000 samples. To cap *at* a budget, pass a fifth of it.

The resolved count drives **both** the graphlet sampler (`cc_run`) and the star
centre samples (`cc_run_stars`). Triangles are exact and never sampled. The same
`adaptive` flag is forwarded by `HybridMotifCounter` to its inner CC sampler. The
defaults are `adaptive=False`, so behaviour is unchanged unless opted in. (This
formula was previously dead code in Block E; it is now an opt-in counter feature.)

## Benchmark

Collected by [`scripts/cc_variance.py`](../../scripts/cc_variance.py) and plotted
by [`scripts/cc_variance_viz.py`](../../scripts/cc_variance_viz.py); outputs in
`experiments/cc_variance_sweeps/`. Two axes: **runtime** (exact vs CC per family)
and **accuracy** (CC spread vs exact ground truth) over an `n_samples ×
n_colorings × seed` grid. Covers triangle (k=3), 4-node motifs (k=4), 5-cycle,
6-cycle, and stars k=2..10.

Two figures: `<csv>.png` (accuracy boxplots + CV table) and `<csv>_runtime.png`
(mean CC runtime vs `n_samples` per family, exact as a reference line).

### Findings on `wn18rr_v4`

(3 861 nodes, 6 785 edges, mean degree 3.5, one degree-68 hub.) Exact per-family
wall-clock: triangle ~0.0003 s, 4-node motifs 0.14 s, 5-cycle 9.0 s,
**6-cycle 150 s**, stars 0.55 s.

- **Exact wins** on triangle (identical in both), 4-node motifs, and stars —
  faster than CC across the whole budget grid, and exact.
- **CC wins** on the cycles: exact 5-cycle (9 s) and especially 6-cycle (150 s)
  are far slower than CC, which estimates them in seconds.
- Accuracy tightens with both `n_samples` and `n_colorings`; `k4_count`, large
  stars (k≥8), and `six_cycle` need more colourings to stabilise.

Future work (see memory `project-cc-vs-exact-graphsize`): find the graph size at
which CC overtakes exact on stars / 4-node motifs.

### Caveats

- **Triangle** uses `list_triangles` in both counters — not a real sampler race
  (variance 0).
- **Stars** are counted jointly (one call yields k=2..10), so `runtime_stars_s`
  is one value per family while accuracy stays per-k.
- **Exact c5/c6 degree guard** — `--exact-max-degree` (default 100) raises the
  ESCAPE guard so `wn18rr_v4`'s lone degree-68 hub doesn't suppress the exact
  cycle baseline. Denser graphs with real hubs fall back to a CC-only baseline.
- **Skip exact entirely** — `--skip-exact` bypasses the exact ground-truth phase
  for all motifs; truth counts and exact per-family runtimes are recorded as
  `None`, leaving only the CC sweep. Use it when exact enumeration is intractable
  or only the CC variance/runtime matters.

### Known CC accuracy limits (surfaced by the hybrid fuzz test)

- **Diamond over-count (k=4).** The CC estimator systematically over-counts the
  diamond graphlet — degree sequence `(2,2,3,3)`, K4 minus an edge. On a verified
  fixture (true = 84) it estimates ~135 even at 500 k samples × 32 colourings, so
  the bias does **not** vanish with budget; P4, paw, C4, and K4 all converge
  correctly. The older `TestExactVsCC` never caught this because its Petersen
  fixture is diamond-free. `test_fuzz_motifs4` therefore excludes the diamond from
  its tight assertion and only bounds it loosely. **TODO: fix the CC diamond
  estimator (likely a σ / spanning-path normalisation issue for `(2,2,3,3)`).**
- **6-cycle variance.** A 6-motif is colourful in only ~1.5 % of colourings, so C6
  estimates are high-variance and can drift ~40–50 % even averaged over seeds;
  `test_fuzz_cycles_c5_c6` asserts C5 tightly but bounds C6 only loosely.

### Reproduce

```
python scripts/cc_variance.py wn18rr_v4 \
    --n-runs 10 --n-samples 1000 10000 100000 --n-colorings 1 4 16 --n-timings 3
python scripts/cc_variance_viz.py experiments/cc_variance_sweeps/wn18rr_v4_sweep.csv
```

To sweep several graphs at once, pass `--graphs`. Each graph gets its own
`<graph>.csv` / `<graph>_meta.json` inside a directory named by the other sweep
options plus a timestamp, so repeated runs never overwrite one another:

```
python scripts/cc_variance.py --graphs wn18rr_v4 fb237_v4_ind \
    --n-runs 10 --n-samples 1000 10000 --n-colorings 1 4 16
# → experiments/cc_variance_sweeps/ns1000-10000_nc1-4-16_runs10_<timestamp>/{wn18rr_v4,fb237_v4_ind}.csv
```
