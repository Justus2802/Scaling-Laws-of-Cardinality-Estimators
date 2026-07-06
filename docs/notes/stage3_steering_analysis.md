# Stage-3 steering: delta cost, SA schedule, and why motif error is hard to move

Analysis of the Stage-3 Maslov–Sneppen + simulated-annealing rewiring loop
(`src/generator/stage3.py`), focused on **why it is slow on hub-heavy graphs**
and **why per-swap motif steering barely moves the loss** on large graphs like
`fb237_v4`. Measurements come from three tools built during this investigation:

- `scripts/profile_stage3_deltas.py` — per-swap incremental-delta cost profiler.
- `refine(swap_log=…)` + `scripts/swap_delta_viz.py` — per-proposal motif-delta /
  loss-delta logging and distribution/leverage/usefulness plots.
- a one-off loss-decomposition analysis (scratch) splitting each proposal's
  `d_loss` into per-motif signed contributions.

Reference graphs: `fb237_v4` (Stage-2 synth: 4707 nodes, 33916 content edges,
simple-degree max ≈ **1383**, mean ≈ 14) vs `wn18rr_v4` (3861 nodes, 9842 edges,
max degree 73, mean ≈ 5). The gap between "has hubs" and "no hubs" drives
almost everything below.

---

## 1. Per-swap delta cost — the 6-cycle delta dominates on hubs

Profiling the four incremental deltas on the Stage-2 graph
(`experiments/stage3_delta_profiling/summary.md`):

| delta | fb237_v4 mean/proposal | wn18rr_v4 mean/proposal |
|---|---|---|
| triangle Δ | ~0 | ~0 |
| 4-motif Δ | ~57 ms (max 1.2 s) | ~0.2 ms |
| 5-cycle Δ | ~0.13 s | ~0.2 ms |
| **6-cycle Δ** | **~2.8 s (≥94 % of total)** | ~1 ms |

- Unguarded, fb237 averages **≥ 2.9 s of delta work per proposal** — ≥ 4 h for a
  5 000-swap budget — while wn18rr averages ~1 ms (a ≳ 2700× gap).
- The cost is entirely the **induced-cycle enumeration** (`_cycle_delta`), which is
  `O(Δ^(k−2))` per changed pair. It explodes not just on hub *endpoints* but on
  high-degree *interior* vertices the path DFS branches through — so an
  endpoint-only guard barely helps on dense graphs (even swaps with all four
  endpoints < 50 average ~2 s).
- `_motif4_delta` is the #2 cost and becomes #1 once cycles are guarded (a
  1000-budget fb237 flamegraph: ~163 s of 175 s). Unlike the cycle DFS, its cost
  is fixed by the **endpoint** neighbourhoods (`N(a)∪N(b)`), so an endpoint guard
  bounds it exactly.

### Mitigations implemented
- **Meet-in-the-middle cycle enumerator** (`_induced_cycles_through_pair_mitm`,
  now the default via the `_cycles_through_pair` switch): ~2.1–3.2× faster than the
  recursive DFS, parity-tested against DFS + brute-force oracle. It is a **constant
  factor** — still `O(Δ^(k−2))`, so guards are still needed on hubs.
- **`CYCLE_DELTA_MAX_DEGREE`** — node-level guard: `_induced_paths` raises
  `_DegreeGuardExceeded` on the first node it is about to expand above the guard
  (endpoint *or* interior); `_cycle_delta` restores the adjacency and returns
  `None`. Dropped swaps carry cycle counts over unchanged (loss terms cancel).
- **`MOTIF4_DELTA_MAX_DEGREE`** — endpoint-degree guard (exact for motif4 cost).
- Both guards log computed/dropped tallies. **Both currently ship at `inf`
  (disabled)** — see §5/§6 for why guarding hubs is expensive in *fidelity*, not
  just a free speedup.

**Guard tradeoff (measured, fb237):** a low guard makes it fast but freezes
steering — at guard 20 the cycle delta is dropped on ~100 % of fb237 proposals.
The speed/fidelity tension has no free lunch; the MITM constant factor helps but
does not remove it.

---

## 2. The SA schedule was mistuned — a random walk, not annealing

The acceptance rule is `accept uphill with prob exp(−Δloss / T)`. The shipped
defaults were `initial_temp = 1.0`, `cooling_rate = 0.9999`.

- Typical per-swap `|Δloss|` is ~0.003–0.008 (wn18rr) — **~100× smaller than
  `T=1.0`**. So `exp(−Δ/T) ≈ 0.99`: the walk accepted ~99 % of *harmful* moves
  start to finish. It was a random walk with a best-seen memory, not annealing.
- `cooling_rate=0.9999` decays per **accepted** swap and needs ~46 000 accepted
  swaps to fall two decades — far more than typical budgets — so `T` barely moved.

**Retuned defaults** (both `refine` and `Generator.sample`): `initial_temp = 0.05`,
`cooling_rate = 0.99993`, tuned for a ~100k budget (temperature sweeps ~0.05 →
~0.001 over ~55k accepted swaps). Verified on wn18rr at budget 100k: the
harmful-move accept rate now falls from **0.86 (first decile) to 0.11 (last)**.

**Schedule is per-graph, not universal.** The loss scale differs by graph:
fb237's per-swap `|Δloss|` is ~**0.0004** (see §6), so `initial_temp=0.05` is
~125× too hot for it. Re-derive `initial_temp ≈ median|Δloss| / ln(1/p_accept)`
per graph (fb237 wants ~0.001–0.003). A short burn-in that measures the `|Δloss|`
distribution and sets `initial_temp` automatically would remove the hand-tuning;
not yet implemented.

---

## 3. Per-proposal swap logging (leverage on fb237)

`refine(swap_log=…)` writes one row per evaluated proposal (endpoint degrees,
per-motif deltas, `d_loss`, accepted, and the `targeted` flag). Analysed with
`scripts/swap_delta_viz.py`; the `targeted` column drives the triangle-steer
attribution in §9.

On fb237 (300-swap diagnostic run, guards off):

- **|delta| scales strongly with endpoint degree** — the biggest motif deltas are
  all at the degree-1383 end (unlike wn18rr, which is flat because it has no hubs).
  Single-swap magnitudes reach c4 58k, diamond 62k, paw 494k, c5 489k,
  **c6 14.9 million**.
- **Leverage is extremely concentrated in hubs:** the top 1 % of proposals carry
  **78–89 %** of total |delta| per motif; top 10 % carry 79–98 %. With 300
  proposals that is ~3 swaps accounting for >80 % of all c4/diamond/k4 movement.
- Therefore **guarding hubs switches motif steering essentially off** on fb237 —
  the dropped swaps are exactly the ones carrying the leverage. k4 in particular
  (84 % of proposals leave it unchanged) is steerable *only* through a handful of
  hub swaps.

---

## 4. The "approximate hub delta" idea — tested and rejected (variance)

**Idea:** rather than *drop* expensive hub deltas (the guard), compute an
*approximate* (sampled) delta so hub swaps stay in the steering signal at bounded
cost — c5/c6 first (biggest deltas, most expensive), motif4 at a higher threshold.

**Estimator tested.** A Horvitz–Thompson neighbour-subsampling estimator for the
induced-cycle delta: for each swap-endpoint hub `h` (deg > K), sample K of its
neighbours `S_h`; a cycle through the changed pairs is *observed* iff both of `h`'s
two in-cycle neighbours land in `S_h`; reweight each observed cycle by
`deg_h(deg_h−1)/(K(K−1))` per subsampled hub. `E[estimate] = true count` by
construction. Validated by Monte-Carlo over the **exact** cycle sets on 47 fb237
hub swaps (endpoint degrees 20–600, exact sets materialisable), K ∈ {8,16,32}.

**Verdict: not viable for cycles — killed by the estimator's own variance, not by
bias or by the §5 cancellation.**

- **Unbiased — confirmed.** At K=32 the count-estimator bias is ~0 (within ±8 %,
  the residual being MC noise on the bias estimate). So the H-T formula is correct.
- **But catastrophic variance on the count, growing with degree.** Relative std of
  the *count* at K=32: **14 %** (deg 20–50) → **80 %** (50–100) → **~176 %**
  (100–200) → **~228 %** (200–600). To observe a cycle you need *both* of `h`'s
  in-cycle neighbours in the sample (prob ~`(K/deg)²` ≈ 2.6 % at deg 200, K 32), so
  you see a tiny, noisy fraction and scale it up ~38×. It gets worse exactly where
  it is needed (high degree); taming it needs K ≈ deg, which defeats the point.
- **The delta is far worse** — it is a difference of two large, near-equal noisy
  counts, so its relative std explodes: at K=32, **45 %** (deg 20–50) → **~900 %**
  (50–100) → **~1470 %** (100–200) → **9 000–28 000 %** (200–600). The estimated
  delta frequently has the **wrong sign**.
- **Downstream: it corrupts the accept decision.** Feeding one estimated c5/c6 draw
  into the loss while keeping tri + motif4 **exact**, the net-loss sign flips on
  **8 % / 33 % / 25 % / 27 %** of proposals across the four degree bins (K=32), and
  the net-`Δloss` magnitude is off by a median of **3–4.5×** for deg ≥ 50. So a
  quarter-to-a-third of the highest-leverage hub decisions would be wrong.

The earlier worry was §5 cancellation flipping the sign; the real killer is one
level up — the sampled cycle count is so high-variance on hubs that the *delta*
(a small residual of two large counts) is essentially unusable. Cancellation never
gets a chance to matter. Only *mild* hubs (deg 20–50) are estimable (14 % count
std, 100 % delta sign-agreement) — but those are the cheap swaps, not the ones
worth approximating.

**Variance scaling (reusable characterisation).** `scripts/estimator_variance.py`
sweeps the sample count `K`, fits the count-estimator's relative std vs endpoint
degree as a power law per `K`, and plots it (`experiments/estimator_variance/`).
Fitted on fb237 (k=6): for K ≥ 32 the relative std scales ≈ **`deg^0.9`** with a
prefactor that roughly **halves as K doubles** — i.e. `rel_std ≈ C·deg/K`. Below
K≈32 the fit flattens (R²≈0) because the estimate is uniformly saturated-noisy at
all degrees. Extrapolating: keeping the count std under ~20 % at deg 200 needs
K ≳ 230 — i.e. sampling essentially the whole neighbourhood, which removes the
speedup entirely. This is the quantitative confirmation that "K ≈ deg" is required.
(One-off downstream loss-flip check: scratch `estimator_test.py`.)

**motif4 not separately tested,** but the case is weak regardless: motif4 counts
are denser (more 4-sets per neighbourhood → more observations → lower variance than
cycles), *and* the exact motif4 delta is endpoint-bounded and cheap, so there is
little to gain from approximating it.

**Better routes if hub steering is ever needed:** keep hub deltas **exact** (guards
off/high) and attack *cost* instead — ration hub swaps (draw only a budgeted
fraction, keep exactness), or skip + periodically re-measure the counts exactly to
stop drift. But §5 suggests the real lever is upstream (§7), not here.

---

## 5. Why motif error barely moves the loss — scale AND cancellation

Central analytical result. The loss is `Σ_k w_k·|count_k − target_k| / target_k`
(all `w_k = 1`). Each proposal's `d_loss` was decomposed into per-motif signed
contributions `contribution_k = (|cur_k + d_k − tgt_k| − |cur_k − tgt_k|)/tgt_k`,
with `cur_k` reconstructed from a measured Stage-2 baseline + cumsum of accepted
deltas. Two aggregates per proposal: `Σ contribution_k` (net motif loss move) and
`Σ|contribution_k|` (the "aligned ceiling" if all terms pulled together).

**Finding — the small `|d_loss|` (~10⁻⁴) is caused by BOTH mechanisms:**

- **Scale sets the ceiling.** Median `Σ|contribution_k|` = **3.1×10⁻⁴**. Because
  targets are in the millions (c4 1.5M, paw 5.5M, c6 165M), each term's relative
  move per swap is ~10⁻⁴ — so even a *perfectly aligned* swap moves the loss by
  only ~3×10⁻⁴, no matter what.
- **Cancellation removes a further ~74 %.** Median alignment
  `|Σcontribution| / Σ|contribution|` = **0.48** (52 % of proposals below 0.5), so
  about half of each swap's motif motion cancels against itself. Actual median
  motif `Δloss` ≈ **8×10⁻⁵** — ~4× below the aligned ceiling.
- Motif terms **dominate** `d_loss`; the non-motif residual (assortativity + CC)
  is ~10× smaller.

So the earlier "opposing terms" framing and the "targets are just huge" intuition
are **both right, and roughly co-equal**: scale caps the ceiling at ~3×10⁻⁴,
cancellation cuts it ~4× more.

**Why cancellation is structural (the key mechanism):** Stage-2 does not miss all
motifs in the same direction. Measured seed-42 baseline vs target:

| motif | Stage-2 baseline | target | direction |
|---|---|---|---|
| triangle | 14 278 | 17 114 | under |
| four_cycle (c4) | 691 849 | 1 520 129 | **under** |
| diamond | 433 193 | 413 970 | over (slight) |
| k4 | 2 521 | 14 239 | **under** (6×) |
| tailed_triangle (paw) | 11 358 296 | 5 533 706 | **over (~2×)** |
| five_cycle (c5) | 7 994 267 | 3 573 855 | **over (~2.2×)** |
| six_cycle (c6) | 186 725 278 | 165 156 712 | over (slight) |

A degree-preserving swap moves *correlated* motifs together in count-space (adding
4-cycles tends to add paws, 5-cycles, triangles…). But c4/k4 need to go **up**
while paw/c5 need to go **down** — so the same move helps one camp and hurts the
other. You literally cannot raise c4 without raising paw, which is already ~2×
too high. That opposition is baked into the target/Stage-2 mismatch, not an SA
artifact.

**Caveat:** the 300 logged swaps are the hot start of the walk near the Stage-2
configuration. The *magnitudes* are start-of-walk; the *sign structure* (which
motifs are over/under) is a slowly-changing Stage-2 property, so the cancellation
finding is representative of at least the early-to-mid run. Metrics measured on
only 300 hot swaps (e.g. the useful/harmful *acceptance* split) should not be read
as steady-state.

---

## 6. Implication: the highest-value lever is upstream (Stage 2), not Stage 3

paw and c5 come out of Stage 2 at ~2× their targets, and they are the
largest-magnitude motif contributions — all pushing the *wrong* way as structure
is added. They are the main source of the cancellation that neuters Stage-3
steering. **Reducing the Stage-2 overshoot of paw/c5 would cut the opposing
pressure at its source** — more promising than any Stage-3 delta-machinery change.
Open question: *why* does CS-first instantiation overshoot tailed-triangles and
5-cycles by ~2× on fb237? (Not yet investigated.)

---

## 7. "Generate N Stage-1/2 starts, pick the easiest to steer"

Discussed as a random-restart strategy. Assessment:

- **Economics favour it on hub graphs:** Stage 1/2 is cheap, Stage 3 is the
  expensive part (seconds/swap on fb237). Stage 1/2 preserve the marginals (A–D)
  by construction, so what varies across seeds is exactly the emergent motif /
  connectivity structure Stage 3 steers — confirmed by the budget-0 sweep
  (`experiments/sweeps/fb237_v4.jsonl`, 10 seeds).
- **The catch:** "easiest to steer" must be predicted *cheaply, before* Stage 3.
  Initial loss is nearly free but an imperfect proxy (a low-initial-loss graph can
  be a local trap). A short burn-in slope is a better predictor at modest cost.
- **Validate before building:** correlate each seed's Stage-2 initial-loss rank
  (budget-0 sweep) against its Stage-3 final-loss rank (full sweep). If they
  correlate, the free initial-loss screen is justified; if not, use the burn-in.
- Connects to §6: a start that happens *not* to overshoot paw/c5 by 2× would be
  genuinely easier to steer — so the restart payoff and the Stage-2 fix are the
  same underlying issue.

---

## 8. Open threads / recommended next steps

1. **Investigate the Stage-2 paw/c5 ~2× overshoot** (§6) — likely the single
   highest-value fidelity lever for fb237-like graphs.
2. **Per-graph `initial_temp`** — auto-calibrate from a burn-in `|Δloss|` measure
   (§2); the current 0.05 default is right for wn18rr, ~125× too hot for fb237.
3. **Full-budget fb237 run** with re-derived `initial_temp≈0.002`, guards off, and
   the convergence + swap logs, to see whether exact hub deltas actually pull motif
   errors down within budget (over the whole hot→cold sweep, not 300 hot swaps).
   *Partly done* (§9): a cooled 50k run (c5/c6 off) confirms triangles converge to
   98.5 % but best-loss barely moves (motif cancellation, §5) — the c4/k4/paw
   errors within budget are still open.
4. **Stage-2-init vs Stage-3-final rank correlation** (§7) to decide the
   restart-screen proxy.
5. Consider **loss reweighting** so the far-off overshoot motifs (paw/c5) are not
   fought by the same swaps that fix the undershoot motifs — though §5 suggests the
   move set itself (correlated motif deltas) limits how much independent control is
   possible.

---

## 9. Triangle-steer attribution — the biased `_targeted_swap` earns its keep only on hub-less graphs

`_targeted_swap` biases the proposal stream toward triangle-closing swaps
(probability ∝ `tri_deficit/target`, capped 0.5). To measure how much of the
achieved triangle steering it actually delivers, `refine()` now tallies — over
evaluated proposals — the split between targeted and random swaps for the
accepted **triangle-up steers** (accepted proposals with `tri_delta > 0`), and
`scripts/swap_delta_viz.py` reproduces it from the per-proposal `targeted`
column (`<csv>_targeted_metrics.csv` / `_targeted.png`). A *steer* is an accepted
`tri_delta>0` proposal; attribution is by both steer **count** and summed
**+Δtri gain**.

Two comparable runs — wn18rr_v4 (no hubs, budget 100k, `initial_temp=0.05`) and
fb237_v4 (hub-heavy, budget 50k, `initial_temp=0.002`, `cooling_rate=0.9998`,
c5/c6 steering off). The fb237 run is **properly cooled** (targeted accept 0.72
vs random 0.51 — real selection; an earlier `initial_temp=0.05` fb237 log was
all-hot at 0.98/1.0 and useless for this):

| metric | wn18rr (cooled) | fb237 (cooled) |
|---|---|---|
| targeted share of proposals | 10.3 % | 5.2 % |
| targeted / random accept rate | 0.81 / 0.72 | 0.72 / 0.51 |
| targeted share of steer **count** | **80.0 %** | **18.7 %** |
| targeted share of **+Δtri gain** | 80.3 % | 25.6 % |
| gain per steer (targeted / random) | — | 2.93 / 1.96 |
| final triangles / target | — | 16 861 / 17 114 (98.5 %) |

**Finding — targeting is efficient everywhere but only *necessary* on hub-less
graphs.** On fb237 it contributes only ~19 % of the steer count and ~26 % of the
triangle gain — the **opposite** of wn18rr's 80 %. But it is not failing:

- **Per-proposal it is 4.2× more efficient** — 39.6 % of targeted proposals
  become triangle-up steers vs 9.4 % of random ones. Its low *volume* (5.2 %) is
  the deficit-scaled probability throttling it as triangles converge, not a lack
  of effect. The anneal also prefers its swaps (accept 0.72 vs 0.51).
- **On fb237 triangles converge (98.5 %) via random swaps regardless** — hub
  wedges close incidentally, so the biased proposal is redundant. On sparse
  wn18rr random swaps rarely close a triangle, so it is essential.
- **Cooling collapses the per-steer gain advantage** (6.45→2.93 vs the hot log):
  the giant hub triangle-mints that dominated the hot walk are now *rejected*
  because they wreck other motifs — the §5 cancellation surfacing as the accept
  filter, exactly on the §3 high-leverage hub swaps.

**Implication.** Triangle targeting is not the fb237 lever: triangles already
reach target and best-loss barely moves (178.48→177.81), because the bottleneck
is motif cancellation (§5), not triangle creation (§6). The `_targeted_swap`
mechanism's payoff is concentrated on sparse/hub-less graphs. If targeted
*motif* proposals are generalised to c4/k4/paw (the natural next step), expect
the same regime split — useful where a motif is hard to form by chance, redundant
where hub structure forms it incidentally.
