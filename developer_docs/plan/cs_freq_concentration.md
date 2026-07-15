# Plan — CS-frequency concentration and count (`cs_freq` / `num_distinct_cs`)

Status: **proposed** (diagnosis complete, fix not yet implemented).

Companion to the already-landed measurement fix (degree W1 truncation, see
`user_docs/generator.md` and the CHANGELOG). That fix removed a *reporting* artifact;
this plan addresses a real *generator* defect that the same reporting artifact
was hiding.

## Symptom

On wn18rr_v4 the roundtrip reports `D:cs_freq` W1 = 917 (8× the target IQR) — the
second-worst distance in the whole signature. The synthetic graph's characteristic
sets are both **too few** and **too flat**:

| | distinct CS | top-1 subject share | top-3 share |
|---|---|---|---|
| target | 43 | 0.48 | 0.81 |
| synth (seed 42) | 18 | 0.29 | 0.68 |

## Two things are conflated in that 917

**(a) A measurement artifact inflated it ~9× — now FIXED (2026-07-15).** `cs_freq`
is stored as a `TruncPowerLawFit`. On the flat synth counts the MLE α **pins at its
lower bound 1.0000**, and `_distance` *clipped* the unbounded Pareto at `v_max`
instead of sampling the truncated law — depositing a point mass at `v_max`, so α≈1
collapsed to a spike there (reconstructed mean = 1020 = `v_max`, i.e. "every CS
occurs ~1020×"). That is what turned an honest ~101 (raw-count W1) into the reported
917. Fixed by sampling the true truncated inverse CDF (see the CHANGELOG,
"W1 reconstruction: sample the truncated power-law, don't clip it"): `cs_freq` W1 is
now **73.6** (IQR-normalised 1.8). **The dashboard number is now trustworthy — what
remains below is the real generator defect, no longer masked.**

**(b) Under the artifact there is a real defect**, and it is **entirely in
Stage 2** — not Stage 3, and not the CS-template *size* bug guessed at earlier.
Measuring the CS distribution on the pre-refinement (Stage-2) graph directly:

```
target: distinct=43 top1=0.48
seed 0: distinct=25 top1=0.502
seed 1: distinct=20 top1=0.513
seed 2: distinct=19 top1=0.405
seed 3: distinct=21 top1=0.234   <- concentration is a coin-flip
seed 4: distinct=17 top1=0.355
seed 5: distinct=24 top1=0.318
seed 6: distinct=15 top1=0.286
seed 7: distinct=20 top1=0.285
```

The Stage-3 refined graph at seed 42 is bit-identical in CS structure to its
Stage-2 input (18 / 0.29 both), so **Stage 3 is not the culprit** — its loss has no
Block-D / CS term at all, but it also does not *degrade* CS here. The whole error
is born in Stage 2.

## Root cause (Stage 2, `stage2.py`)

Two independent Stage-2 mechanisms, both in the CS-template code path
(`_build_distinct` / `_assign_templates`):

1. **Concentration is gambled, not targeted.** `_assign_templates` builds the
   reuse distribution as `fit = sample_powerlaw_trunc(zipf, vmin, vmax, n_p, rng)`
   — `n_p` *independent* power-law draws, then normalised — and assigns the
   "extra" entities by `rng.choice(n_p, p=fit)`. The realised top-1 subject share
   is therefore `max(fit) / sum(fit)`: **the maximum order statistic of an `n_p`-
   sample of a heavy-tailed law**, divided by its (also random) sum.

   Why that is fatal for a small `n_p` and a heavy tail (α≈1.3, wn18rr):
   - Inverse-CDF sampling places the extreme values entirely in the hands of the
     few draws whose `u` is closest to 1; the realised maximum is governed by
     `min(1−u)` over `n_p` draws — a high-variance quantity. Measured over 400
     seeds at `n_p = 18`: top-1 share **mean 0.468, std 0.155, p10 0.29, p90 0.71**.
     The mean is near target (0.48) but any single seed is a coin-flip across that
     whole range — matching the 0.23–0.51 spread seen end-to-end.
   - It is literally a coin-flip whether a dominant template appears *at all*:
     `P(a single draw > 800) ≈ 0.032`, so `P(≥1 of 18 draws is dominant-sized) ≈
     0.45`. The generator *hopes* for the dominant CS instead of *placing* it.

   **This is why it "depends on how the power-law sampling works":** concentration
   is not a parameter the code sets — it is an emergent order statistic of the
   random draw, so it inherits that draw's variance. The target's 0.48 is instead a
   fixed property of **one realised frequency vector** (the specific gap 1746 vs
   620 vs 572…), not of the fitted law's shape; redrawing from the law throws that
   realised structure away and re-rolls it each seed.

   **The two defects interact — you cannot fix the count alone.** Because the same
   heavy-tailed law is spread over `n_p` templates, raising `n_p` *lowers* the top-1
   share: at the correct `n_p = 43` the measured mean top-1 share falls to **0.288**
   (std 0.085) — further from target than today's accidental 0.47 at the wrong
   `n_p = 18`. So "fix `num_distinct_cs` to 43" under the current iid-draw scheme
   would make concentration *worse*. Only a **deterministic** assignment that sets
   each template's share directly (item 3 below) decouples concentration from count
   and lets both be fixed at once.

2. **The distinct-CS count systematically undershoots (~half).** `_build_distinct`
   realised 15–25 distinct CS against a target of 43. The cause — **measured, not the
   capacity/attempt-cap story first guessed here** — is **cross-group pool overlap**.
   `num_distinct_cs` is a *union over co-occurrence groups*, but each group runs its
   own independent `_build_distinct` with a near-identical relation-probability row
   (`subj_group_probs` ≈ [0.64, 0.24, 0.04, …] for every group), so the groups all
   redraw the *same* high-probability relation-sets. On wn18rr Σ(per-group quota) = 44
   but the realised union was only 23. Every group in fact fills its own quota exactly
   (grp 0: quota 21 → pool 21) and carries the full support (`nz_g = 9`, capacity
   `C(9,·) = 381` ≫ quota) — so neither the attempt cap nor combinatorial capacity
   binds. **Status: FIXED** (2026-07-15). `_build_distinct` now dedups against a `seen`
   set **shared across all groups of a family**, so later groups are pushed off sets
   earlier groups already claimed; the union approaches Σquotas. wn18rr realised
   distinct rose to 18–37 (pool-union 23 → 40); codex_l/swdf pool-union now exact.
   |E| unchanged (this path sets relation *membership* only).

   **Confirmed coupling — this count fix alone regresses concentration.** With the
   count fix in and defect (1) still gambling, wn18rr top-1 share fell to 0.17–0.44
   (mean ≈0.31, was ≈0.36) exactly as the order-statistic analysis predicts: more
   templates spread the same heavy-tail draw thinner. The count fix and the
   deterministic-concentration fix (item 3) **must ship together** — landing either
   alone leaves `cs_freq` no better (and possibly worse) end-to-end.

## The powerlaw→quantile question

Yes — switching `cs_freq` from `TruncPowerLawFit` to a **log-share quantile fit**
(mirroring `rel_freq_logq`, which already replaced the relation-frequency Zipf for
the same reason) is the recommended core of the *generation* fix. Note the reporting
half of the original motivation is already handled: the clip→truncated-sampling fix
made the `TruncPowerLawFit` W1 honest (917 → 74), so the quantile migration is no
longer needed to fix the dashboard — its value is now almost entirely on the
generation side.

- **Measurement / reporting.** A quantile of the per-CS shares cannot collapse to a
  degenerate α, so it is *inherently* robust rather than relying on a correct
  reconstruction. A modest further tidy-up on top of the now-fixed truncated
  sampling. Use *log* shares, not raw: the head is extremely heavy (top CS = 48% of
  subjects, median CS = 5), and a linear 7-point quantile grid would smear the
  q90→q100 jump.

- **Generation — quantile-fn evaluation alone is NOT sufficient, verified.**
  The naive version of "the real payoff" above — invert the (log-)quantile at
  `n_p` evenly spaced levels, normalise — was checked numerically and **still
  drifts with `n_p`**, just less violently than the power-law family: reconstructing
  the target's own measured vector this way gives top1_share **0.68 at n_p=18,
  0.43 at n_p=43, 0.22 at n_p=100** (target 0.48). The mechanism is the same one
  that breaks the power-law family: the top value is bounded (saturates at the
  measured max), the sum grows with `n_p`, so `max/sum → 0` as `n_p` grows — no
  amount of picking a "better" family fixes a scheme where concentration is read
  off as an emergent function of how many slots you divide the pool into. A
  quantile fit removes the *seed noise* and gives an honest *shape* (which is why
  it's still worth doing, see item 1), but it does not by itself pin the *level*.

- **What does fix it: reserve the hub, mirroring `sample_degree_sequence`'s
  body/tail split.** Degrees already solved exactly this problem — a single
  family can't be trusted to land the max, so the sampler splits the population
  into a body (drawn from the fit) and an explicit tail (top ~10%, extreme-value
  matched to hit `p90`/`max` directly), then `repair_degree_sum` conserves the
  total by adjusting only the body (`adjustable` mask), never the tail. Apply the
  same architecture to CS templates:
  1. **Reserve rank 1** (or the top-`k`) and pin its subject count directly from a
     stored target statistic — a new `cs_top1_share` (mirrors `out_degree_max`/
     `out_degree_p90` being stored as explicit hub targets alongside the smooth fit).
  2. **Draw the body** (ranks 2..`n_p`) from the (log-)quantile fit of the
     *non-hub* frequencies. The family choice barely matters here — the
     rank-by-rank W1 decomposition upstream showed ranks past ~10 carry only ~8%
     of the total distributional gap.
  3. **Repair only the body** to the remaining budget (`total − hub`), never
     touching the pinned hub — the exact `repair_degree_sum` pattern.

  Verified numerically: reconstructing the target vector this way gives
  **top1_share = 0.480 (target 0.480) at every `n_p` tested, 18 through 200** —
  the count and the concentration become fully independent, which also resolves
  the coupling found above at the root (no more "fixing the count regresses
  concentration" trade-off to manage).

It (the quantile-fn migration, item 1) does **not** fix defect (2), the
distinct-count undershoot — that is a pool-construction limit, orthogonal to how
frequencies are assigned. See below. The hub-reservation design above is what
makes defects (1) and (2) independently fixable.

## Proposed changes

1. **Signature (Block D).** Replace `cs_freq_fit: TruncPowerLawFit` with
   `cs_freq_logq: QuantileFit` over the log of the per-CS subject counts (or their
   shares), **plus a new scalar `cs_top1_share`** (the hub-reservation target —
   see item 3). Do the same for `inv_cs_freq_fit`/`inv_cs_top1_share`. Update
   `distribution_fits()` to report the quantile with `_distance.QUANTILE` (the
   scalar hub share is a steering target, not a reported distance — same role as
   `out_degree_max`). Mirrors `rel_freq_logq` exactly, so the fitter
   (`fit_quantiles` on `log1p(counts)`), the NaN guard, and the serialized type
   already exist. Feature-vector length and the perturbation surface
   (`_surface.py` COUPLED group), `_domains.py`, and the from-features round-trip
   all shift — budget for the same breadth of edits the `rel_freq_logq` change
   touched.

2. **Stage 1 schema.** `cs_template_zipf` / `cs_template_vmin` / `cs_template_vmax`
   (and the `inv_` trio) become a stored quantile vector `cs_freq_logq` plus the
   `cs_top1_share` scalar. Stage 1 stops deriving a Zipf exponent for reuse and
   just forwards the quantiles + hub share.

3. **Stage 2 `_assign_templates` — hub reservation, not plain quantile-fn
   evaluation.** Numerically verified that evaluating the quantile function at
   `n_p` points and normalising (the originally-sketched approach) still drifts
   with `n_p` (0.68 at n_p=18 → 0.22 at n_p=100 against a target of 0.48) — same
   failure mode as the power-law family, just gentler. The fix that actually
   decouples concentration from `n_p`, mirroring `sample_degree_sequence`'s
   body/tail split + `repair_degree_sum`'s hub-protecting `adjustable` mask:
   - Reserve rank 1 (the single largest-pool template) and assign it
     `round(cs_top1_share · entity_budget)` directly — pinned, not drawn.
   - Draw ranks 2..`n_p` from the (log-)quantile fit of the non-hub frequencies
     via inverse-transform at `n_p − 1` evenly spaced points.
   - Repair *only the body* (largest-remainder rounding) to conserve
     `entity_budget − hub`; never touch the pinned hub.
   Keep the ≥1-entity floor on every pooled template so all are realised.
   Verified numerically (target vector, `n_p` 18–200): top1_share lands at
   **0.480 (target 0.480) at every `n_p`**, vs. 0.22–0.68 for plain quantile-fn
   evaluation. Replaces the `sample_powerlaw_trunc` + normalise +
   per-entity `rng.choice` path entirely.

4. **Distinct-count undershoot (defect 2).** *Landed (Stage 2 only).* The diagnosis
   in this sketch (attempt cap / per-group capacity / quota-vs-weight) turned out
   **not** to be the mechanism. Measured on wn18rr_v4: every group *individually*
   fills its quota exactly (grp 0 quota 21 → pool 21 in 174/420 attempts) and every
   group has the full support (`nz_g = 9`, combinatorial capacity 381 ≫ any quota),
   so neither the attempt cap (a) nor group capacity (b) binds. The real cause is
   **cross-group pool overlap**: `num_distinct_cs` is a *union-over-groups* count, but
   the `subj_group_probs`/`obj_group_probs` rows are near-identical (all ≈ [0.64,
   0.24, 0.04, …]), so each group's independent `_build_distinct` redrew the same
   high-probability sets. Σquotas = 44 but the union was only 23 → realised ~20.
   **Fix:** `_build_distinct` now dedups against a `seen` set **shared across all
   groups of a family** (one for forward, one for inverse), so later groups are forced
   past the sets earlier groups already claimed and the union approaches Σquotas.
   A starved group (attempt cap hit, every drawable set taken) is floored to one
   duplicate draw so it still contributes ≥1 template. Result (realised, Stage-2 graph):
   wn18rr_v4 distinct ~20 → ~30 (target 43; pool-union 23 → 40); codex_l/swdf pool-union
   exact. |E| unchanged (this path sets relation membership only). Capacity-proportional
   allocation (option b) would not have helped — every group has identical capacity, and
   identical prototype rows would keep the pools overlapping regardless of the split.

## Out of scope

- The CS-template *size* inflation flagged in an earlier changelog (`_build_distinct`
  drawing medians larger than target on swdf) is a distinct bug; not addressed here.
- Stage 3 gaining a Block-D loss term — not needed for this symptom (Stage 3 is
  CS-neutral on wn18rr); revisit only if a future graph shows Stage 3 *eroding* CS.

## Validation

Re-run `signature_roundtrip.py wn18rr_v4` and the 8-seed Stage-2 sweep above.
Success = top-1 share within ~±0.05 of target across seeds (variance gone), and
`D:cs_freq` W1/IQR back to O(1). Watch `swdf` and `codex_l` for regressions, and
confirm |E| and the degree distances are unchanged (this path touches only CS
frequency assignment, not edge count or degrees).
