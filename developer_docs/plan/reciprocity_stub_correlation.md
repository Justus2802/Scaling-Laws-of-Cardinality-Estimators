# Plan — reciprocal pairs are capped by Σmin(out,in), not by average stub supply

Status: **diagnosed, not designed**. A third item in the same neighbourhood as
[`cs_freq_concentration.md`](cs_freq_concentration.md) and
[`cs_edge_budget_conflict.md`](cs_edge_budget_conflict.md) — all three are cases
where a per-entity allocation step doesn't know about a downstream structural
requirement, so the requirement silently loses.

## What "reciprocity" means here

Some relations in a real KG are naturally **mirror-paired**: if `(a, similar_to, b)`
holds, `(b, similar_to, a)` usually holds too (WordNet's `similar_to`/`also_see`
are like this). Others are inherently one-directional (`part_of`: `a part_of b`
does not imply `b part_of a`). **Reciprocity `ρ_r`** is the measured fraction of
relation `r`'s edges that come in such a mirrored pair — Block B measures it per
relation from the real graph (`recip_symmetric_frac`/`recip_symmetric_value`), and
Stage 1 assigns each synthetic relation a target `ρ_r` (frequency-rank-matched to
the real relation it's meant to resemble — see
[`relation_reciprocity_and_bidirectionality.md`](../notes/relation_reciprocity_and_bidirectionality.md)
— so *which* relation is symmetric is preserved, not just how many are). Stage 2
then has to actually **build** ~`ρ_r · edges_r / 2` mirror-pairs for relation `r`
when it wires the graph — that construction step is what this doc is about, and
it is falling well short of its own target.

## Symptom — the exact shortfall

`developer_docs/problems.md` carried a scratch note: "the biggest relation needs
~3.4 stubs/entity on average from its shared pool to hit its reciprocity target but
only has ~2.85 available." Re-measured on a fresh wn18rr_v4 run (seed 42) against
the current code, the *average*-based framing turns out to understate the problem
by an order of magnitude — and, as shown below, is the wrong lens entirely.

**The exact shortfall, in one line: rel 8 (the dominant relation, ρ_r = 0.902)
targets 2727 mirror-pairs and builds exactly 1948 — a 28.6% shortfall — despite
every single one of its 2800 shared-pool entities starting with both an out-stub
and an in-stub available.** Rel 5 (same ρ_r, a smaller relation) misses by 66.5%.

Instrumented Stage 2's mutual-pair construction (`stage2.py`, the `rho_r > 0.0`
block around line 765 and the Phase-A loop around line 943) for both relations that
carry a reciprocity target on wn18rr_v4:

| relation | ρ_r | edges_r | \|shared pool\| | target pairs | **built** | shortfall |
|---|---|---|---|---|---|---|
| rel 5 | 0.902 | 510 | 839 | 230 | **77** | 66.5% |
| rel 8 (dominant) | 0.902 | 6048 | 2800 | 2727 | **1948** | 28.6% |

Rel 8 — the biggest relation, and the one the old note's numbers describe — has
**100% of its shared-pool entities with both an out-stub and an in-stub** (no zeros
at all), and its average need (3.896/entity) is within 1.4% of its average supply
(3.841/entity). By the old note's framing this relation should be nearly fine. It
realises 28.6% *fewer* mutual pairs than targeted anyway.

## Root cause — verified exactly, not estimated

A mutual pair (e1↔e2) needs **the same entity** to hold an out-stub *and* an
in-stub simultaneously (`_place(e1,e2)` then `_place(e2,e1)` — see
[stage2.py:944-972](../../src/kgsynth/generator/stage2.py#L944-L972)). So the
number of pairs any entity can ever participate in is capped at
`min(out_i, in_i)`, not at `(out_i + in_i)/2` or any other average-based quantity.
Summing that per-entity cap and halving it (each pair uses two entities) gives the
**true achievable ceiling**:

```
cap_pairs = Σ_i min(out_i, in_i)  //  2
```

Measured directly against the pre-pairing stub allocation (`m_obj`, `m_in` sliced
per relation, before Phase A runs):

- rel 5: `Σ min(out,in) = 154` → `cap_pairs = 77` — **exactly equal to the 77 built.**
- rel 8: `Σ min(out,in) = 3897` → `cap_pairs = 1948` — **exactly equal to the 1948
  built.**

Both match the actual output bit-for-bit, and re-running the pairing simulation
standalone with 5 different seeds on rel 8's real (out, in) stub arrays gives
**1948 every time** — not a distribution, a constant. **Phase A's pairing loop is
not the bottleneck; it already achieves the theoretical maximum given the stubs it
was handed.** The loss happens earlier, in how those stubs were allocated.

**Why Σmin(out,in) undershoots the naive average-based expectation:** the
out-stub count and in-stub count for a given entity are drawn as two *independent*
values — `out_val`/`in_val` in the joint IPF stub fit
([stage2.py:672-687](../../src/kgsynth/generator/stage2.py#L672-L687)) each come
from their own `sample_powerlaw_trunc` call, with no mechanism tying one entity's
out-side draw to its in-side draw. Measured directly: `corr(out, in) = 0.042` for
rel 8's shared pool — indistinguishable from zero. For independent random
variables with equal means, `E[min(X,Y)] < mean` always (strictly, unless X=Y
almost surely), and the gap widens with the variance/skew of the draws. So
whenever an entity's out-allocation and in-allocation are independent, its
*usable* reciprocal capacity is systematically smaller than either side's average
suggests — regardless of how the pairing algorithm downstream is written. This is
also why rel 5, a sparser relation, is hit far harder: independence more often
produces an outright zero on one side when the underlying draws are small
(582/839 entities have zero out-stubs, 503/839 have zero in-stubs), collapsing
`min(out,in)` to 0 for most of the pool.

## What this means for the earlier scratch note

The "3.4 needed vs 2.85 available" framing (average supply vs average demand) is
not just imprecise, it's **the wrong quantity** — reciprocal capacity is governed
by the *joint* distribution of (out, in) per entity, not by either marginal's
average. Two relations with identical `out̄`, `īn` values can have wildly
different achievable reciprocity depending on whether out and in are correlated
across entities.

## Open questions for a fix (not designed here)

- **Correlate the out/in stub draws for shared-pool entities.** Instead of two
  independent `sample_powerlaw_trunc` calls, both-pool entities could get a single
  shared multiplicity draw (or a copula-correlated pair) so an entity destined for
  a large out-allocation on a reciprocal relation tends to also get a large
  in-allocation — raising `Σ min(out,in)` without disturbing `Σ out = Σ in =
  edges_r`. Needs to preserve the existing IPF row/column margins exactly.
- **Extend the joint IPF (`generator/_ipf.py`) with a reciprocity-aware
  objective/constraint** for `both_r` entities specifically, rather than
  patching post hoc — more principled, larger change.
- **Stop chasing an unreachable `n_mutual_target`.** Compute `Σmin(out,in)/2`
  up front (cheap — the stub allocation already exists before Phase A runs) and
  either report the achievable ceiling honestly instead of the ρ_r-implied target,
  or redistribute the shortfall into the ordinary (non-reciprocal) pairing budget
  so the edge count still lands exactly, mirroring how `cs_edge_budget_conflict.md`
  handles its own over-subscribed budget.
- Whichever direction is chosen, validate on a relation with a severe shortfall
  (rel 5 here, 66.5%) as well as a mild one (rel 8, 28.6%) — a fix tuned only to
  the mild case may not generalise, same lesson as `cs_edge_budget_conflict.md`.

## Out of scope here

- The CS-frequency concentration/count fix and the CS/edge-budget conflict — both
  already filed separately; unrelated mechanisms even though they share the
  "assignment fine, wiring can't afford it" shape.
- `_reserve`'s degree-tail donor exclusion and the disabled tail-protection
  follow-up (see the CHANGELOG's 2026-07 stub-balance entries) — orthogonal;
  `_reserve` already ran in the measurements above and only marginally helped
  (rel 5's both-nonzero fraction moved 15% → 18%), consistent with this being a
  structural allocation-independence problem rather than something a donor pass
  can meaningfully patch.

## Validation

Re-run the instrumentation (or a permanent lower-verbosity version of it) on
wn18rr_v4's rel 5 and rel 8, and on at least one other corpus graph with a
reciprocity target, comparing `built` against both the ρ_r-implied target and the
`Σmin(out,in)/2` ceiling. Success = the *ceiling itself* rises (not just how much
of a fixed ceiling gets used) without breaking `Σ out = Σ in = edges_r` or
regressing degree fidelity.
