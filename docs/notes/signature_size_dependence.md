# Size dependence of the reduced signature features

Which of the 99 reduced-signature features (`src/signature/`, blocks A/B/C/D/E/F)
**scale with graph size** and which are **size-free**. This is the distinction the
Stage-1 *conditional-on-size* model needs: extensive features must be conditioned on `V`;
intensive ones form the size-free shape. See [signature.md](../signature.md) for the
feature reference.

## Principle

- **Size-dependent (extensive)** — value scales with the number of entities `V` or edges
  `E`.
- **Size-independent (intensive)** — a shape, exponent, ratio, or a quantity bounded by
  the *vocabulary* (`R` relations, `T` classes) rather than by `V`.

## Size-dependent (extensive — scale with V or E)

| Feature | Block | Why it scales |
|---|---|---|
| `num_entities` (V) | A | *is* the size / Stage-1 conditioning variable |
| `num_relations` (R) | A | vocabulary count — unbounded, grows with size |
| `num_classes` (T) | C | type-vocabulary count |
| `num_distinct_cs` | D | count of distinct characteristic sets |
| `inv_num_distinct_cs` | D | count of distinct inverse characteristic sets |
| `num_components` | F | raw component count |
| `triangle_count`, `four_cycle_count`, `five_cycle_count`, `six_cycle_count`, `diamond_count`, `k4_count`, `tailed_triangle_count` | E | **raw motif counts** — scale strongly (≈ super-linear in E); the most size-dependent block by design (see below) |
| `two_step_vmax` | D | upper truncation cutoff of the path-count set → bounded by `freq(q*)·freq(p*)·E² ≤ E²` (see below) |
| `two_step_vmin` | D | lower truncation cutoff → hard floor of 1 (size-free), but realized value drifts with `V/R` density (see below) |

## Weakly / logarithmically size-dependent (the `x_min` family + path length)

Count thresholds and lengths that *drift upward* with size but do not scale linearly:

| Feature | Block | Behaviour |
|---|---|---|
| `out_degree_xmin`, `in_degree_xmin` | B | power-law onset — shifts up as hubs grow |
| `relation_zipf_xmin` | B | threshold over per-relation edge counts (∝ E) |
| `class_size_xmin` | C | threshold over entities-per-class (∝ V) |
| `cs_freq_vmax`, `inv_cs_freq_vmax` | D | max CS recurrence count (∝ V); `v_min` sits at the observed minimum (usually 1) |
| `shortest_path_loc` (and `lo`/`hi`) | F | grows ~`log V` (small-world); `shape` stays size-free |

## Size-independent (intensive — shape / exponent / ratio / vocab-bounded)

The remaining ~74 features:

- **`mean_degree`** (A) — `E/V`, the deliberately size-stable edge handle.
- **All exponents** — `out_degree_alpha`, `in_degree_alpha`, `relation_zipf_exponent`,
  `class_size_alpha`, `cs_freq_alpha`, `inv_cs_freq_alpha`, `two_step_alpha`, and the Block E template Zipf
  exponents (`path_template_zipf_k2..k10`, `tree_template_zipf`) — label-sequence skew, a
  shape independent of how many paths/trees exist.
- **All exp-decay rates** — `subj_cooc_rate`, `obj_cooc_rate`, `type_rel_spectrum_rate`,
  `per_type_entropy_rate`.
- **`type_rel_spectrum_scale`, `per_type_entropy_scale`** — bounded because `P(r|t)` is
  row-normalised / entropy ≤ `ln R`.
- **`subj_cooc_scale`, `obj_cooc_scale`** — V-normalised (`M/V`, implemented; see below), so
  the magnitude is the empirical joint `P(i,j)` and no longer scales with V.
- **`subj_cooc_density`, `obj_cooc_density`** — nnz fractions ∈ (0,1].
- **All skew-normal `(loc, scale, shape, lo, hi)`** for: object/subject multiplicity-α
  (distributions over *exponents*; `lo`/`hi` are the fixed [1.4, 3.0] cutoffs), row
  entropy (bounded by `ln R`), CS size and inverse-CS size (bounded by `R`).
- **`a_obj`, `a_subj`** (B) — log-log OLS slopes.
- **Block E template entropies** — `path_template_entropy_k2..k10`, `tree_template_entropy`:
  Shannon entropy of the label-sequence distribution, bounded by `≈ k·ln R` (vocabulary),
  not by `V` — it saturates as templates fill in, like the row / per-type entropies.
- **`largest_component_fraction`, `clustering_coefficient`, `degree_assortativity`,
  `shortest_path_shape`** (F) — bounded ratios / correlations.

## Features worth pinning down

### Block E motif counts are raw and scale super-linearly

The 7 motif counts (`triangle_count`, `four_cycle_count`, `five_cycle_count`,
`six_cycle_count`, `diamond_count`, `k4_count`, `tailed_triangle_count`) are stored as
**raw counts** — a deliberate decision (G5 in [signature.md](../signature.md): *"raw counts
are strongly size-dependent → more work for the Stage-1 conditional-on-size model"*). They
are the most size-dependent block: subgraph counts grow super-linearly with edges (e.g.
triangles up to `C(V,3)`, bounded in sparse graphs nearer `O(E^{3/2})`; 4-cliques faster
still), so they must be conditioned on size in Stage-1, not compared raw across graphs. The
five-/six-cycle counts are color-coding *estimates* but are estimates of the same extensive
quantities, so the size class is unchanged.

Only the **template** features of Block E are size-free: the Zipf exponents
(`*_template_zipf`) are label-sequence skew, and the entropies (`*_template_entropy`) are
vocabulary-bounded (`≈ k·ln R`) — both intensive (listed above). If a size-free motif
*shape* is ever wanted, the counts would need normalising (e.g. per-node, or against the
expected count under a degree-preserving null), mirroring the `M/V` fix for the spectrum.

### `two_step_vmin` / `two_step_vmax` are truncation cutoffs, and both scale

`fit_truncated_powerlaw` (`_fits.py`) pins `v_min = arr.min()`, `v_max = arr.max()` and
passes them as the `xmin`/`xmax` of `powerlaw.Fit`. So they are the **observed value range
(truncation bounds)** of the two-step path-count set — the same kind of object as the
skew-normal `lo`/`hi`, not a separate statistic. Measured values:

| Graph | V | R | `two_step_alpha` | `v_min` | `v_max` |
|---|---|---|---|---|---|
| aids | 254 207 | 5 | 1.015 | 1473 | 563 422 |
| codex_l | 77 951 | 69 | 1.174 | 1.0 | 6 778 374 |

`v_max` (largest path count) scales strongly with size/density. `v_min` is **not** a fixed
floor of 1: with few relations over many entities (aids: 5/254k) every `(q,p)` pair is
bridged by many nodes, so even the smallest pair count is large (1473). It tracks the
`V/R` density. Only `two_step_alpha` is the size-free shape.

**Hard bounds.** Writing `path_count(q,p) = Σ_x deg_in(x,q)·deg_out(x,p)` with `a_x =
deg_in(x,q)`, `b_x = deg_out(x,p)`, note `Σ_x a_x = |edges_q|` and `Σ_x b_x = |edges_p|`
(each `q`-edge has one head, each `p`-edge one tail). All terms non-negative ⇒ the diagonal
sum is bounded by the full product:

```
v_max ≤ max_{q,p} |edges_q|·|edges_p| = max_{q,p} freq(q)·freq(p)·E²  ≤  E²
```

a genuine size-dependent ceiling, **quadratic in `E`**, attained in the limit when one
bridge node carries all of `q`'s in-edges and all of `p`'s out-edges (a star). The
practical form is the product of the two most-used relations' edge counts; the data sits
well under it (aids `v_max=563k` vs `E²≈6·10¹¹`) only because the graphs are not
star-concentrated. For `v_min`, a pair that appears at all needs ≥1 bridge node
contributing ≥1, so

```
v_min ≥ 1   (constant — no size-dependent floor)
```

So the two cutoffs are asymmetric: `v_max` has a true `E²` upper bound, while `v_min` has
only the size-free floor of 1. `v_min`'s elevation in aids is *structural* (no low-degree
single-node bridge happens to exist), not forced by any bound — codex_l hits exactly 1. So
`v_min` has no size-dependent *bound*, but its realized value still drifts with `V/R`
density and is not a usable constant across graphs.

| Cutoff | Hard bound | Size-dependent? |
|---|---|---|
| `v_max` | `≤ freq(q*)·freq(p*)·E² ≤ E²` | yes — quadratic in E |
| `v_min` | `≥ 1` | no — constant floor (realized value still drifts) |

### How `*_cooc_scale` was made size-free (`M/V`, implemented)

**Resolved.** The co-occurrence spectrum is now V-normalised in
`signature/block_c.py` (`BlockC.calculate`): the singular values are divided by the
entity count `V` before the exp-decay fit, so `subj_cooc_scale` / `obj_cooc_scale` are
size-free. This section keeps the comparison that motivated the choice.

The co-occurrence matrix `M` is R×R with **raw entity counts**: `M[i,j]` = #entities using
both relation `i` and `j`; the diagonal `M[i,i]` = #entities using relation `i`
(`signature/block_c.py:360-373` `_build_cooc_matrix`; `_cooc_stats` runs the SVD on `M`
unnormalised). So the *raw* top singular value is O(V) — measured 414 950 for aids (V=254k)
vs 197 488 for codex_l (V=78k) — and the exp-decay `scale` (≈ that top value) inherited it.
The `rate` is decay-per-rank and was always size-free.

By contrast the `P(r|t)` spectrum runs the SVD on a **row-normalised** matrix
(`signature/block_c.py:280-287`), so its `scale` is already bounded.

**What needed fixing.** Only the `scale` is extensive. The `rate` is the
slope of `ln v_k = ln A − λk`, and a *scalar* normalisation only shifts the intercept
(`ln A`), leaving the slope — i.e. the stored `rate` — **unchanged**. So any global rescale
fixes `scale` without touching the shape we already keep; the non-scalar transforms below
change the shape itself, which is a deliberate redefinition, not just a size fix.

### Normalisation approaches compared

| Approach | Formula | Size-free | Symmetric¹ | Removes freq.² | Changes `rate`? | Main problem |
|---|---|---|---|---|---|---|
| **None (current)** | `M` | ✗ (`scale`∝V) | ✓ | ✗ | — | `scale` extensive |
| **Scalar / V** | `M / V` | ✓ | ✓ | ✗ | **no** | freq. still dominates spectrum |
| **Row-normalise** | `M[i,j]/Σ_k M[i,k]` | ✓ | ✗ | partial | yes | asymmetric + diagonal dominates |
| **Cosine / Ochiai** | `M[i,j]/√(M[i,i]·M[j,j])` | ✓ | ✓ | ✓ | yes | diagonal ≡ 1 adds a baseline |
| **Jaccard** | `M[i,j]/(M[i,i]+M[j,j]−M[i,j])` | ✓ | ✓ | ✓ | yes | non-linear; same baseline issue |
| **(P)PMI** | `max(0, log P(i,j)/(P(i)P(j)))` | ✓ | ✓ | ✓ | yes | `−∞`/negatives; sparse, heavier |

¹ `M` is symmetric, so a symmetric transform keeps a real spectrum and SVD = eigendecomposition.
² whether two always-together relations score the same regardless of how *frequent* they are.

**Scalar (`M/V`).** `M'[i,j]` = fraction of entities using both — an empirical joint
`P(i,j)`. *Advantage:* trivial, preserves symmetry, and leaves the stored `rate`
identical (only `scale` becomes a bounded probability), so it is the minimal change that
makes `scale` comparable across graphs. *Problem:* does nothing about frequency
confounding — a few high-frequency relations still own the top singular vector, so the
spectrum conflates "frequently used" with "co-occurs broadly." Choice of denominator (`V`
vs total mass vs max diagonal) only moves `scale`; dividing by the top singular value is a
bad denominator (forces `scale`≡1, discarding the magnitude).

**Row-normalise (`P(j|i)`).** *Advantage:* reuses exactly the `P(r|t)` machinery and
interpretation (conditional "given relation i, what co-occurs"), so the generator's
low-rank factorisation already consumes this form — consistency across blocks. *Problems:*
(a) breaks symmetry — the result is row-stochastic (a relation→relation transition
matrix), so the spectrum is of a non-symmetric operator and loses the correlation reading;
(b) the diagonal `M[i,i]` (self-count) is the largest entry in every row, so after
row-normalisation each row is dominated by its own self-loop and the spectrum reflects the
diagonal unless it is zeroed first; (c) decoupling is asymmetric — conditioning removes the
*source* relation's frequency but globally frequent *targets* still attract mass.

**Cosine / Ochiai.** *Advantages:* symmetric (real spectrum), bounded to [0,1], and the
*strongest* frequency decoupling — two relations on the same entity support score ≈1
regardless of frequency, so the spectrum becomes pure association structure (which
relations cluster), the most size- and scale-robust descriptor. *Problem:* the diagonal
becomes exactly 1 for every relation, so `M' = I + (off-diagonal)` carries a constant
baseline that inflates the spectrum with `R` unit eigenvalues; zero the diagonal (or
subtract `I`) before the SVD. Jaccard is the same trade-off with a union denominator;
**(P)PMI** decouples frequency too but introduces `−∞`/negative entries (use PPMI = clamp
at 0) and is the heaviest to compute.

**Decision (implemented).** `M/V` was chosen and is in place: it makes `scale` comparable
across graphs while keeping the current spectrum meaning and leaving `rate` unchanged — the
minimal fix. If a genuinely frequency-decoupled co-occurrence *shape* is later wanted,
**cosine with a zeroed diagonal** is the cleanest upgrade (symmetric, bounded,
frequency-free); **row-normalise** only if cross-block consistency with `P(r|t)` and the
existing generator path outweighs the symmetry/diagonal cost.

## Takeaways

- The reduction worked as intended: the extensive features are almost entirely the raw
  counts (`V, R, T, num_distinct_cs, inv_num_distinct_cs, num_components`, and the Block E motif counts) plus the
  path-count cutoff (`two_step_vmax`). The `x_min` thresholds and `shortest_path_loc` are a
  soft middle ground. Everything stored *as a distribution shape* is genuinely size-free —
  which is why Stage-1 can fit shape independently of `V`.
- **Resolved:** `*_cooc_scale` was the last spectrum magnitude still V-scaled; it is now
  V-normalised (`M/V`) so `scale` is size-free with `rate` unchanged. A cleaner redefinition
  (cosine with a zeroed diagonal) is recorded above if frequency-decoupling is later wanted.
