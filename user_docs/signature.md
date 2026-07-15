# The Reduced, Non-Over-Determined Signature

The `signature` package (Blocks A, B, C, D, E, F). This document is the design **and the
reasoning** behind that signature. For the concrete module/feature reference jump to
[Implemented module](#implemented-module); for the per-block measurement assumptions see
[notes/assumptions.md](../developer_docs/notes/assumptions.md). Empirical
basis: [notes/signature_observations.md](../developer_docs/notes/signature_observations.md) (referred to
below as "the notes"). For which features scale with graph size vs. which are size-free
(the Stage-1 conditioning split), see
[notes/signature_size_dependence.md](../developer_docs/notes/signature_size_dependence.md). For where this
implementation intentionally departs from the proposal, see
[Deviations from the proposal](#deviations-from-the-proposal).

## Implemented module

`src/kgsynth/signature/` is the signature package (Blocks A–F). Each block is a single
non-over-determined ("reduced") measurement — one class in `block_<x>.py`. The package reuses a
shared block infrastructure (the
`SignatureBlock` ABC with `as_dict` / `to_serializable`, JSON
serialization, logging, and the `powerlaw` fitter). Library-backed distribution fits live
in `_fits.py` (`np.quantile` for the non-parametric quantile-function fits,
`scipy.stats.linregress`, the `powerlaw` package);
`_plot_helpers.py` overlays each fit on the raw data it was computed from — every block
keeps that pre-fit data (singular values, row entropies, per-relation exponents,
per-relation edge counts, class sizes, path counts, path lengths, CS-frequency counts)
on the object for `visualize`. The overlay helpers cover every fitted distribution:
`overlay_quantiles`, `overlay_exp_decay_rank`, `overlay_truncated_powerlaw`,
`overlay_powerlaw` (open-tailed `PowerLawStats`) and `overlay_zipf` (`ZipfFit`). Each
block's plot shows all of its distribution fits:
- **B** (2×3): out/in-degree power-laws, relation-usage **log-share quantile function**, obj/subj multiplicity-α quantiles.
- **C** (3×3): three co-occurrence/`P(r|t)` spectra (exp-decay), subj/obj row-entropy quantiles, per-type entropy, class-size **power-law**.
- **D** (2×3): forward/inverse CS **size** (quantiles), two-step path counts (trunc. power-law), forward/inverse CS **frequency** (trunc. power-law).
- **E** (2×2): motif counts, path-template stats by k, tree-template scalars (the star-count panel is disabled).

| Block | Vec | Stored representation (rationale in the sections below) |
|---|---|---|
| **A** — G0 | 4 | `num_entities`, `num_relations`, **mean degree** `E/V`, `type_edge_frac` (rdf:type share of E — splits the budget into type vs content edges, so `content_E = E·(1−type_edge_frac)` is the mean Block B's degree fits actually describe) |
| **B** — G1/G2/G2b | 42 | out/in-degree power-law of the **entity content degrees** (rdf:type edges and the class nodes they point at are excluded — a class node is not an entity, and counting it put a class's instance count in the in-degree tail: aids measured an in-degree max of 184493, i.e. \|class0\|, which the generator then imposed on entities); relation-usage **log-share quantile function** (7 levels, over `log(E_r/ΣE_r)` — supersedes the originally-proposed Zipf exponent, see below); obj/subj multiplicity-α **quantile function** (7 levels, cutoffs [1.4,3.0]) + the two laws' upper bounds `obj_mult_max`, `subj_mult_max`; CS-size offsets `a_obj`, `a_subj`; high-end degree targets `out/in_degree_max`, `out/in_degree_p90` (explicit hub-steering targets for Stage 2); `subject_frac`/`object_frac` (share of entities with nonzero out-/in-degree); per-relation **reciprocity** (`recip_symmetric_frac_bin0..5` over 6 frequency bins + `recip_symmetric_value`) |
| **C** — G3 | 29 | class-size **power-law**; subj/obj co-occurrence **exp-decay** + density; `edge_multiplicity` + `bidirectional_ratio` (pair-overlap / two-way scalars); row entropy **quantile function** (7 levels); `P(r\|t)` spectrum **exp-decay**; per-type entropy **exp-decay** |
| **D** — G3 | 25 | `num_distinct_cs`; CS-freq **truncated power-law** (α, v_min, v_max); CS-size **quantile function** (7 levels); symmetric inverse side (`inv_num_distinct_cs`, inverse-CS-freq **truncated power-law**, inverse-CS-size **quantile function**); two-step path-count **truncated power-law** |
| **E** — G5 | 27 | raw motif counts (triangle, 4-/5-/6-cycle, diamond, k4, tailed triangle); path-template **Zipf** + entropy (k=2..10); tree-template Zipf + entropy. Induced star counts are not measured or vectorised — the characteristic-set distribution (Block D) pins per-subject fan-out; the `count_stars` helper is retained (unused) for tests / `scripts/cc_variance.py` |
| **F** — G4 | 7 | components, LCC fraction, avg-local clustering, assortativity; shortest-path **max/mean/var** summary (`shortest_path_max` = diameter, `shortest_path_mean`, `shortest_path_var`) |

Total **134** features (A4 + B42 + C29 + D25 + E27 + F7).
The fits are stored as NamedTuples that restore as plain tuples through the JSON
round-trip, so each block property re-wraps them to preserve attribute access.

Run it: `python scripts/measure_signature.py <graph> [--blocks a,b,c,d,e,f]` (or the installed
`kgsynth measure` CLI) → `<graph-dir>/signature/` (override with `--output-dir`); the curated
corpus lives in `data/graphs/<name>/signature/`. Or `scripts/measure_all_raw.py` for every raw KG
in `data/graphs/` and the test corpus `data/test_graphs/`. The on-disk layout (per-block
`block_<x>.png`/`.json`, `summary.txt`, combined `signature.json`) is produced by the shared
`signature.write_signature_outputs` helper, which `scripts/signature_roundtrip.py` also uses to
dump re-measured **generated** graphs to a parallel `signature_synth/` directory. Tests:
`tests/test_signature_reduced_fits.py`, `tests/test_signature_reduced_blocks.py`.

The rest of this document is the **reasoning**: why the signature is reduced this way (the
derivability criterion), how per-relation multiplicity and degree relate (the
investigation), and the per-group justification (G0–G6).

## Goal

A **complete** signature — enough to instantiate a KG in Stage 2 — that is also
**non-over-determined**: no entry is an exact function of the others. Two kinds of
over-determination are avoided:

1. **Algebraic redundancy** — values that are exact functions of others
   (`density = E/V²`, `relation_reuse = E/R`, …).
2. **Cross-statistic redundancy** — a feature that is a guaranteed function of another
   stored one (e.g. `functionality` is the head `P(count=1)` of the per-relation
   multiplicity law). *Note:* aggregate degree ↔ multiplicity and co-occurrence spectrum
   ↔ CS distribution are **not** redundant — the marginals/lossy spectrum don't pin them
   (see the derivability criterion) — so those are **kept**, not dropped.

Raw `mean/std/median` triples are likewise avoided: each quantity stores the
**parameters of its distribution family** (see the conventions table) instead. The
parameters regenerate the shape; the moments do not.

Each retained value carries a **why** (what it controls in generation) and a **nature**:

- **C (constructive)** — realisable by direct construction / sampling.
- **E (emergent)** — falls out of the topology; reachable only by steering.
- **D (derived)** — exact function of other entries → **excluded** from the base.

### Decisions already made

- **Per-relation multiplicity must be accounted for** as a first-class free parameter.
  Consequences (made precise in *Investigation* below):
  - the **current generator is incorrect** — it collapses all per-relation
    functionalities to one scalar and draws multiplicity from a Geometric
    (`n_obj = geometric(mean_functionality)`), ignoring per-relation multiplicity.
    Treated here as a **deficiency to fix**, not an option.
  - **functionality / inverse functionality are derived**, not stored (head of the
    stored multiplicity law).
  - **aggregate out/in-degree are kept as targets** — the multiplicity *marginals* do
    not pin the compound sum (see Investigation / derivability criterion); only the
    multiplicity *scale* is derived.
- **Relation frequency** → originally proposed as Zipf / power-law; implemented as a
  measured log-share quantile function instead (see
  [§ G1](#g1--relation-usage-skew)).
- **Emergent targets** → stored as **raw counts** (not size-normalised).
- **Literals / datatype properties** → **out of scope** (flagged completeness gap).

---

## Convention — distribution families come from the notes, not chosen here

| Quantity | Family (from notes) | Stored parameters |
|---|---|---|
| Per-relation object/subject multiplicity α (spread across relations) | **quantile function** + cutoffs | 7 quantiles at levels (0, .1, .25, .5, .75, .9, 1); q@0/q@1 pinned to ~1.4 / ~3.0 |
| Class size (entities per class) | **power-law** | α (+ x_min) |
| Characteristic-set size \|CS\| | **quantile function** | 7 quantiles (q@0 … q@1) |
| `M` co-occurrence singular values (rank curve, **V-normalised** `M/V`) | **exponential decay** | rate, magnitude scale (size-free) |
| `P(r\|t)` type-relation singular values (rank curve) | **exponential decay** | rate, magnitude scale |
| Per-type relation entropy (rank curve) | **exponential decay** | rate, magnitude scale |
| Co-occurrence row entropy | **quantile function** | 7 quantiles (q@0 … q@1) |
| Two-step pair frequencies (value set) | **truncated power-law** (free α) | α, v_min, v_max |
| Shortest-path length | **max/mean/var summary** (was skew-normal in the notes) | max (diameter), mean, var |
| Relation-usage frequency | **quantile function** (superseded the original Zipf-exponent decision — see below) | 7 quantiles (q@0 … q@1) over `log(E_r / Σ E_r)` |

**CS-frequency** is not covered by the notes, but **confirmed power-law** (consistent
with class size and the existing `cs_freq_stats` fit). It is stored as a **truncated
power-law** on the observed `[v_min, v_max]` range (recurrence counts are bounded by the
entity count): pinning the range keeps fits comparable across graphs — a free `x_min`
can land on the tail of one graph but the full body of another — and the bounded
reconstruction keeps the roundtrip W1 distance finite even when α ≤ 2.

### Reading these representations (what the parameters mean)

- **power-law `(α, x_min)`** — `P(x) ∝ x^(−α)` for `x ≥ x_min`. `α` is the **tail
  exponent**: larger α ⇒ lighter tail (few hubs, counts stay small); smaller α ⇒
  heavier tail (strong hubs). `x_min` is where power-law behaviour begins; below it the
  body is not power-law. Used for strictly-heavy-tailed counts (class size).

  Every such fit (`_fit_powerlaw`) is **truncated**: the support is pinned to `[1, max]`
  — the data's own range — not left unbounded. The quantities fitted this way (degrees,
  class sizes, per-relation multiplicity) are all bounded, and the generator samples them
  from a law truncated to those bounds, so the fit must be the MLE of the *same* law. An
  unbounded MLE has to normalise over an infinite tail, which forces `α > 1` and biases it
  upward; released from that constraint, the truncated α is markedly **shallower** —
  across the corpus the degree exponents fall from ≈2.2–2.9 to ≈1.0–1.8. Reported α values
  are therefore **not comparable to signature files measured before this change**.

  The roundtrip **W1 reconstruction** of the degree fits is bounded at the stored
  `{out,in}_degree_max` to match: `BlockB.distribution_fits()` emits them as
  *truncated* power-laws (`_distance.TRUNC_POWERLAW`), not the plain unbounded
  `POWERLAW`. Without the bound, a degree α near 1.5 makes the reconstructed tail
  diverge — on wn18rr_v4 the p99.9 cap landed at ~248 000 against a real max of 26,
  so a 0.06 difference in α blew `out_degree`'s W1 up to 468. Bounded at the max, the
  same two fits give W1 ≈ 1.2. (The degree *fit* was already pinned to `[1, max]`; only
  the reconstruction was throwing the bound away.)
- **quantile function (7 quantiles at levels 0, .1, .25, .5, .75, .9, 1)** — the
  non-parametric empirical inverse CDF: the stored values are the sample quantiles, so
  q@0/q@1 are the min/max (hard truncation cutoffs, e.g. per-relation α confined to
  ≈[1.4, 3.0]) and q@0.5 is the median. Replaces the former skew-normal fit: it is far
  more stable to estimate (no MLE shape parameter), directly invertible for
  inverse-transform sampling, and its L1 difference is the Wasserstein-1 distance. Used
  for skewed-but-unimodal real-valued quantities (per-relation multiplicity-α spread, CS
  size, row entropy). (Block F's shortest-path length is instead reduced to three
  summary statistics — max/mean/var — rather than a fitted distribution.)
- **exponential decay `(rate λ, magnitude scale A)`** — value at rank `k` ≈
  `A · exp(−λ k)`. `A` is the magnitude of the top-ranked value; `λ` is **how fast**
  values fall with rank (large λ ⇒ only the first few ranks matter; small λ ⇒ a long
  flat tail). Used for the **rank curves** whose order/top must be preserved: `M`
  co-occurrence singular values, `P(r|t)` type-relation singular values, and **per-type
  relation entropy** (the top = the most diffuse/generalist types, special like `M`'s
  dominant singular value).
- **truncated power-law `(α, v_min, v_max)`** — `p(v) ∝ v^(−α)` on `[v_min, v_max]`,
  describing the *set* of values without rank order. `α` is the free decay exponent;
  the bounds are required because these quantities are inherently bounded (and `α`≈1
  isn't normalisable unbounded). **Contains log-uniform as the special case `α = 1`.**
  Used for **two-step pair frequencies** — **originally an exponential-decay rank curve in
  the notes**, reformulated here as a value distribution (see below) — and for the
  **forward/inverse CS-frequency** recurrence counts (bounded by the entity count).
- **Zipf / power-law over frequencies `(exponent)`** — rank-ordered frequency
  `f(rank) ∝ rank^(−exponent)`. Larger exponent ⇒ usage dominated by the top few
  ranks; near 0 ⇒ near-uniform usage. Used for **path-template** and
  **tree-template** usage (Block E). Relation-usage frequency used this
  representation too, originally — see [§ Relation frequency is now a measured
  quantile function, not a fitted Zipf exponent](#relation-frequency-is-now-a-measured-quantile-function-not-a-fitted-zipf-exponent).
- **scalars** — `clustering_coefficient ∈ [0,1]` (fraction of closed triads),
  `degree_assortativity ∈ [−1,1]` (degree correlation across edges),
  `largest_component_fraction ∈ (0,1]` (share of nodes in the giant component).

### Rank curves as value distributions — two-step pair frequencies

The notes originally describe **two-step pair frequencies** as an **exponential-decay
rank curve**. This proposal reformulates it as a **value distribution**, specifically a
**truncated power-law `(α, v_min, v_max)`** with a free exponent. A rank curve and a value
distribution are the *same object*: the sorted-descending curve `r(k) = v₍ₖ₎` is the
inverse CCDF scaled by the item count `n` (`F̄(v) = rank(v)/n`).

(*Per-type relation entropy* was also a value-distribution candidate, but its **rank
order is meaningful** — the top = the most diffuse types, special like `M`'s dominant
singular value — so it is kept as an **exp-decay rank curve**, not reformulated. Only
two-step pairs make the switch.)

Why a **free-α power-law** rather than fixed log-uniform: the implied value
distribution of an exponential rank curve is exactly **log-uniform** (`p(v) ∝ 1/v`,
i.e. `α = 1`) — but "exponential" was an eyeballed read on noisy few-graph curves.
A truncated power-law leaves `α` free, **contains log-uniform as `α = 1`**, and unifies
these entries with the signature's other power-law features (class size, CS frequency,
relation frequency). It can only fit better; if the decay really is exponential-in-rank
the fit returns `α ≈ 1`. Truncation to `[v_min, v_max]` is kept because the quantities
are bounded (entropy ≤ ln R; frequency ≤ 1) and `α ≈ 1` is not normalisable unbounded.
⚠️ Once more graphs are measured, **check whether the fitted `α ≈ 1`**: if it
consistently does, log-uniform was right and `α` is just insurance; if not, the free
exponent has bought real accuracy.

**Why switch two-step pairs:**

1. **Size-decoupling / cross-graph comparability (Stage-1).** Rank `k` is not
   comparable across graphs — the number of distinct two-step pairs differs. A value
   distribution is independent of that count, so the Stage-1 conditional-on-size fit sees
   a size-free *shape* plus the count stored separately. The rank-curve form conflates
   "how many pairs" with "how the values spread."
2. **Non-over-determination.** The number of distinct pairs is its own scalar; a rank
   curve re-encodes it, the distribution view stores shape only and reconstructs the
   curve from shape + count.
3. **Order is not meaningful for pairs.** Two-step pair frequency has no privileged top
   element whose exact value must be preserved — unlike singular values and per-type
   entropy, where the top rank *is* special. So discarding rank order loses nothing here.
4. **Generative fit.** Stage 2 produces a set of values of known cardinality (one
   frequency per pair): sampling that many draws is the natural operation.

**Singular values and per-type entropy stay rank curves** (exponential decay), precisely
because their top ranks dominate / are special, and a resampled distribution
under-represents the rare top values.

This applies whether the two quantities stay Stage-3 targets or become free parameters;
only the representation changes.

---

## Investigation — how per-relation multiplicity and degree depend on each other

This is why the multiplicity *scale* is **derived** (guaranteed by edge conservation),
while aggregate degree is **not guaranteed** by the marginals and is **kept as a
target**. Notation for a directed RDF graph with edges `(s, r, o)`:

- `m_obj(s, r)` = #distinct objects subject `s` reaches via relation `r` (its fan-out
  on `r`). The **object-multiplicity** distribution of `r` is `{m_obj(s,r)}` over
  subjects using `r`; Block B fits its tail → α.
- `m_subj(o, r)` = #distinct subjects reaching object `o` via `r` (fan-in on `r`).
- `CS(v)` = set of relations `v` uses as a subject.

**1. Degree is a compound sum of per-relation multiplicities.**

```
out_degree(v) = Σ_{r ∈ CS(v)}  m_obj(v, r)
in_degree(v)  = Σ_{r}          m_subj(v, r)
```

Out-degree is the sum of `|CS(v)|` terms, each drawn from the corresponding relation's
multiplicity law. So the aggregate out-degree distribution is a **compound
distribution** determined by three things already in the signature:

- the **CS-size** distribution (how many terms, G3),
- which relations a subject uses (**relation frequency** / `P(r|t)`, G1/G3),
- the **per-relation multiplicity** laws (the terms, G2).

Given those, aggregate out-degree is a **consequence** of the stored params — but only
of their **joint**, not their marginals. ⚠️ **Correction (see derivability criterion):**
the stored CS-size and multiplicity-α **marginals do not pin the compound sum** — it
also depends on *which* relations sit in each CS and on CS-size↔multiplicity
correlation. So aggregate degree is **not guaranteed** from the stored params and is
**kept as a target**, *unless* one explicitly assumes independence (then it is
guaranteed-by-construction and may be dropped).

**2. The degree tail is inherited from the heaviest multiplicity tail.** For a sum of
heavy-tailed (power-law) terms, the sum's tail exponent equals the smallest exponent
among the terms (the "single big jump": one large summand dominates). So *under
independence* the aggregate out-degree exponent ≈ the heaviest per-relation multiplicity
exponent present in a typical CS — a useful sanity check, but **not** a guarantee, since
real CS×multiplicity correlation shifts it.

**3. Edge conservation pins the multiplicity *scale* (so we store only the shape).**
Let `n_s(r)` = #distinct subjects using `r` and `n_o(r)` = #distinct objects of `r`
(both fixed by the schema / CS, G3). The number of `r`-edges is fixed by the edge
budget and relation frequency:

```
|edges_r| = freq(r) · E
mean m_obj(·, r) = |edges_r| / n_s(r)
mean m_subj(·, r) = |edges_r| / n_o(r)
```

So each relation's multiplicity distribution must hit a **mean fixed by edge
conservation**. Therefore the signature stores only the multiplicity **shape**
(the tail exponent α, via the quantile function across relations); the **scale / x_min** is
**derived** from `freq(r)·E / n_s(r)`. Storing a scale too would over-determine.

**4. The two multiplicity sides are coupled per relation (bipartite realisability).**
For one relation `r`, its edges form a bipartite graph between its subjects and
objects. The subject-side degree sequence *is* the object-multiplicity; the object-side
degree sequence *is* the subject-multiplicity. Both can be specified, but only subject
to `Σ m_obj(·,r) = Σ m_subj(·,r) = |edges_r|`. So object- and subject-multiplicity for
the same relation are jointly realisable, not independent — a bipartite
degree-sequence (configuration-model) constraint.

**5. Functionality is the head of the same distribution.**
`functionality[r] = P(m_obj(·,r) = 1)` — a point of the very distribution G2 stores. So
it is derived, not a separate value. The notes' "peak at 1 / peak at 0" mirror
observation is just this head probability being large.

**Summary of dependencies**

```
E, freq(r)                → |edges_r|                         (edge budget × frequency)
schema / CS               → n_s(r), n_o(r), CS(v)             (who uses r; CS sizes)
|edges_r|, n_s(r), n_o(r) → multiplicity MEAN/scale per r     (derived, guaranteed)
G2 quantile fn of α       → multiplicity SHAPE per r          (free)
object-multiplicity head  → functionality                     (derived, guaranteed)
per-relation multiplicity → out/in-degree distributions       (NOT guaranteed — joint)
```

---

## Derivability criterion — what may actually be dropped

A quantity may be dropped from the signature **only if it is guaranteed by the stored
parameters** — i.e. an exact function of stored values, with **no unstored
joint/correlation entering**. "Derivable under an independence assumption" is **not**
sufficient. Three tiers:

1. **Exact arithmetic of stored scalars** — e.g. `density = E/V²`. Guaranteed → drop.
2. **A property read off one stored distribution itself** — e.g.
   `functionality[r] = P(count = 1)` is the mass-at-1 of relation `r`'s *own*
   multiplicity law (whose α is stored). No cross-relation joint → guaranteed → drop.
3. **An aggregation across relations/entities that needs the joint** — e.g.
   `out_degree(v) = Σ_{r∈CS(v)} m_obj(v,r)` is a compound sum whose result depends on
   *which* relations sit in each CS and on CS-size↔multiplicity correlation. The stored
   **marginals** (CS-size dist, multiplicity-α dist) do **not** pin it → **not
   guaranteed → keep as a target.**

The dividing line is whether the derivation stays inside a single stored distribution
(guaranteed) or crosses into unstored joint structure (not guaranteed). This is also why
the lossy **co-occurrence spectrum** does not pin row entropy / cooc density /
type-relation entropy, and why object-side **wiring aggregations** (in-degree,
inverse-CS size) are not pinned by subject-side params.

### Final classification

**Drop (guaranteed):**

- `num_triples` (E), `density`, `relation_reuse` — tier 1. (Mean degree `E/V` = the old
  `triples_per_entity` is **kept** as the edge-budget handle.)
- `functionality`, `inverse_functionality` — tier 2 (head of the stored multiplicity law).
- per-relation multiplicity **scale / x_min** — fixed by edge conservation (tier 1 given
  the schema counts).
- all `*_ks` fit-quality fields — measurement diagnostics.

**Keep as targets (NOT guaranteed — cross into unstored joint):**

- aggregate **out/in-degree** distributions — compound sum; depends on CS×multiplicity
  joint. *(Drop only if you explicitly assume independence and accept the gap.)*
- **inverse-CS size** — object-side wiring aggregation; subject-multiplicity gives
  per-relation fan-in magnitude, not the count of distinct in-relations.
- **row entropy** (quantile function), **co-occurrence density** (scalar), **`P(r|t)` spectrum**
  (exp-decay) + **per-type relation entropy** (exp-decay rank curve) — type/co-occurrence
  functionals the lossy `M` spectrum alone does not pin.
- **two-step pair frequencies** (value-set truncated power-law) — wiring functional.
- connectivity (components, LCC, clustering, assortativity, shortest path) and the other
  motifs/templates — global emergent.

### Two caveats that are load-bearing

- **Clustering must stay average-*local*** (`transitivity_avglocal_undirected`). *Global*
  transitivity `= 3T / Σ C(deg,2)` would be exactly derivable from triangle count +
  degree → redundant. Average-local is a per-node functional and is **not**, so it stays.
- **Consistency web ≠ redundancy.** Edge conservation links several free params
  (`meanCS_size · mean_multiplicity = E/V`; `|edges_r| = freq(r)·E`;
  `Σ CS-frequencies = V`). These are satisfied by **deriving** the scales (multiplicity
  scale, CS-frequency scale). Do **not** also store those scales, degree mean, etc. — that
  would re-introduce over-determination.

---

## Design rule 0 — the edge-assignment primitive

With per-relation multiplicity free, the only remaining choice is its pairing with CS:

- **P2 — Multiplicity-first.** Free: per-relation multiplicity shapes + each subject's
  CS membership; aggregate degree emerges as the compound sum.
- **P3 — CS-first.** Free: per-subject CS (size + content) first, then per-relation
  multiplicity *given* the CS; degree emerges.

(**P1 — degree-first is excluded**: it would make per-relation multiplicity emergent,
contradicting the decision.) P2 vs P3 differ in whether CS membership or multiplicity is
sampled first; both keep per-relation multiplicity free.

---

## The proposed signature

Legend: **C** constructive · **E** emergent · **D** derived (excluded).

### G0 — Global size & vocabulary (root parameters)

| Value | Repr. | Nature | Why included |
|---|---|---|---|
| `num_entities` V | int | C | Root scale and the Stage-1 conditioning variable. |
| **mean degree** `E/V` (= old `triples_per_entity`) | float | C | Edge-budget handle (**decided**: mean degree, the most size-stable of E / mean-degree / density). With V it fixes E; with frequency it fixes `\|edges_r\|` (see Investigation). |
| `num_relations` R | int | C | Predicate-vocabulary size; schema width. |
| `num_classes` T | int (≥0) | C | Type-vocabulary size; 0 ⇒ untyped KG. |

Excluded **D**: `num_triples` (= `mean_deg·V`), `density` (= `mean_deg/V`),
`relation_reuse` (= `mean_deg·V/R`) — all functions of V, mean degree, R.

### G1 — Relation-usage skew

<a id="relation-frequency-is-now-a-measured-quantile-function-not-a-fitted-zipf-exponent"></a>

| Value | Repr. | Nature | Why included |
|---|---|---|---|
| relation-frequency distribution (`rel_freq_logq`) | **quantile function** (7 levels) over `log(E_r / Σ E_r)` — the log relation-edge-share | C | How unevenly relations are used. With E, fixes `\|edges_r\|` per relation. |

**Superseded design note.** This block originally proposed a fitted Zipf/power-law
**exponent**, and an early implementation had no measurement for it at all
(`generator.py` hard-coded `Zipf = 2.0`). Both are now stale: relation frequency is
measured directly as the 7-level quantile function above and Stage 1 reconstructs the
per-relation weights from it (evaluating the stored rank curve at `R` points), because
with `R` small the rank curve *is* the signal and a single fitted exponent lost against
it on every corpus graph. `relation_zipf_exponent` survives only as
`Generator.sample()`'s fallback for a graph with no relations at all. See
[generator.md § Relation frequency](generator.md#relation-frequency) for the
current mechanism.

### G2 — Per-relation multiplicity (core free block)

| Value | Repr. | Nature | Why included |
|---|---|---|---|
| object-multiplicity α spread | **quantile function** (7 levels) + cutoffs ≈[1.4, 3.0] over the per-relation tail exponents | C | Stores the **shape** of each relation's fan-out (the scale is derived by edge conservation). Sample an α per relation, then per-(subject,relation) counts. |
| subject-multiplicity α spread | **quantile function** (7 levels) + cutoffs | C | Mirror: shape of each relation's fan-in. Jointly realised with object-multiplicity per relation (bipartite constraint). |
| object / subject multiplicity **maximum** | 2 scalars `obj_mult_max`, `subj_mult_max` (max over all `(subject, relation)` / `(object, relation)` counts) | C | The **support** of the two α laws above. Multiplicity is inherently bounded and its α is a truncated MLE over `[1, max]`, so the generator needs the bound to draw from the same law it fitted — without it the draw is unbounded and overshoots the observed maximum. See the truncated power-law contract in `user_docs/generator.md`. |
| per-relation reciprocity | `P(symmetric)` per **edge-frequency bin** (6 fixed bins, cumulative-edge-mass thresholds) + one scalar `symmetric_recip_value` (magnitude of the symmetric mode) | C | Whether a relation's edges are bidirectionally reciprocated (`a→b,r` ⇒ `b→a,r`) is near-**bimodal** across the corpus and strongly tied to the relation's own **frequency rank** — a plain marginal quantile over relations (independent of frequency) was found to put reciprocity on the wrong relations at generation time (e.g. the single biggest relation drawing ρ≈0 despite being symmetric in the original). Binning by frequency and reconstructing via a frequency-rank lookup preserves that pairing; the bimodality is what collapses the general `P(reciprocity｜frequency)` joint to a 1-D mixing weight instead of a full 2-D quantile grid. See `developer_docs/notes/relation_reciprocity_and_bidirectionality.md`. |

Derived from G2 (not stored): `functionality`, `inverse_functionality` (distribution
heads), and the multiplicity **scale/x_min** (edge conservation). **Aggregate
out/in-degree are NOT derived from the marginals** — they are kept as targets in G4
(compound-sum joint). See Investigation.

#### G2b — CS-size offset on multiplicity (chosen)

| Value | Repr. | Nature | Why included |
|---|---|---|---|
| CS-size→multiplicity offset coefficient | 1–2 **global scalars** `a_obj` (`a_subj`) | C | Injects the CS-size↔multiplicity correlation the marginals discard → makes **out-degree** come out right at the constructive stage (closes the out-degree gap, Investigation §1). |

**Where it comes from.** The spec's Block D includes a *per-CS cardinality vector*
(`E[mult | CS, r]` — for each CS template, the typical fan-out of its relations). That
cannot enter the signature as a **CS-keyed table**: CS keys don't transfer across graphs
and don't exist until Stage 2 samples concrete CSs. So it is **reduced** to a dependence
on the one CS property available when a CS is drawn — its **size** `cs_size(s) = |CS(s)|`,
the number of predicates subject `s` uses. (Distinct from the spec's `|CS|` =
*number of distinct* CSs = `num_distinct_cs`.)

**What is stored vs computed.** Only the coefficient(s) `a` are stored (global, *not*
per-relation; the per-relation shape `α_r` already lives in G2). The offset itself is
computed per subject at generation time:

```
log m_obj(s, r) = a_obj · log cs_size(s)  +  (per-relation base: α_r from G2, scale from edge conservation)
```

i.e. a multiplicative factor `cs_size^a` on the multiplicity **location** (not its shape).
`a = 0` ⇒ no CS-size dependence (pure marginal); `a > 0` ⇒ big-CS subjects fan out more;
`a < 0` ⇒ less. **Measured** by regressing per-edge `log m_obj` on `log cs_size(subject)`
(per side).

**Decision: use this (option a) now** — CS-size exists for every graph (typed or untyped),
so it always works. A **type-conditioned** multiplicity table (option b: `mult(r|t)`
low-rank like `P(r|t)`, valid to the degree *type determines CS* — measurable via schema
regularity) is **deferred to later**.

*Optional G2c — per-relation joint.* Store the **joint** over relations of
`(freq, obj_α, subj_α)` instead of independent marginals (G1 + G2). *Why:* captures
correlations (frequent relations may be more functional) the marginals miss. *Cost:* a
3-D joint, harder to fit from few graphs. Option, not default.

### G3 — Schema: types & relation co-occurrence (decided: S-C = both)

The three options are listed for rationale; **S-C (both spectrum and CS distributions)
is chosen** — they are complementary (see S-C below). The "emergent under S-A/S-B only"
notes describe what each option *alone* would lose; with both stored, neither's losses
apply.

**Common to all:**

| Value | Repr. | Nature | Why included |
|---|---|---|---|
| class-size distribution | **power-law** exponent (+ x_min) over entities-per-class | C | How entities spread over types; also fixes `n_s(r)`/`n_o(r)` jointly with the relation usage. |

**Option S-A — Type→relation P(r\|t) via spectrum** (what the generator does)

| Value | Repr. | Nature | Why |
|---|---|---|---|
| `M` co-occurrence spectrum | **exponential-decay** (rate, scale) of subject- and object-side singular values, **V-normalised** (`M/V`) so `scale` is size-free | C | Relation co-occurrence on entities (type-agnostic). |
| `P(r\|t)` type-relation spectrum | **exponential-decay** (rate, scale) of the `T×R` singular values | C | The type→relation structure itself (separate object from `M`; resolves the item-3 type-light gap). Fed to the generator's low-rank `P(r\|t)`. |

Note: the `generator/` package reconstructs `P(r|t)` from `P(r|t)`'s **own** `type_rel_spectrum_exp`
(`stage1.sample_schema`), kept separate from `M`'s co-occurrence spectrum.

Under S-A, **derived/emergent:** `cooc_density`, `row_entropy`, the CS distribution,
two-step pairs.

**Option S-B — CS-centric**

| Value | Repr. | Nature | Why |
|---|---|---|---|
| CS-size distribution | **quantile function** (7 levels) over `cs_size(s)` | C | Per subject, how many distinct predicates it uses (number of terms in the degree sum). |
| distinct-CS count | `num_distinct_cs` (or fraction of V) | C | Degree of CS reuse / schema regularity. |
| CS-frequency distribution | **truncated power-law** (α, v_min, v_max) over CS occurrence counts | C | How skewed CS reuse is (schema regularity / template reuse). |

Under S-B, **emergent:** co-occurrence spectrum, densities, row/type entropies, inverse
CS, two-step pairs.

**Option S-C — Both (chosen).** Store spectrum **and** CS-size/CS-frequency. These are
**complementary, not redundant**: the spectrum captures *which* predicates co-occur
(correlation → P(r\|t)); CS-size captures *how many* predicates per subject; CS-frequency
captures *how often* a set recurs. The reduced CS parameters carry **no** co-occurrence
information (they discard *which* predicates form each set), so a CS cannot be sampled
to a target co-occurrence structure without the spectrum. Both are required.

**Schema-side kept targets** (not pinned by the lossy stored params — see the
derivability criterion):

| Target | Repr. | Why kept |
|---|---|---|
| row entropy | **quantile function** (7 levels) | a co-occurrence-matrix functional the spectrum doesn't pin |
| co-occurrence density | scalar ∈ (0,1] | nnz fraction; not pinned by the spectrum |
| **type-relation `P(r\|t)` spectrum** | **exp-decay** (rate λ, scale A) of the `T×R` matrix's singular values — *like `M`* | structure: #type-archetypes + concentration; fed directly to the generator's low-rank `P(r\|t)` factorisation. Optional scalar `I(R;T)` = how much type determines relation usage (validity of option b). |
| per-type relation entropy | **exp-decay rank curve** (rate, scale) | per-row spread of `P(r\|t)`; complementary to its spectrum (like `M` carries both SVs and row entropy). Rank order kept — the top = most diffuse/generalist types. |
| two-step pair frequencies | **truncated power-law** (α, v_min, v_max) over the **path-count** values `path_count(q,p)=Σ_x deg_in(x,q)·deg_out(x,p)` | multiplicity-weighted 2-hop path count → predicts path-2 selectivity (not a bridge-node count) |
| inverse-CS size | **quantile function** (7 levels) over #distinct in-predicates per object | object-side wiring aggregation; mirror of forward CS-size, **not** given by subject-multiplicity |
| edge multiplicity (parallel) | scalar ≥1: directed content edges / distinct directed `(s,o)` pairs | how much two-or-more relations pack onto the *same* directed pair — the entity/CS marginals never pin this (they only fix which relations an entity uses, not whether two relations coincide on one pair). Zero pair-overlap collapses the synthetic simple graph toward a plain configuration model, inflating its edge count relative to the original and steering it off the original's degree sequence. See `developer_docs/notes/motif_reachability_and_edge_multiplicity.md`. |
| bidirectional ratio | scalar ∈[1,2]: distinct directed pairs / distinct undirected pairs | how often a pair is connected **both** directions; the aggregate companion to per-relation reciprocity (G2) — also catches *cross-relational* reciprocation (`a→b,r1` / `b→a,r2`) that a same-relation reciprocity feature cannot see. |

Genuinely derived/dropped: nothing extra here beyond the global drop list.

### G4 — Connectivity & degree (emergent targets; raw per decision)

| Value | Repr. | Nature | Why included |
|---|---|---|---|
| out/in-degree distribution | power-law α_out, α_in (+ x_min) | E | **Not** pinned by per-relation multiplicity marginals (compound sum needs the joint) → kept as a target. Drop only if independence is assumed. |
| component structure | **largest-component fraction** ∈ (0,1] (+ `num_components`) | E | Global reachability; one blob vs many islands. Keep one unless both wanted as targets. |
| shortest-path length | **max/mean/var** summary (`shortest_path_max`/`_mean`/`_var`) | E | Small-world-ness / diameter; central to query cost. Stored as three summary statistics (the notes' skew-normal fit was dropped as unstable to estimate). |
| average-local clustering | `transitivity_avglocal_undirected` ∈ [0,1] | E | Local triadic closure. **Must be average-*local*** — global transitivity `3T/ΣC(deg,2)` would be redundant with triangles + degree. |
| degree assortativity | scalar ∈ [−1,1] | E | Whether hubs attach to hubs; shapes navigation and motifs. |

### G5 — Motifs & templates (emergent targets; raw counts per decision)

| Value | Repr. | Nature | Why included |
|---|---|---|---|
| triangle count | raw count | E | 3-node closure. |
| 4-node motifs | raw counts: four-cycle, diamond, K4, tailed-triangle | E | Higher-order density beyond triangles. |
| 5/6-cycle counts | raw (sampled/estimated) counts | E | Longer cyclic structure. |
| path templates | per-k **Zipf exponent + entropy** (k = 2..K) | E | Label-sequence diversity along paths — query-shape realism. |
| tree templates | **Zipf exponent + entropy** | E | Branching label diversity. |
| star counts | **not included** — induced `star_count_k2..k10` are pinned by the CS distribution (Block D), so they are redundant. The counter's `count_stars` helper is retained, unused. |

> Raw counts are strongly size-dependent (per the decision) → more work for the Stage-1
> conditional-on-size model.

### G6 — Literals / attributes — **out of scope (decision)**

Not included. Consequence recorded honestly: a KG instantiated from this signature has
**no datatype/literal edges** — only object-to-object structure. The one acknowledged
gap in the "complete" claim.

---

## Explicitly removed (and why)

Only quantities that are **guaranteed** by the stored params (derivability criterion,
tiers 1–2) are excluded. Everything that crosses into an unstored joint is **kept as a
target** (see the criterion's classification) — notably aggregate degree, inverse-CS
size, row entropy, cooc density, type-relation entropy, and two-step pairs.

| Excluded | Reason |
|---|---|
| `num_triples` (E), `density`, `relation_reuse` | algebraic functions of V, mean-degree, R (tier 1). Mean degree `E/V` (the proposal's `triples_per_entity`) is kept as the handle. |
| **per-relation multiplicity scale / x_min** | fixed by edge conservation `freq(r)·E / n_s(r)` (tier 1 given schema counts) — store shape only |
| `functionality_*`, `inverse_functionality_*` | head `P(count=1)` of the stored multiplicity law (tier 2) |
| all `*_ks*` goodness-of-fit fields | measurement diagnostics, not generative parameters |
| raw singular **values** | the spectrum is stored as exponential-decay parameters (per notes) — its shape, not its raw values |

---

## Deviations from the proposal

Where this implementation intentionally departs from `scaling_laws_student_project.pdf`.
Each is a deliberate scope or design decision, not an omission; the inline note at the
corresponding block/function repeats the short version.

| Proposal asks | What we do | Why |
|---|---|---|
| Block A stores `density`, `triples_per_entity`, `relation_reuse` | **Dropped.** Block A keeps only `num_entities`, `num_relations`, and **mean degree** `E/V` (= the old `triples_per_entity`). | All three are exact algebraic functions of `V`, mean degree, and `R` (`density = E/V²`, `relation_reuse = E/R`) — tier-1 derivable, so storing them would over-determine the signature. See [Derivability criterion](#derivability-criterion--what-may-actually-be-dropped). |
| Block E induced **star counts** (`star_count_k2..k10`) | **Dropped** — not measured or vectorised (the `count_stars` helper is retained, unused, for tests / `scripts/cc_variance.py`). | The characteristic-set distribution (Block D) already pins per-subject fan-out, so induced stars are redundant with it. |
| Per-feature **standard errors** on every signature value (§3.3 step 2) | **Not stored** on the signature. | Estimator variance is characterised once, offline, in `scripts/cc_variance.py` / `scripts/estimator_variance.py` (the sampled Block E motif estimators) and reported in the writeup, rather than carried per-feature on every measurement. |
| `Generator.sample(num_triples=…)` — instantiate at an arbitrary target size (§3.4) | **Not implemented.** Size is pinned by Block A: `num_triples = round(V × mean_degree)` ([stage1.py](../src/kgsynth/generator/stage1.py)). | Honouring an arbitrary `num_triples` needs a rescaling law for the *extensive* features (raw motif counts, `\|R\|`, `\|T\|`, `num_distinct_cs`, `num_components` — see [notes/signature_size_dependence.md](../developer_docs/notes/signature_size_dependence.md)); that is the conditional-on-size model in [plan/stage1_population_sampler.md](../developer_docs/plan/stage1_population_sampler.md), blocked on data and needed only by Phase 2. |
| Phase 2 — the scaling-law study (query generation, QLever labelling, FICE grid, `Qerror(N)` fitting) | **Out of scope.** | This submission is Phase 1 (the `kgsynth` package). Phase 2 is documented as future work. |
| Block F path-length steering | **Not steered** — `shortest_path_max`/`_mean`/`_var` are measured but no target drives them. | The only cheap lever (hub-shortcut injection) is one-sided — it can shorten paths but not lengthen them, and the synthetic graph undershoots; see [generator.md § Path-length steering](generator.md#path-length-steering) (the archived root-cause analysis has since been pruned from this tree; git history has it). |
| Block B relation frequency: Zipf/power-law **exponent** (G1, §3.1) | **Replaced** with a measured 7-level **quantile function** (`rel_freq_logq`) over `log(E_r / Σ E_r)`; Stage 1 evaluates the stored rank curve directly rather than sampling from a fitted exponent. | With `R` (relation count) small, the rank curve *is* the signal — a single fitted exponent lost against it on every corpus graph. `relation_zipf_exponent` survives only as the Stage-1 fallback for a graph with zero relations. See [§ G1](#g1--relation-usage-skew) and [generator.md § Relation frequency](generator.md#relation-frequency). |

## Feature finiteness — which values are guaranteed on any real graph

Every NaN feature across all 9 corpus signatures (`data/signatures/*.json`) was enumerated once
(see `CHANGELOG.md` for the working notes) to separate "the generator forgot to handle this" from
"this NaN is the correct measurement." The generator's consumption of this signature
(`kgsynth.generator`) is built on that split — see
[generator.md § Target signature must be complete](generator.md#target-signature-must-be-complete)
for the consumer-side contract (`_validate_target`). From the measurement side:

**Never NaN on any corpus KG** — a real graph always has enough data to fit these:
`num_entities`, `num_relations`, `mean_degree` (Block A); `out/in_degree_fit.alpha`,
`out/in_degree_p90`, `out/in_degree_max`, `a_obj`, `a_subj` (Block B); `subj_cooc_exp`,
`obj_cooc_exp` (Block C); `cs_size_q`, `inv_cs_size_q`, `num_distinct_cs`, `inv_num_distinct_cs`
(Block D); `num_components`, `largest_component_fraction` (Block F).

**Legitimately NaN** — a real measurement outcome or a small-sample fit failure, not missing data:

| Feature group | Occurs when | Why |
|---|---|---|
| `obj/subj_mult_alpha_q*`, `subj/obj_row_entropy_q*` | fewer than `MIN_SAMPLES_FOR_FIT` (10) **relations** | too few points to fit a power-law |
| `recip_symmetric_frac_bin*` | empty frequency bin (small `R`) | no relation landed in that bin; borrow the nearest non-empty one |
| `recip_symmetric_value` | **no symmetric relation exists** | real outcome, not a fit failure |
| `cs_freq_*`, `inv_cs_freq_*` | small `R` → too few distinct CSs to fit | relation-driven, same root cause as the first row |
| `class_size_*`, `type_rel_spectrum_*`, `per_type_entropy_*` | `num_classes == 0` | **untyped KGs are the majority case** in the corpus (7 of 9), not an edge case |
| `path_template_*_k*` | no path of that length exists | real outcome |

This table is the KEEP side of the fallback-removal plan
(`developer_docs/plan/remove_unnecessary_fallbacks.md`) — everything else that was ever NaN-guarded on the
generator side and is *not* in this table turned out to be unreachable on real data, and that
guard has been deleted.

## Decisions — resolved

Schema = **both** spectrum + CS distributions (complementary). **Edge-budget handle =
mean degree** (`E/V`); E and density derived. **CS-frequency = truncated power-law**. Per-relation
multiplicity is free (shape only); functionality and multiplicity *scale* are derived
(guaranteed); **aggregate degree, inverse-CS size, row entropy, cooc density, `P(r|t)`
spectrum + per-type relation entropy, and two-step pairs are kept as targets**; **induced
star counts are not in the signature** (pinned by the CS distribution; the `count_stars`
helper is retained, unused).

Resolved against the project spec (the document):

| Decision | Resolution | Doc basis |
|---|---|---|
| Primitive | **P3 (CS-first)** | Stage 2 is CS-first |
| `P(r\|t)` construction | **spectrum-only** (low-rank factorisation) | Stage 1 step 4 |
| Per-relation marginals vs G2c joint | **marginals** | Stage 1 step 2 samples per-relation marginals |
| Aggregate degree | **keep as target** (PA hits in-degree; G2b the out side) | Block B + Stage 2 PA + degree-preserving Stage 3 |
| CS×multiplicity coupling | **(a) CS-size offset**; type-conditioned **(b) deferred** | Stage 2 uses per-relation/per-archetype multiplicity |
| Per-type relation entropy | **keep as exp-decay rank curve** (over-specified under spectrum-only, but kept for now) | — (user) |
| `I(R;T)` scalar | **omit for now** | not in the doc |

Still open / not addressed by the doc: **generator rewrite scope** (full refactor vs
compatibility shim) — implementation choice, default incremental. See *Future work* in
[notes/generation_algorithm_fit.md](../developer_docs/notes/generation_algorithm_fit.md) for the best-effort gaps.
