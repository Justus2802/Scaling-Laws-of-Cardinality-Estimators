# Plan: measure the reduced (non-over-determined) signature

> **Superseded historical build record.** This is the original build plan for the reduced
> signature and describes an early state: a **coexisting module** (before the `_orig_block_*`
> "full" measurements were folded in and removed), a **69-feature** vector (A3 + B18 + C23 +
> D16 + F9, Block E deferred, multiplicity/row-entropy/CS-size as **skew-normal**). None of
> those numbers or representations are current. The signature now has **124 features**
> (A3 + B33 + C29 + D25 + E27 + F7), with quantile functions in place of the skew-normal fits
> and Block F shortest-path stored as max/mean/var. For the live module reference see
> [../signature.md](../signature.md); this file is retained only as a record of how the
> build was originally scoped.

Branch: `feat/measure-reduced-signature`. Implements
[../signature.md](../signature.md). **Integration mode: coexisting module**
(revised from the original in-place *replace*). The reduced signature lives in a new
sibling package `src/kgsynth/signature/` that reuses the existing block infrastructure;
the original `src/kgsynth/signature/` package and `generator.py` are left untouched, so both
signatures remain runnable side by side. See **Realised implementation** below.

Schema decision: measure **both** the co-occurrence spectrum **and** the CS-size /
CS-frequency distributions — they are complementary (which predicates co-occur vs how
many / how often), not redundant.

## Realised implementation (coexisting module)

Built as `src/kgsynth/signature/`, importing the shared base from `signature` (the
`SignatureBlock` ABC with its `as_dict`/`to_serializable`/`from_serializable`,
`_serialize`, `_logging`, and `_utils`'s `_fit_powerlaw`/`PowerLawStats`). Scope =
Blocks **A, B, C, D, F**; Block **E (motifs) is deferred**. Library-backed fits live in
`_fits.py` (`scipy.stats.skewnorm`, `scipy.stats.linregress`, the `powerlaw` package);
`_plot_helpers.py` overlays each fit on the unsummarised data it was computed from
(every block keeps that pre-fit data — singular values, row entropies, per-relation
exponents, path counts, path lengths — on the object for `visualize`).

| File | Block | Vec len | Stored representation |
|---|---|---|---|
| `block_a.py` | A — G0 | 3 | `num_entities`, `num_relations`, **mean degree** `E/V` |
| `block_b.py` | B — G1/G2/G2b | 18 | out/in-degree power-law (target); relation-usage **Zipf**; obj/subj multiplicity-α **skew-normal** (cutoffs [1.4,3.0]); CS-size offsets `a_obj`,`a_subj` |
| `block_c.py` | C — G3 | 23 | class-size **power-law**; subj/obj co-occurrence **exp-decay** + density; row entropy **skew-normal**; `P(r\|t)` spectrum **exp-decay**; per-type entropy **exp-decay** |
| `block_d.py` | D — G3 | 16 | `num_distinct_cs`; CS-freq **power-law**; CS-size & inverse-CS-size **skew-normal**; two-step path-count **truncated power-law** |
| `block_f.py` | F — G4 | 9 | components, LCC fraction, avg-local clustering, assortativity; shortest-path **skew-normal** |

Reduced-signature feature count = **69** (A3 + B18 + C23 + D16 + F9). The NamedTuple fits
(`SkewNormFit`, `ExpDecayFit`, …) restore as plain tuples through the generic JSON
round-trip, so each block property re-wraps them to preserve attribute access.

CLI: `scripts/measure_signature_reduced.py` (mirrors `measure_signature.py`) writes to
**`<graph-dir>/signature/`** (the `data/graphs/<name>/signature/` corpus layout);
historically this was a separate top-level `sig_out_reduced/` dir, now retired. Tests: `tests/test_signature_reduced_fits.py` and
`tests/test_signature_reduced_blocks.py`.

The Steps below document the original *replace* design and the per-quantity rationale;
the realised module follows the same per-block content but as a sibling package.

## Context

The current 133-feature signature is over-determined (algebraic + cross-statistic
redundancy) and stores `mean/std/median` summaries that cannot regenerate a skewed
distribution. The reduced signature stores the **parameters of the distribution family
the notes fix** for each quantity, drops all derived values, and keeps emergent
connectivity/motif quantities as raw-count targets.

## Derivability criterion (governs every drop/keep below)

Drop a quantity **only if guaranteed by the stored params** — an exact function of
stored values with no unstored joint/correlation entering. "Derivable under an
independence assumption" is not enough. Tiers: (1) arithmetic of stored scalars; (2) a
property of one stored distribution read off itself; (3) an aggregation needing the
joint → **not guaranteed → keep as target**. Full statement + classification in
[../signature.md](../signature.md#derivability-criterion--what-may-actually-be-dropped).

**Kept as targets (not guaranteed):** aggregate out/in-degree, inverse-CS size, row
entropy, cooc density, `P(r|t)` spectrum + per-type relation entropy, two-step pair
frequencies, and all connectivity/motif/template quantities.
**Genuinely dropped:** density/ratios, functionality/inverse-functionality, multiplicity
scale, all `*_ks` fields. (Induced `star_count_k*` are **kept** — they are not the
degree-fixed `Σ C(deg,k)`.)

## Step 0 — doc fix (done)

`signature_redesign.md` corrected: schema = **both** spectrum + CS distributions
(complementary, not redundant); the derivability criterion added; aggregate degree,
inverse-CS, row entropy, cooc density, type-rel entropy, two-step, and induced stars
moved from "removed" to "kept targets".

## Step 1 — new fitting utilities (`src/kgsynth/signature/_utils.py` or new `_fits.py`)

All new representations need fitters; centralise them so every block reuses them.

| Fitter | Returns | Used by |
|---|---|---|
| `fit_skewnorm_truncated(values, lo=None, hi=None)` | loc ξ, scale ω, shape α_skew, lower/upper cutoff | per-relation multiplicity-α spread, CS size, row-entropy*, shortest-path length |
| `fit_powerlaw(values)` *(exists)* | α, x_min | class size, CS frequency, relation frequency |
| `fit_truncated_powerlaw(values)` | α, v_min, v_max | two-step pair frequencies (path-count) |
| `fit_exp_decay_rank(values)` | rate λ, scale A | `M` co-occurrence SVs; `P(r\|t)` type-relation SVs; per-type relation entropy (sorted) |
| `fit_zipf(counts)` | exponent (+ scale) | relation-usage frequency |
| `fit_cs_size_offset(cs_size, mult)` | slope `a_obj` (`a_subj`) | CS-size→multiplicity offset (G2b): OLS of `log m_obj` on `log cs_size(subject)` |

`fit_skewnorm_truncated` wraps `scipy.stats.skewnorm.fit`; cutoffs = observed min/max
(or fixed bounds for multiplicity-α ≈[1.4,3.0]). `fit_exp_decay_rank` = linear
regression of `ln(value)` on rank. Each returns a small fixed-length tuple/namedtuple
so `as_vector` stays fixed-length.

## Step 2 — per-block refactor (the reduced signature)

Maps redesign groups G0–G6 onto the existing block files. Each block keeps its class,
base-class serialization, and `visualize`; only the computed state + `as_vector` +
`feature_names` + `get_na_vec` change.

**Block A — G0 size/vocabulary.** Keep `num_entities`, `num_relations`, and **mean degree
`E/V`** as the edge-budget handle. **Drop** `num_triples` (= mean_deg·V), `density`,
`triples_per_entity`, `relation_reuse` (all derived from V + mean degree).

**Block B — G1 relation frequency + G2 per-relation multiplicity (+ degree target).**
- New: relation-frequency **Zipf exponent** (fit per-predicate edge counts).
- Replace per-relation `mean/std/median` with: object-multiplicity **α skew-normal**
  (loc, scale, shape, lo, hi) and subject-multiplicity **α skew-normal**.
- New (**G2b**): **CS-size→multiplicity offset** scalar(s) `a_obj` (`a_subj`) — OLS slope
  of per-edge `log m_obj` on `log cs_size(subject)` (CS-size = #predicates the subject
  uses, **not** `num_distinct_cs`). The reduced form of the spec's per-CS cardinality
  vector; closes the out-degree gap at construction. Needs each subject's CS size (from
  Block D) joined with its per-relation multiplicities. (Type-conditioned `mult(r|t)` =
  option b, deferred.)
- **Keep** aggregate out/in-degree power-laws as **targets** (NOT guaranteed by the
  multiplicity marginals — compound-sum joint). *(Drop only if independence is assumed.)*
- **Drop** `functionality` / `inverse_functionality` (guaranteed — head of the stored
  multiplicity law), multiplicity scale/x_min (edge conservation), all `*_ks` fields.

**Block C — G3 schema (co-occurrence + type-relation) + class size + schema targets.**
- Keep class-size **power-law** (α, x_min).
- Replace raw 10 `M` singular values with co-occurrence **exp-decay** params (subj
  rate+scale, obj rate+scale).
- New: **`P(r|t)` type-relation spectrum** → **exp-decay** (rate, scale) of the `T×R`
  matrix's singular values (the better summary of `P(r|t)`; fed to the generator's
  low-rank factorisation). Optional scalar `I(R;T)`.
- **Keep as targets** (NOT pinned by the lossy spectrum): `row_entropy` → **skew-normal**;
  `cooc_density` → scalar; **per-type relation entropy** → **exp-decay rank curve**
  (rate, scale) — rank order matters (top = most diffuse types), like `M`'s SVs.
  (`num_classes` → reported under G0.)

**Block D — G3 schema (CS side) + inverse-CS / two-step targets.**
- CS-size **skew-normal** (loc, scale, shape); CS-frequency **power-law** (α, x_min);
  `num_distinct_cs`.
- **Keep as targets** (object-side wiring aggregations, not pinned by forward CS):
  **inverse-CS size** distribution; **two-step pair frequencies** — the **path-count**
  `Σ_x deg_in(x,q)·deg_out(x,p)` distribution (truncated power-law, free α; predicts
  path-2 selectivity). *(Implemented in `_two_step_pair_stats`.)*
- **Drop** the old `mean/median/p90` summaries (replaced by distribution params).

**Block E — G5 motifs (raw-count targets).** Keep triangle, 4-/5-/6-cycle, diamond, k4,
tailed-triangle (raw counts); **induced `star_count_k*`** (k=2..10); path/tree template
zipf+entropy. The stars are *induced* (centre + k mutually non-adjacent leaves, as
measured by the color-coding counter), **not** the degree-fixed non-induced `Σ C(deg,k)`,
so they carry structure independent of the degree sequence.

**Block F — G4 connectivity (targets).** Keep `largest_component_fraction`,
`num_components`, `degree_assortativity`, and **average-local clustering**
(`transitivity_avglocal_undirected` — must stay *local*; global transitivity would be
redundant with triangles + degree). Replace `avg_shortest_path_length` + `_se` with
shortest-path **skew-normal** (loc, scale, shape).

## Step 3 — package wiring (`src/kgsynth/signature/__init__.py`)

- Update `GraphSignature.as_vector`/`as_dict` (lengths change automatically from the
  blocks) and `_BLOCK_NA_VEC`.
- `to_serializable`/`from_serializable` need no change (they round-trip `__dict__`).

## Step 4 — downstream updates (the blast radius of "replace")

- **`generator.py`** consumes removed/changed attributes and **will break**:
  - `b.functionality` → removed: derive multiplicity from the new object-multiplicity
    α skew-normal instead (fixes the "current generator is incorrect" item).
  - `c.subj_singular_values` (raw) → reconstruct singular values from the exp-decay
    params (rate, scale, rank) to build P(r\|t).
  - `d.cs_size_mean` → take the mean of the CS-size skew-normal; `d.cs_freq_stats.alpha`
    → CS-frequency power-law α (kept).
  Update `sample_schema`/`instantiate` accordingly.
- **Tests** (`tests/test_signature_block_*.py`): every `_VECTOR_LEN` and attribute
  assertion changes. Rewrite per block: new vector length, new feature names,
  serialization round-trip (already generic), and add fitter unit tests.
- **`scripts/measure_signature.py`** + **`scripts/plot_signature_distributions.py`**:
  feature names/counts change automatically via `feature_names()`; verify block-letter
  groupings still hold.

## Step 5 — verification

1. `pytest tests/` green (rewritten block + new fitter tests).
2. Run `measure_signature.py` on a small fixture and on AIDS; inspect `signature.json`
   — confirm reduced feature set, no derived values, parameters present.
3. `Generator.from_file(...).sample()` round-trip runs and `compute_signature` on the
   output is finite.
4. Re-run `measure_all_raw` on the small graphs; regenerate distribution plots.

## Resolved (against the project spec)

Primitive **P3** (CS-first); `P(r|t)` **spectrum-only** low-rank; per-relation
**marginals** (not G2c joint); aggregate degree **kept** as target; CS×multiplicity via
**(a) CS-size offset**, type-conditioning **(b) deferred**; `I(R;T)` **omitted**.

**Measurement action item from the spec:** add **depth-3 tree templates** (Block E) —
the spec wants rooted trees of depth 2 **and 3**; the code currently computes depth-2
only.

## Open items (not blocking)

1. **Generator rewrite scope:** full update now, or a thin compatibility shim
   (reconstruct old attributes from new params) to defer the generator refactor.
   (Default: incremental — "start tiny.")

> Resolved: edge-budget handle = **mean degree** (`E/V`); CS-frequency = **power-law**.

## Suggested order (keep tests green incrementally)

`_utils` fitters → Block A → B → C → D → F → E → `__init__` → generator shim/update →
tests → scripts → verification. One block per commit.
