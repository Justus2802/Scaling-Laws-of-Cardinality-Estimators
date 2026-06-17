# The Generator (`kgsynth`) — algorithms, step by step

The generator turns a measured **reduced signature** into a synthetic KG whose re-measured
signature lands near the target. It implements the project brief's three-stage procedure
([notes/generation_algorithm_fit.md](notes/generation_algorithm_fit.md),
[plan/generation_implementation_plan.md](plan/generation_implementation_plan.md)) against the
reduced blocks ([signature.md](signature.md)).

Code lives in the **`src/generator/`** package:

| Module | Role |
|---|---|
| `pipeline.py` | `Signature` (target) + `Generator.sample()` orchestrator |
| `schema.py` | `Schema` dataclass — Stage-1 output handed to Stage 2 |
| `stage1.py` | `sample_schema` — abstract schema from BlockA/B/C/D |
| `stage2.py` | `instantiate` — CS-first graph wiring |
| `stage3.py` | `refine` — Maslov–Sneppen rewiring + simulated annealing |
| `_adapters.py` | reduced-signature reconstructions (see below) |
| `_logging.py` | package logger (`generator.*`, INFO progress lines) |

Public API (re-exported from `__init__.py`, unchanged by the refactor):
```python
from generator import Signature, Generator
g = Generator(Signature.from_file("target.ttl")).sample(seed=42, rewire_budget=5000)
```
`Signature` holds reduced blocks `a, c, e` (required) and `b, d, f` (optional — each enables
more faithful structure). `sample()` derives sub-seeds so the whole pipeline is reproducible
from one integer: Stage 1 `seed`, Stage 2 `seed+1`, Stage 3 `seed+2`.

---

## Inputs — which signature fields drive generation

| Block | Field | Used for |
|---|---|---|
| A | `num_entities`, `num_relations`, `mean_degree` | `V`, `R`, edge budget `E = round(mean_degree·V)` |
| B | `relation_zipf` | relation-frequency weights (Zipf exponent) |
| B | `obj_alpha_skew` | per-relation **object**-multiplicity tail α (out-degree shape) |
| B | `subj_alpha_skew` | per-relation **subject**-multiplicity tail α (in-side shape) |
| B | `a_obj` | G2b forward CS-size→multiplicity offset (`cs_size^a_obj`) |
| B | `a_subj` | G2b inverse CS-size→multiplicity offset (`inv_cs_size^a_subj`) |
| B | `in_degree_fit.alpha` | PA exponent + expected max in-degree |
| C | `num_classes`, `class_size_fit.alpha` | type count + type-size weights |
| C | `type_rel_spectrum_exp` | `P(r\|t)` low-rank reconstruction (used for post-hoc type scoring) |
| C | `subj_cooc_exp` | forward co-occurrence group prototypes (CS source) |
| C | `obj_cooc_exp` | inverse co-occurrence group prototypes (inverse-CS source) |
| D | `cs_size_skew`, `num_distinct_cs`, `cs_freq_fit.alpha` | forward CS templates: size, count, reuse skew |
| D | `inv_cs_size_skew`, `inv_num_distinct_cs`, `inv_cs_freq_fit.alpha` | inverse CS templates (object side): size, count, reuse skew |
| E | motif counts | Stage-3 targets (triangles, 4-cycle, diamond, k4, tailed) |
| F | `degree_assortativity` | Stage-3 target |

**Validation-only (measured, deliberately *not* used constructively):** C `subj/obj_cooc_density`,
`subj/obj_row_entropy_skew`, `per_type_entropy_exp`; D `two_step_fit`; F `num_components`,
`largest_component_fraction`, `clustering_coefficient`, `shortest_path_skew`. These are
diagnostics the brief marks "get near," not directly steered. Tuning constants for each stage
are module-level at the top of `stage1.py`/`stage2.py`/`stage3.py`.

---

## Stage 1 — schema sampler (`sample_schema`)

Produces a `Schema`: relations, types, `P(r|t)`, and the per-relation/CS parameters Stage 2
needs. All randomness from one seeded `np.random.Generator`.

1. **Counts & budget.** `R = max(1, num_relations)`, `T = max(0, num_classes)`,
   `E = round(num_entities · mean_degree)` (reduced Block A stores mean degree, not `|E|`).
2. **Relations.** `relation_weights = Zipf(R, exponent)` where the exponent is the **measured**
   `b.relation_zipf.exponent` when Block B is present (else the `relation_zipf_exponent` param).
3. **Types.** `type_weights = Zipf(T, class_size_fit.alpha)`, uniform fallback when α is NaN/`T` tiny.
4. **`P(r|t)`.** Reconstruct a singular-value spectrum from `type_rel_spectrum_exp`
   (`scale·exp(−rate·k)`) and feed it to a low-rank random factorisation
   (`_sample_type_relation_probs`). This uses `P(r|t)`'s **own** T×R spectrum — *not* the `M`
   co-occurrence spectrum it used to be conflated with. No types (`T=0`) → empty `(0,R)` table.
5. **CS structure (Block D).** `cs_num_templates = num_distinct_cs`;
   `cs_template_zipf = cs_freq_fit.alpha` (CS reuse skew); `cs_size_skew` passed through; the
   CS-size-skew mean gates whether template mode is enabled.
6. **Co-occurrence group prototypes (Block C).** When `subj_cooc_exp` / `obj_cooc_exp` have a
   usable fit, reconstruct `COOC_NUM_GROUPS = 10` singular values and call
   `_sample_type_relation_probs` to build one `(k, R)` group-prototype matrix per side, plus a
   normalized weight vector from the singular values. Stage 2 draws entity CSes from these
   prototypes (replacing the `P(r|t)` path) and assigns types post-hoc. See
   [§ Co-occurrence groups](#co-occurrence-groups) below.
7. **Multiplicity / degree (Block B).** `obj_alpha_skew`, `subj_alpha_skew`, `a_obj` passed
   through; `in_pa_exponent = clip(1/(α_in−2), 0.1, 2)` (Dorogovtsev–Mendes); expected
   `max_in_degree = n^(1/(α_in−1))`.

---

## Stage 2 — CS-first instantiation (`instantiate`)

Builds the graph by sampling characteristic sets first, then wiring edges per relation. This is
where most of the fidelity fixes live.

1. **Budget split.** `content_E = E − n_type_edges` (one `rdf:type` edge per entity when `T>0`).
2. **CS size source.** Each CS's *size* is drawn from `cs_size_skew` (`sample_skewnorm_trunc`),
   falling back to a budget-derived Poisson mean when Block D is absent. CS size sets **relation
   membership only** — the per-relation allocation (step 5) owns the edge budget.
3. **Type assignment (initial).** Each entity gets a provisional type via `type_weights`
   (all untyped when `T=0`). If co-occurrence groups are active, this is overwritten post-hoc (step 5b).
4. **Distinct CS templates (forward + inverse) — two paths:**

   *Group path* (when `subj_group_probs` / `obj_group_probs` are set in Schema):
   - Assign each entity to a co-occurrence group drawn from the Zipf-weighted group distribution.
   - Build one template pool per group, sized ∝ group weight (`_build_distinct` from the group prototype).
   - Assign entities within each group to templates via `_assign_templates`.
   - (Inverse side mirrors this using `obj_group_probs`.)

   *Type path* (fallback when groups are None):
   Build `num_distinct_cs` **distinct** forward templates from `P(r|t)` (typed) / `relation_weights`
   (untyped) and `inv_num_distinct_cs` inverse-CS templates from `relation_weights`, as before.

5. **Entity → template assignment.** Per pool: **floor each template at ≥1 entity** (so every
   distinct CS is realised), then distribute the rest by a `power-law(reuse_zipf)` reuse tail —
   for forward (`cs_template_zipf`) and inverse (`inv_cs_template_zipf`) alike. This steers the
   distinct-CS counts *and* the reuse skews together. No inverse templates → every object eligible
   for every relation (today's behaviour) and `a_subj` stays inert.

   5b. **Post-hoc type assignment** (group path only, when `T>0`): once CSes are fixed, score each
   entity's CS against every type: `score(v, t) = Σ_{r ∈ CS(v)} log P(r|t)`. Assign
   `entity_type[v] = argmax_t score`. Entities with empty CSes fall back to sampling from
   `type_weights`. This makes type labels emerge from relation usage (the real causal direction)
   rather than being set independently of CS content.
6. **Per-relation wiring — multiplicity-then-PA with edge conservation, matched within `S_r × O_r`.**
   `S_r` = subjects whose forward CS contains `r`; `O_r` = objects whose inverse CS contains `r`
   (all entities when no inverse templates). For each present relation (`S_r`, `O_r` non-empty;
   weights renormalised over them):
   - `|edges_r| = min(round(renorm_weight[r]·content_E), |S_r|·|O_r|)` (capacity bound).
   - **Out-side** (per subject): weight `power-law(α_obj_r) · cs_size^a_obj` (G2 tail × G2b). **Floor
     each subject at 1**, allocate the surplus by `multinomial`, then **cap at `|O_r|`** + redistribute.
   - **In-side** (per object over `O_r`): weight `power-law(α_subj_r) · in_degree^pa · inv_cs_size^a_subj`
     (subject-multiplicity tail × hub preference × **G2b in-side offset**), masked by `max_in_degree`;
     allocate by `multinomial`, then **cap at `|S_r|`** + redistribute. The cap prevents condensation
     (`α_subj<2` or superlinear PA) from dumping the whole budget onto one object.
   - **Pair** subject-stubs with object-stubs within `S_r × O_r` (configuration model); on a
     self-loop or duplicate `(s,o)` **retry** by swapping in another pending object stub.
7. **Connect components** — bridge isolated components into the giant.
8. **`rdf:type` edges** for typed entities; assemble the `igraph.Graph` with the `kg_io.load_kg`
   attribute contract (so `compute_reduced_signature` can read it back).

The `_cap_redistribute` helper implements the symmetric cap (object ≤ `|S_r|`, subject ≤ `|O_r|`).
Stage-2 tuning constants (`MAX_PAIR_RETRY`, `CAP_REDISTRIBUTE_PASSES`, `SIZE_ESCAPE_FAILS`,
`TEMPLATE_ATTEMPT_*`, `FALLBACK_CS_MEAN_FLOOR`) are module-level at the top of `stage2.py`.

---

## Stage 3 — refinement (`refine`)

Degree-preserving **Maslov–Sneppen** double-edge swaps under **simulated annealing**, steering the
free-emergent targets *without* disturbing Stage-1/2 marginals (degree, CS, P(r|t) are invariant
under same-relation swaps):

- **Triangle count** — exact, incremental `_triangle_delta` (O(degree) per swap); targeted swaps
  preferentially close open wedges when below target.
- **4-node motifs** (C4, diamond, K4, tailed) — no cheap delta, so re-measured periodically via
  `igraph.motifs_randesu(size=4)` every `remeasure_interval` accepted swaps.
- **Degree assortativity** — exact, incremental (only the cross-product sum `Q` changes).

The SA loss is a weighted sum of relative errors; the best graph seen is returned, then components
are re-bridged.

> **Runtime note.** `motifs_randesu(size=4)` cost scales with `Σ_v C(deg(v),3)`, so it is expensive
> on hub-heavy graphs and is the dominant Stage-3 cost. Lower `--rewire-budget`, raise
> `remeasure_interval`, or skip 4-node steering (triangles + assortativity only) to speed it up.

---

## Reduced-signature adapters (`_adapters.py`)

The reduced blocks store distribution *parameters*, not the raw moments the generator originally
read. These helpers reconstruct what Stage 1/2 need (NaN-safe; NaN → neutral fallback):

- `_skewnorm_mean(fit)` — mean of a skew-normal fit (e.g. `cs_size_mean`).
- `_functionality_from_alpha(fit)` — `1/ζ(α)` = fraction of single-valued slots, from a
  multiplicity-α skew-normal (out-side `mean_functionality` fallback).
- `_reconstruct_singular_values(exp_fit, k)` — `scale·exp(−rate·k)` spectrum for `P(r|t)`.
- `sample_skewnorm_trunc(fit, n, rng)` — draws clipped to `[lo,hi]`; `None` when the fit is NaN.
- `sample_powerlaw(alpha, n, rng)` — `power-law(α>1)` draws via inverse-CDF; uniform ones when α
  is NaN/≤1 (the neutral fallback).

---

## Evolution & fixes (why the code looks the way it does)

In roughly the order they were made:

1. **Reduced-signature consumption + package split.** The monolithic `generator.py` was converted
   to read the reduced blocks and split into the `stage*`/`schema`/`pipeline`/`_adapters` modules.
   Stage 1/2 reads that lacked a direct attribute are reconstructed in `_adapters.py`.
2. **`P(r|t)` de-conflation.** Stage 1 reconstructs `P(r|t)` from its **own** `type_rel_spectrum_exp`
   instead of borrowing the `M` co-occurrence spectrum.
3. **Measured relation Zipf.** Relation weights use the measured `relation_zipf` exponent rather
   than a hard-coded 2.0 (restores brief §Stage-1 behaviour).
4. **Per-relation multiplicity-then-PA with edge conservation.** Replaced the per-entity
   `geometric(mean_functionality)` + global-throttle wiring with the per-relation `multinomial`
   allocation — wiring per-relation α (out-side) and the G2b `a_obj` offset, hitting the relation
   budget exactly.
5. **Faithful in-side allocation.** Replaced the hard inverse-functionality cap (which collapsed
   subject-multiplicity to `{1,2}`) with a per-relation **subject-multiplicity** allocation from
   `subj_alpha` × PA. The old cap is gone.
6. **Realizability cap + redistribute (budget-collapse fix).** With `α_subj<2` (infinite mean) or
   superlinear PA (`in_pa>1`, condensation), the in-side `multinomial` dumped almost all of
   `|edges_r|` onto one object → the excess was unplaceable duplicates → the budget collapsed
   (e.g. 14,554 → 6,159). Capping each object at `|S_r|` and redistributing the overflow restored it
   (→ ~0.99 of budget).
7. **`num_distinct_cs` fix.** Distinct templates alone weren't enough: a too-steep rank-Zipf
   assignment left most templates unused, and out-side multinomial zeros shrank realised CSs.
   Fixed with (a) distinct templates by rejection + size-escape, (b) **floored** entity→template
   assignment (≥1 entity per template) + power-law reuse tail, (c) **out-side floor** (every subject
   of `r` gets ≥1 edge). Realised `num_distinct_cs` on fb237-like went 229 → ~1030 (target 1298).
8. **Symmetric inverse CS + `a_subj` wiring.** Restored `inv_num_distinct_cs` + `inv_cs_freq` to the
   reduced Block D (Block D 16→19; total signature 96→99) and added inverse-CS templates (object
   side) — so edges are matched within `S_r × O_r` and the in-side weight gains the
   `inv_cs_size^a_subj` G2b factor. The old hard inverse-functionality cap is replaced by the in-side
   allocation; the out-side gains a symmetric cap at `|O_r|`. Re-measured `a_subj` tracks target
   (e.g. 0.6 → ~0.65).
9. **Configurable constants.** All tuning magic numbers hoisted to module-level constants at the top
   of `stage1.py` / `stage2.py` / `stage3.py`.

---

## Co-occurrence groups

### Why

`subj_cooc_exp` is the exp-decay fit of the V-normalised singular spectrum of M_subj (the R×R
relation co-occurrence matrix, where M_subj[r1,r2] = # entities using both r1 and r2). Without
explicit targeting, M_subj's spectrum, density, and row-entropy distribution are not reproduced.

The generator's CS assignment previously used `P(r|t)` (driven by `type_rel_spectrum_exp`, the
T×R spectrum) as the sole source of entity relation diversity. `P(r|t)`'s spectrum is a different
quantity from M_subj's spectrum — using one to target the other doesn't work.

### Mechanism

`subj_cooc_exp` / `obj_cooc_exp` are reconstructed into `COOC_NUM_GROUPS = 10` singular values
and fed into the same `_sample_type_relation_probs` low-rank factorisation used for `P(r|t)`.
This produces a `(k, R)` group-prototype matrix where each row is a probability distribution over
relations, and a weight vector (Zipf-shaped, from the spectrum's relative magnitudes).

Stage 2 then:
1. Assigns each entity to a group via the weight vector.
2. Draws the entity's CS from the group prototype (instead of `P(r|t)`).
3. Assigns types post-hoc: `entity_type[v] = argmax_t Σ_{r ∈ CS(v)} log P(r|t)`.

This makes co-occurrence structure primary and type assignment emergent, matching the real causal
direction in KGs.

### Why k = 10 (fixed)

Spectral entropy (`k_eff = exp(H(p_i ∝ σ_i))`) was considered as the criterion but conflates
group *count* with weight *uniformity*: a KG with 5 groups where one dominates gives k_eff ≈ 1.4
instead of 5. The weight distribution already encodes skewness; k only needs to be "large enough
not to miss structure." Block C measures exactly 10 singular values, so `COOC_NUM_GROUPS = 10`
uses the full available signal without over-extrapolating.

### Features targeted

| Feature | Path |
|---|---|
| `subj_cooc_exp.rate/scale` | Group weight spectrum → M_subj singular value shape |
| `obj_cooc_exp.rate/scale` | Inverse group weight spectrum → M_obj shape |
| `subj/obj_cooc_density` | Indirectly: group prototype spread × CS size |
| `subj/obj_row_entropy_skew` | Indirectly: group prototype concentration |

`per_type_entropy_exp` and class sizes now emerge from post-hoc type assignment (no longer
directly controlled).

---

## Known limitations / open items

- **Co-occurrence density and row-entropy not analytically pinned.** These emerge from
  group prototype concentration and CS size but have no independent control knob. Remaining
  deviations require either softmax temperature tuning per group or Stage-3 obj-side steering.
- **Inverse `num_distinct_cs` partially hit.** The inverse-CS templates fix `a_subj` and improve the
  inverse count, but the *realised* `inv_num_distinct_cs` undershoots the target (an object's realised
  inverse CS ⊆ its assigned template, since the in-side allocation can give 0 edges for some of its
  relations — there is no symmetric in-side floor). Forward side is closer (it has the out-side floor).
- **Aggregate vs per-relation in-degree.** The in-side prioritises per-relation `subj_alpha`;
  aggregate `in_degree.alpha` emerges (best-effort). Superlinear PA from `1/(α_in−2)` for
  `2<α_in<3` is condensation-prone — clamping `in_pa≤1` is the lever if it drifts.
- **Stage-3 motif over-shoot.** When Stage 2 produces a graph already far from the motif targets, a
  small `rewire-budget` can't close the gap; motif counts can land well above target. Larger budget
  / better Stage-2 clustering control is the open item.
- **Out of scope:** literals/datatypes (G6); semantic types (synthetic clusters only); depth-3 tree
  templates.

---

## Verification

```bash
.venv/bin/python -m pytest tests/test_generator_stage1.py tests/test_generator_stage2.py -q
.venv/bin/python scripts/signature_roundtrip.py <graph> [--rewire-budget N]
```
The round-trip loads a cached target signature, generates, re-measures, and prints a per-block
comparison (median relative error is the meaningful aggregate; mean/max are inflated by
near-zero-target features).
