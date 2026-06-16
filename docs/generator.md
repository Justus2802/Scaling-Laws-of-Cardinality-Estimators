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
| B | `a_obj` | G2b CS-size→multiplicity offset (`cs_size^a_obj`) |
| B | `in_degree_fit.alpha` | PA exponent + expected max in-degree |
| C | `num_classes`, `class_size_fit.alpha` | type count + type-size weights |
| C | `type_rel_spectrum_exp` | `P(r\|t)` low-rank reconstruction |
| D | `cs_size_skew` | CS size per template |
| D | `num_distinct_cs` | number of distinct CS templates |
| D | `cs_freq_fit.alpha` | CS reuse skew (`cs_template_zipf`) |
| E | motif counts | Stage-3 targets (triangles, 4-cycle, diamond, k4, tailed) |
| F | `degree_assortativity` | Stage-3 target |

**Validation-only (measured, deliberately *not* used constructively):** C `subj/obj_cooc_*`,
row-entropy; D `inv_cs_size_skew`, `two_step_fit`; F `num_components`, `largest_component_fraction`,
`clustering_coefficient`, `shortest_path_skew`. These are diagnostics the brief marks "get near,"
not steered. `a_subj` is currently best-effort (not wired — see Limitations).

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
6. **Multiplicity / degree (Block B).** `obj_alpha_skew`, `subj_alpha_skew`, `a_obj` passed
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
3. **Type assignment.** Each entity gets a type via `type_weights` (all untyped when `T=0`).
4. **Distinct CS templates.** Build `num_distinct_cs` **distinct** templates by rejection
   (`_build_distinct_templates`): draw a CS from `P(r|t)` (typed) / `relation_weights` (untyped),
   dedup by relation-set, and **escalate the minimum size** once small combos saturate (so the
   count keeps climbing up to each type's `P(r|t)` support). Without distinctness, size-1 CSs and
   frequency-concentrated draws collapse the realised `num_distinct_cs`.
5. **Entity → template assignment.** Per type pool: **floor each template at ≥1 entity** (so every
   distinct CS is realised), then distribute the remaining entities by a `power-law(cs_template_zipf)`
   reuse tail. This steers `num_distinct_cs` *and* the `cs_freq` reuse skew together.
6. **Per-relation wiring — multiplicity-then-PA with edge conservation.** For each relation `r`
   present in some CS (relation weights **renormalised over present relations**):
   - `|edges_r| = round(renorm_weight[r] · content_E)`.
   - **Out-side** (edges per subject `S_r`): weight `power-law(α_obj_r) · cs_size^a_obj` (G2 tail ×
     G2b offset). **Floor each subject at 1 edge** (an entity with `r` in its CS has
     object-multiplicity ≥1), then allocate the surplus by `multinomial`.
   - **In-side** (edges per object): weight `power-law(α_subj_r) · in_degree^pa` (per-relation
     subject-multiplicity tail × aggregate-hub preference), masked by `max_in_degree`; allocate by
     `multinomial`. **Cap each object at `|S_r|`** (an object can't have more distinct subjects than
     exist) and **redistribute the overflow** — this is what prevents condensation (`α_subj<2` or
     superlinear PA) from dumping the whole budget onto one object.
   - **Pair** subject-stubs with object-stubs (configuration model); on a self-loop or duplicate
     `(s,o)` **retry** by swapping in another still-pending object stub, so edges are re-routed, not
     dropped.
7. **Throttle** to `content_E` (safety net; the per-relation allocation already targets it).
8. **Connect components** — bridge isolated components into the giant.
9. **`rdf:type` edges** for typed entities; assemble the `igraph.Graph` with the `kg_io.load_kg`
   attribute contract (so `compute_reduced_signature` can read it back).

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

---

## Known limitations / open items

- **Type-less co-occurrence gap.** With `num_classes=0` there is no `P(r|t)`; CS composition uses
  marginal `relation_weights` only, so relation co-occurrence (`subj_cooc`) isn't reproduced. The
  `M` co-occurrence spectrum is measured but never consumed.
- **Aggregate vs per-relation in-degree.** The in-side now prioritises per-relation `subj_alpha`;
  aggregate `in_degree.alpha` emerges (best-effort). Superlinear PA from `1/(α_in−2)` for
  `2<α_in<3` is condensation-prone — clamping `in_pa≤1` is the lever if it drifts.
- **`a_subj` (G2b in-side) not wired** — needs the object's emergent inverse-CS size; left
  best-effort.
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
