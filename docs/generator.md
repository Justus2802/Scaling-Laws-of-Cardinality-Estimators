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
| B | `obj_alpha_q` | per-relation **object**-multiplicity tail α (out-degree shape), as a quantile function |
| B | `subj_alpha_q` | per-relation **subject**-multiplicity tail α (in-side shape), as a quantile function |
| B | `a_obj` | G2b forward CS-size→multiplicity offset (`cs_size^a_obj`) |
| B | `a_subj` | G2b inverse CS-size→multiplicity offset (`inv_cs_size^a_subj`) |
| B | `in_degree_fit.alpha` | PA exponent + expected max in-degree |
| C | `num_classes`, `class_size_fit.alpha` | type count + type-size weights |
| C | `type_rel_spectrum_exp` | `P(r\|t)` low-rank reconstruction (used for post-hoc type scoring) |
| C | `subj_cooc_exp` | forward co-occurrence group prototypes (CS source) |
| C | `obj_cooc_exp` | inverse co-occurrence group prototypes (inverse-CS source) |
| D | `cs_size_q`, `num_distinct_cs`, `cs_freq_fit.alpha`/`.v_max` | forward CS templates: size (quantile function), count, reuse skew + truncation |
| D | `inv_cs_size_q`, `inv_num_distinct_cs`, `inv_cs_freq_fit.alpha`/`.v_max` | inverse CS templates (object side): size (quantile function), count, reuse skew + truncation |
| E | motif counts | Stage-3 targets (triangles, 4-cycle, diamond, k4, tailed) |
| E | `rel_pair_affinity` | Stage-2 in-side affinity boost (see §Relation-pair affinity) |
| F | `degree_assortativity` | Stage-3 target |
| F | `num_components`, `largest_component_fraction` | Stage-2 connectivity target (nc, LCC fraction) |
| F | `shortest_path_max` | Stage-2 diameter cap (`path_hi_target = int(max)`) |
| F | `shortest_path_mean` | Stage-2 mean path-length target (passed through directly) |

**Validation-only (measured, deliberately *not* used constructively):** C `subj/obj_cooc_density`,
`subj/obj_row_entropy_q`, `per_type_entropy_exp`; D `two_step_fit`; F
`clustering_coefficient`, `shortest_path_var` (emergent — not steered). These are diagnostics
the brief marks "get near," not directly steered. Tuning constants for each stage
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
   `cs_template_zipf = cs_freq_fit.alpha` (CS reuse skew) and
   `cs_template_vmax = cs_freq_fit.v_max` (reuse-draw truncation — the fitted α covers the
   full bounded range, so unbounded draws would over-skew); `cs_size_q` passed through; the
   CS-size quantile-function mean (trapezoid integral) gates whether template mode is enabled.
6. **Path-length targets (Block F).** Read `path_mean_target = f.shortest_path_mean`
   (NaN when Block F absent or paths not sampled) and `path_hi_target = int(f.shortest_path_max)`
   (0 when absent or NaN). Both are stored in `Schema` and consumed by Stage 2 step 7b.
7. **Co-occurrence group prototypes (Block C).** When `subj_cooc_exp` / `obj_cooc_exp` have a
   usable fit, reconstruct `COOC_NUM_GROUPS = 10` singular values and call
   `_sample_type_relation_probs` to build one `(k, R)` group-prototype matrix per side, plus a
   normalized weight vector from the singular values. Stage 2 draws entity CSes from these
   prototypes (replacing the `P(r|t)` path) and assigns types post-hoc. See
   [§ Co-occurrence groups](#co-occurrence-groups) below.
7. **Multiplicity / degree (Block B).** `obj_alpha_q`, `subj_alpha_q`, `a_obj` passed
   through. Per-entity **target degree sequences** (`target_out_degrees`,
   `target_in_degrees`) are sampled purely from **signature-vector components** —
   never Block B's raw retained arrays (`sample_degree_sequence`): the top 10% of
   nodes draw from a power law with the fitted degree α truncated to
   `[p90, max]` (the `out/in_degree_p90` / `out/in_degree_max` scalars pin the
   tail), the remaining 90% from a Poisson body clipped at p90 whose mean is
   solved so the overall mean matches `E/V`. These replace the old extreme-value
   max-degree caps — the whole distribution (body, p90, max) is targeted rather
   than a single hard bound. When the p90/max scalars are unavailable (stale
   signatures) the targets are `None` and Stage 2 wires without degree steering.

---

## Stage 2 — CS-first instantiation (`instantiate`)

Builds the graph by sampling characteristic sets first, then wiring edges per relation. This is
where most of the fidelity fixes live.

1. **Budget split.** `content_E = E − n_type_edges` (one `rdf:type` edge per entity when `T>0`).
2. **CS size source.** Each CS's *size* is drawn from `cs_size_q` (`sample_quantiles_trunc`,
   inverse-transform of the stored quantile function), falling back to a budget-derived Poisson
   mean when Block D is absent. CS size sets **relation
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
   distinct CS is realised), then distribute the rest by a `power-law(reuse_zipf)` reuse tail
   with raw draws truncated at `reuse_vmax` (the measured max recurrence, mirroring the
   truncated-power-law fit) — for forward (`cs_template_zipf`/`cs_template_vmax`) and inverse
   (`inv_cs_template_zipf`/`inv_cs_template_vmax`) alike. This steers the
   distinct-CS counts *and* the reuse skews together. No inverse templates → every object eligible
   for every relation (today's behaviour) and `a_subj` stays inert.

   5b. **Post-hoc type assignment** (group path only, when `T>0`): once CSes are fixed, score each
   entity's CS against every type: `score(v, t) = Σ_{r ∈ CS(v)} log P(r|t)`. Assign
   `entity_type[v] = argmax_t score`. Entities with empty CSes fall back to sampling from
   `type_weights`. This makes type labels emerge from relation usage (the real causal direction)
   rather than being set independently of CS content.
5c. **Target degree assignment** (when Stage 1 sampled degree sequences): sampled target values
   are **rank-matched** to CS size (out-side; largest target → largest CS, floored at `|CS|` so the
   ≥1-edge-per-CS-relation floor stays feasible) and to inverse-CS size (in-side; random when no
   inverse CS). A multinomial top-up ensures `Σ targets ≥ content_E` so quota caps cannot starve
   edge conservation. The steering mechanism is selected by `schema.degree_mechanism`:
   - `"capacity"` (default, empirically best): allocation weight ∝ remaining quota
     `(target − placed)⁺`, plus a **hard per-node quota** via `_cap_redistribute(hard_cap=…)`.
   - `"chunglu"`: weight ∝ target degree (expected-degree model), no hard cap — matches the
     distribution in expectation only; evaluated and rejected (tail overshoot ≈ +116% on max-out).
6. **Per-relation wiring — multiplicity-then-degree-targeting with edge conservation, matched
   within `S_r × O_r`.** `S_r` = subjects whose forward CS contains `r`; `O_r` = objects whose
   inverse CS contains `r` (all entities when no inverse templates). For each present relation
   (`S_r`, `O_r` non-empty; weights renormalised over them):
   - `|edges_r| = min(round(renorm_weight[r]·content_E), |S_r|·|O_r|)` (capacity bound).
   - **Out-side** (per subject): weight `power-law(α_obj_r) · cs_size^a_obj · degree-target factor`
     (G2 tail × G2b × capacity/expected-degree). **Floor each subject at 1**, allocate the surplus
     by `multinomial`, then **cap at `|O_r|`** + redistribute; in capacity mode a per-subject hard
     quota (`target − placed`) is enforced via `_cap_redistribute`.
   - **In-side** (per object over `O_r`): weight `power-law(α_subj_r) · degree-target factor ·
     inv_cs_size^a_subj · exp(λ·affinity_boost)` (subject-multiplicity tail × quota/expected-degree
     × **G2b in-side offset** × **relation-pair affinity boost**); allocate by `multinomial`, then
     **cap at `|S_r|`** + redistribute, plus the per-object hard quota in capacity mode. When no
     degree targets exist, no degree factor is applied. The affinity boost
     `exp(λ · Σ_{r2 ∈ CS(o)} affinity[r, r2])` favours objects whose forward CS relations commonly
     follow relation `r` in the source KG — see §Relation-pair affinity.
   - **Pair** subject-stubs with object-stubs within `S_r × O_r` (configuration model); on a
     self-loop or duplicate `(s,o)` **retry** by swapping in another pending object stub.
7. **Connect components** — bridge isolated components into the giant, *selectively*: keeps up
   to `target_nc − 1` satellite components unbridged (chosen so their combined size is closest to
   `(1 − target_lcc) · V`), bridges the rest. Runs **first** among the connectivity-affecting
   steps and returns an `is_satellite` mask that the two passes below must respect — neither may
   place an edge touching a satellite node, or it would silently reconnect a component this step
   chose to leave isolated, undoing the `target_nc` / `target_lcc` guarantee.
7a. **Deficit recovery.** Per-relation budget vs degree-quota misalignment can leave part of the
   edge budget unplaced (capacity mode drops overflow rather than exceeding a node's target).
   The remainder is placed by sampling `(subject, object)` pairs weighted by remaining quota
   (+1e-3 smoothing so saturated pools still accept soft overflow) — edge conservation always
   wins. Subject/object pools are filtered to non-satellite nodes first; a relation left with an
   empty pool on either side is skipped for that attempt.
7b. **Path-length steering** (`_steer_path_lengths`) — runs only when `path_mean_target` or
    `path_hi_target` are set. Builds a temporary undirected igraph entity graph (igraph C backend)
    and runs up to 4 rounds of estimate → inject shortcuts, with all BFS-source and
    shortcut-endpoint sampling restricted to non-satellite nodes:
    - *Diameter (hi):* find farthest-pair from a random BFS; add
      `⌈(diam − hi_target)/2⌉` shortcuts, each source→its-farthest-node, until `diam ≤ hi_target`.
    - *Mean:* sample 50 BFS sources; if `mean > mean_target + 0.5`, inject
      `max(1, round(√V_ns · (mean − mean_target) / mean))` shortcuts (`V_ns` = non-satellite
      count) between nodes sampled ∝ degree.
    Each shortcut is added to both the igraph object and `content_edges`; `in_degrees` and `seen`
    are updated. See [§ Path-length steering](#path-length-steering) for design rationale.
8. **`rdf:type` edges** for typed entities; assemble the `igraph.Graph` with the `kg_io.load_kg`
   attribute contract (so `compute_reduced_signature` can read it back).

The `_cap_redistribute` helper implements the symmetric cap (object ≤ `|S_r|`, subject ≤ `|O_r|`).
Stage-2 tuning constants (`MAX_PAIR_RETRY`, `CAP_REDISTRIBUTE_PASSES`, `SIZE_ESCAPE_FAILS`,
`TEMPLATE_ATTEMPT_*`, `FALLBACK_CS_MEAN_FLOOR`) are module-level at the top of `stage2.py`.

### Relation-pair affinity

When the source KG's Block E is computed from a live graph (not loaded from cache), it also
measures an **(R × R) row-stochastic affinity matrix** `P(r2 | r1)` where entry `[r1, r2]`
is the conditional probability that a path through a relation-`r1` edge continues with a
relation-`r2` edge at the object node.

This matrix is stored as `Schema.rel_pair_affinity` and used in Stage 2's **in-side weight**
for each relation `r`:

```
affinity_boost[o] = Σ_{r2 ∈ CS(o)} affinity[r, r2]
w_in[o] ×= exp(λ · affinity_boost[o])   (λ = 3.0)
```

Objects whose forward CS (outgoing relations) commonly follow relation `r` in the source KG
receive a multiplicative boost of up to ~20×, biasing the configuration model to wire
relation-`r1` edges into nodes that will serve as sources for relation-`r2` edges.
This directly addresses path/tree entropy errors by building the local two-step
relation-pair co-occurrence structure into the wiring rather than relying on Stage 3 rewiring.

**Availability:** the affinity matrix is computed only when `BlockE.calculate(g, ...)` is
called with a live graph (e.g. `--kg-file` mode in `signature_roundtrip.py`). When Block E
is loaded from a cached `block_e.json` (standard corpus mode), `rel_pair_affinity` is `None`
and the boost is skipped (no effect on wiring).

---

## Stage 3 — refinement (`refine`)

Degree-preserving **Maslov–Sneppen** double-edge swaps under **simulated annealing**, steering the
free-emergent targets *without* disturbing Stage-1/2 marginals (degree, CS, P(r|t) are invariant
under same-relation swaps):

All per-swap incremental delta helpers live in `generator/local_updates.py`; they operate on the
live adjacency dict (not an `igraph.Graph`) so each swap is cheap.

- **Triangle count** — exact, incremental `_triangle_node_delta` (O(degree) per swap); targeted
  swaps preferentially close open wedges when below target.
- **CC_avg** (avg local clustering) — exact, incremental; the `C(k_v,2)` denominators are invariant
  under degree-preserving swaps, so only per-node `Δt_v` (from the triangle delta) drives it.
- **4-node motifs** (C4, diamond, K4, paw) — exact, incremental `_motif4_delta`: enumerates the
  motif 4-sets touching the four swap endpoints before and after, and diffs per-type counts
  (O(Δ³) per swap).
- **5-/6-cycles** — these are **induced (chordless)** cycles (degree sequences `(2,2,2,2,2)` /
  `(2,2,2,2,2,2)`), matching the motif counters. Exact, incremental `_cycle_delta`: counts induced
  cycles through the swap endpoints before and after and diffs them (O(Δ⁴)/O(Δ⁵) per swap — best
  left off for hub-heavy graphs). A swap changes the count both by adding/removing a cycle edge
  **and** by adding a chord (destroys an induced cycle) or removing one (can create an induced
  cycle). Only active when a Block-E cycle target is > 0.
  Swaps that touch a node with simple degree > `CYCLE_DELTA_MAX_DEGREE` (default 50) **skip** the
  delta — on such hubs the O(Δ⁴)/O(Δ⁵) enumeration can stall the walk for minutes — and carry the
  cycle counts over unchanged, so the cycle loss terms cancel in the accept test and neither favour
  nor penalise the hub swap.
- **Degree assortativity** — exact, incremental (only the cross-product sum `Q` changes).
- **Depth-2 tree template entropy** — exact, incremental `_tree_entropy_delta` (O(Δ) per swap).
  Maintains a live `(r1, r2)` pair-frequency dict across the SA walk; on each candidate swap the
  old child's outgoing relations are removed and the new child's are inserted before computing
  Shannon entropy.  Only steered when `BlockE.tree_template_entropy > 0` and
  `LOSS_WEIGHT_TREE_ENTROPY > 0`.
- **k=3 path template entropy** — exact, incremental `_path_entropy_delta` (O(Δ) for k=2,
  O(Δ²) for k=3 per swap).  Tracks live `(r1, r2)` and `(r1, r2, r3)` path-frequency dicts using
  `out_edges[v] = [(rel, target), ...]` (directed adjacency).  The target object changes on accept,
  so `out_edges` is updated in-place per accepted swap.  Only steered when
  `BlockE.path_template_entropy[3] > 0` and `LOSS_WEIGHT_PATH_ENTROPY > 0`; inactive when Block E
  was measured with `skip_stars_and_paths=True`.
- **Induced k-star counts** (`k ∈ STAR_K_TRACKED = (2,3,4,5)`) — baseline measured via the CC
  star sampler (`initial_motif_counter.count_stars`); the exact inclusion-exclusion baseline is
  O(2^(inner edges)) per centre and stalls on clustered hubs, whereas the CC estimator is bounded.
  Per-swap updates use the exact, incremental `_star_count_delta` (O(Δ²) per swap), with a
  `STAR_CENTER_MAX_DEGREE` guard: centres above that simple degree are skipped (their
  inclusion-exclusion would explode). Degree is swap-invariant, so a skipped hub is excluded from
  both sides of the delta and contributes 0 — hub star changes neither favour nor penalise a swap.
  (The CC baseline still *includes* those hubs, so their contribution is a constant offset that
  cancels in the accept test rather than being tracked.) Unlike non-induced stars (`C(k_v,2)`, fixed by degree),
  these are **chordless** stars whose leaves must be mutually non-adjacent, so a degree-preserving
  swap that removes an inner edge among a hub's neighbours *raises* that hub's induced star count —
  the lever Stage 3 uses. Steered only when `LOSS_WEIGHT_STARS > 0` (and the Block-E target is > 0);
  **off by default** since stars are largely set by the Stage-2 out-degree distribution. An optional
  targeted move (`_targeted_star_swap`, gated by `MAX_STAR_TARGETED_PROB`) biases proposals toward
  breaking triangles among high-degree hubs to close a star deficit faster; it antagonises the
  clustering terms (triangles/CC/diamond/K4/paw), so keep the probability modest.

The SA loss is a weighted sum of relative errors. Both the loss and the convergence log derive
from a single `_error_terms(state)` helper (returning one relative error per active target), so each
term is defined once: the loss is the weighted sum, the log is the unweighted dump. The best graph
seen is returned, then components are re-bridged. When a `convergence_log` is given, each active
term writes a relative-error column every `CONVERGENCE_LOG_INTERVAL` accepted swaps (`tri_err`,
motif4 `*_err`, `c5/c6_err`, `cc_err`, `assort_err`, `tree_entropy_err`, `path_entropy_k3_err`, and
`star_k{k}_err` per tracked k). Setting `CONVERGENCE_LOG_GLOBAL_REMEASURE = True` additionally
re-measures the full graph each logged row and appends ground-truth `sig_*_err` columns (validates
the incremental deltas; expensive).

> **Runtime note.** The incremental deltas are O(Δᵏ⁻¹) per *attempted* swap, so they dominate
> Stage-3 cost on hub-heavy graphs. Lower `--rewire-budget` or disable a steering term (set its
> `LOSS_WEIGHT_*` to 0). The costliest cycle/star deltas also honour the `CYCLE_DELTA_MAX_DEGREE`
> and `STAR_CENTER_MAX_DEGREE` hub guards.

---

## Reduced-signature adapters (`_adapters.py`)

The reduced blocks store distribution *parameters*, not the raw moments the generator originally
read. These helpers reconstruct what Stage 1/2 need (NaN-safe; NaN → neutral fallback):

- `_quantile_mean(fit)` — mean of a quantile-function fit via `np.trapezoid` (e.g. `cs_size_mean`).
- `_functionality_from_alpha(fit)` — `1/ζ(α)` = fraction of single-valued slots, from the
  median of a multiplicity-α quantile function (out-side `mean_functionality` fallback).
- `_reconstruct_singular_values(exp_fit, k)` — `scale·exp(−rate·k)` spectrum for `P(r|t)`.
- `sample_quantiles_trunc(fit, n, rng)` — inverse-transform draws (`np.interp`) naturally clipped
  to `[q@0, q@1]`; `None` when the fit is NaN.
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
| `subj/obj_row_entropy_q` | Indirectly: group prototype concentration |

`per_type_entropy_exp` and class sizes now emerge from post-hoc type assignment (no longer
directly controlled).

---

## Path-length steering

### Why mean, not loc/scale/shape

`shortest_path_skew` stores a skew-normal fit with five parameters. The full shape
(loc/scale/shape) is emergent — it is determined by density (Block A) and degree structure
(Block B). Trying to steer all five parameters independently would require expensive BFS-in-the-
wiring-loop feedback and would conflict with the degree/CS targets already set.

Instead, the generator targets only two derived scalars:
- **`path_hi_target`** (`int(skew.hi)`) — the observed maximum path length (diameter cap).
  Controllable directly via shortcuts.
- **`path_mean_target`** (`_skewnorm_mean(skew)`) — the mean of the untruncated skew-normal.
  Partially controllable by adding hub-to-hub shortcuts that compress long paths.

`loc`, `scale`, and `shape` emerge from density + degree structure and are validated in the
roundtrip report but not explicitly steered.

### Mechanism

`_steer_path_lengths` runs after `_connect_components` (step 7b) so BFS distances are
well-defined on the connected entity subgraph. It builds a **temporary undirected igraph object**
from `content_edges` so igraph's C-backend `diameter()` and `distances()` can be used without
switching to the final graph representation early.

Each round:
1. Estimate current diameter (`ig.diameter()`) and mean (sampled BFS from 50 random sources).
2. If `diameter > hi_target`: add `⌈(diam − hi_target)/2⌉` shortcuts; each connects a random
   source to its farthest reachable node (different source each time → parallel coverage).
3. If `mean > mean_target + 0.5`: add `max(1, round(√V · (mean − mean_target) / mean))`
   shortcuts between nodes sampled with probability ∝ degree. Hub-to-hub links act as
   long-range relays that globally compress path lengths.
4. Shortcuts are added to both the igraph object and `content_edges`; the next round
   re-estimates on the updated graph without rebuilding.

### Limitations

- **One-sided:** shortcuts can only reduce mean/diameter, not increase them. If the synthetic
  graph has shorter paths than the target (unusual — typical issue is too-long paths), no
  correction is possible without removing edges.
- **Heuristic count for mean:** `√V · relative_overshoot` is not derived analytically; it is
  a coarse estimate. Multiple rounds (up to 4) course-correct.
- **loc/scale/shape remain unsteered:** only mean and hi are targeted. The skew of the
  path-length distribution (shape parameter) is emergent and will track the degree structure.

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
  templates; path template entropy for k≥4 (cost grows as O(Δ^(k-1)) per swap — prohibitive for
  large k).

---

## Verification

```bash
.venv/bin/python -m pytest tests/test_generator_stage1.py tests/test_generator_stage2.py -q
.venv/bin/python scripts/signature_roundtrip.py <graph> [--rewire-budget N]
```
The round-trip loads a cached target signature, generates, re-measures, and prints a per-block
comparison (median relative error is the meaningful aggregate; mean/max are inflated by
near-zero-target features). The synthetic re-measurement runs Block E at a reduced
`_FINAL_SAMPLE_BUDGET = 20_000` (CC motif/star sampling + path/tree walks) instead of the 100k
Block-E default, to keep the round-trip fast — the cached target side is unaffected.

The re-measured synthetic signature is also dumped to a `signature_synth/` directory next to the
source graph (`data/graphs/<name>/signature_synth/`, `data/test_graphs/<name>/signature_synth/`),
using the same layout as the measured `signature/` dir — per-block `block_<x>.png` plots and
`block_<x>.json` state, plus `summary.txt` and combined `signature.json`. This mirrors what
`measure_signature.py` writes for real graphs (both now share
`signature.write_signature_outputs`), so a measured and a generated signature are structurally
identical and directly comparable for validation.

Pass `--convergence-log` with no value to record the Stage 3 convergence CSV; the file is
auto-named from the graph name and run options (e.g. `conv_<graph>_seed42_rb5000.csv`) and written
to `experiments/convergence_logs/`. An explicit `--convergence-log <path>` overrides both the name
and location. Plot the result with `scripts/convergence_plot.py`.
