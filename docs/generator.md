# The Generator (`kgsynth`) — algorithms, step by step

The generator turns a measured **reduced signature** into a synthetic KG whose re-measured
signature lands near the target. It implements the project brief's three-stage procedure
([notes/generation_algorithm_fit.md](notes/generation_algorithm_fit.md),
[archive/generation_implementation_plan.md](archive/generation_implementation_plan.md)) against the
reduced blocks ([signature.md](signature.md)).

Code lives in the **`src/kgsynth/generator/`** package:

| Module | Role |
|---|---|
| `pipeline.py` | `Signature` (target) + `Generator.sample()` orchestrator |
| `schema.py` | `Schema` dataclass — Stage-1 output handed to Stage 2 |
| `stage1.py` | `sample_schema` — abstract schema from BlockA/B/C/D |
| `stage2.py` | `instantiate` — CS-first graph wiring |
| `stage3.py` | `refine` — Maslov–Sneppen rewiring + simulated annealing |
| `_adapters.py` | reduced-signature reconstructions (see below) |
| `_logging.py` | package logger (`generator.*`, INFO progress lines) |

Public API (re-exported from `__init__.py`):
```python
from kgsynth import Signature, Generator
g = Generator(Signature.from_file("target.ttl")).sample(seed=42, rewire_budget=5000)
```
`Signature` holds reduced blocks `a, c, e` (required) and `b, d, f` (optional — each enables
more faithful structure). `sample()` derives sub-seeds so the whole pipeline is reproducible
from one integer: Stage 1 `seed`, Stage 2 `seed+1`, Stage 3 `seed+2`.

A target signature can also come from a YAML file — `Signature.from_config(path)` and its inverse
`sig.to_config(path)` — instead of measuring a graph. The file holds one top-level key per block
letter (`a`..`f`), each mapping to that block's `to_serializable()` state (the same shape the
tracked corpus's `block_*.json` files use, just YAML instead of JSON — PyYAML round-trips `NaN`
natively, unlike stdlib `json`). `a`, `c`, `e` are required; `b`, `d`, `f` default to `None` if
absent. This is the backing for `kgsynth generate --config <file>`, and for hand-editing or
versioning a target signature independent of any single measured graph.

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
| F | `degree_assortativity` | Stage-3 target |
| F | `num_components`, `largest_component_fraction` | Stage-2 connectivity target (nc, LCC fraction) |

Block F's `shortest_path_mean` / `shortest_path_max` are measured but **not** targeted (see
[§ Path-length steering](#path-length-steering)).

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
   (`_sample_type_relation_probs`). This uses `P(r|t)`'s **own** T×R spectrum, kept separate
   from the `M` co-occurrence spectrum. No types (`T=0`) → empty `(0,R)` table.
5. **CS structure (Block D).** `cs_num_templates = num_distinct_cs`;
   `cs_template_zipf = cs_freq_fit.alpha` (CS reuse skew) and
   `cs_template_vmax = cs_freq_fit.v_max` (reuse-draw truncation — the fitted α covers the
   full bounded range, so unbounded draws would over-skew); `cs_size_q` passed through; the
   CS-size quantile-function mean (trapezoid integral) gates whether template mode is enabled.
6. **Co-occurrence group prototypes (Block C).** When `subj_cooc_exp` / `obj_cooc_exp` have a
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
   solved so the overall mean matches `E/V`. The whole distribution (body, p90,
   max) is targeted, not just a single hard bound. When the p90/max scalars are
   absent (Block B not measured) the targets are `None` and Stage 2 wires without
   degree steering.

---

## Stage 2 — CS-first instantiation (`instantiate`)

Builds the graph by sampling characteristic sets first, then wiring edges per relation. This is
where most of the structural fidelity is established.

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
   - Build one template pool per group, sized ∝ group weight via `_allocate_quotas` (largest-remainder
     method: distributes `cs_num_templates` slots exactly, with floor 1 per group — a naive
     `max(1, round(w_g · total))` would inflate the total when many groups have small weights).
     Applied to both forward (`_fwd_quotas`) and inverse (`_inv_quotas`) pools.
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
5d. **Pool overlap for reciprocal relations** (when `relation_reciprocity` is set):
   real graphs pack directed content edges onto **shared** node pairs (parallel/multi-relational
   overlap and bidirectional pairs), whereas the CS-first construction above assigns forward and
   inverse CS **independently**, so `S_r ∩ O_r` (entities eligible to both emit *and* receive `r`)
   is tiny — even for a relation Stage 1 marked symmetric (`ρ_r≈1`). This pass adds `r` to a `ρ_r`
   fraction of `r`'s emitters' inverse CS (swapping out one existing entry so inverse-CS *size* —
   and the `inv_cs_size_q` / degree-rank-matching above — is unaffected; only *which* relations an
   entity receives changes), enlarging `S_r ∩ O_r` before the wiring loop below needs it. See
   `docs/notes/motif_reachability_and_edge_multiplicity.md` and
   `docs/notes/relation_reciprocity_and_bidirectionality.md` for the diagnosis.
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
     inv_cs_size^a_subj` (subject-multiplicity tail × quota/expected-degree × **G2b in-side
     offset**); allocate by `multinomial`, then **cap at `|S_r|`** + redistribute, plus the
     per-object hard quota in capacity mode. When no degree targets exist, no degree factor
     is applied.
   - **Stub reservation** (when `ρ_r > 0`): even with `S_r ∩ O_r` enlarged by 5d, the out-side and
     in-side multinomials above are independent draws, so an entity eligible for both roles rarely
     gets a stub on *both* sides by chance. For up to `round(ρ_r·edges_r/2)` entities in
     `S_r ∩ O_r`, force their out-stub count and in-stub count to ≥1 each — stealing one stub from
     the current max-count entity on the respective side, so the edge budget is untouched.
   - **Pair** subject-stubs with object-stubs within `S_r × O_r` (configuration model); on a
     self-loop or duplicate `(s,o)` **retry** by swapping in another pending object stub. Within
     this pairing:
     - **Mutual-pair construction** (reciprocal relations): draws entities *with replacement* from
       `S_r ∩ O_r` (an entity stays available across multiple mutual pairs until either its
       out-stub or in-stub supply is exhausted) and places `e1→e2` + `e2→e1`, up to the reserved
       target — this is what actually realises bidirectional pairs.
     - **Multi-relational biasing** (`edge_multiplicity`/`bidirectional_ratio` targets, independent
       of per-relation reciprocity): the default draw first tries an object the subject already
       links to (parallel overlap) before falling back to the unbiased reservoir.
   Attainment is **capped by the entity pool and average stub multiplicity**, not just the
   mechanism: e.g. on wn18rr_v4 the biggest relation needs ~3.4 stubs/entity on average from its
   shared pool to hit its reciprocity target but only has ~2.85 available, so a real shortfall
   remains even with reservation + full stub reuse (measured: bidirectional-pair attainment ≈45–50%
   of target across fb237/wn18rr/aids, up from the ≈20–25% opportunistic baseline before 5d/reservation).
6b. **Inv-CS template completion** (step 4b, when `entity_inv_cs` is assigned): after the main
   wiring loop, detects object nodes whose actual in-predicate set is a strict subset of their
   assigned inverse-CS template. For each missing predicate `r`, finds an existing edge
   `(s', o', r)` where `o'` already receives `r` from ≥2 edges (so removing one doesn't break
   its template), and **redirects** it to `(s', o, r)`. No net edge-count change. This enforces
   that every template predicate is realised on each object, reducing `inv_num_distinct_cs`
   from ~63 to ~32 (target) before Stage 3. Stage 3 degree-preserving swaps partially undo this
   (ending around 55), but Stage 3 initial loss improves (49.6 vs 56.2 without this pass).
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

All per-swap incremental delta helpers live in `generator/local_updates.py`; they operate on the
live adjacency dict (not an `igraph.Graph`) so each swap is cheap.

- **Triangle count** — exact, incremental `_triangle_node_delta` (O(degree) per swap); targeted
  swaps preferentially close open wedges when below target.
- **CC_avg** (avg local clustering) — exact, incremental; the `C(k_v,2)` denominators are invariant
  under degree-preserving swaps, so only per-node `Δt_v` (from the triangle delta) drives it.
- **4-node motifs** (C4, diamond, K4, paw) — exact, incremental `_motif4_delta`: enumerates the
  motif 4-sets touching the four swap endpoints before and after, and diffs per-type counts
  (O(Δ³) per swap). Guarded by `MOTIF4_DELTA_MAX_DEGREE` on the **max endpoint degree** — unlike
  the cycle DFS, motif4 candidates come from the endpoint neighbourhoods `N(a)∪N(b)`, so the
  endpoint degree bounds the cost exactly (no interior explosion). Swaps above the guard skip
  the delta and carry the motif4 counts over (loss terms cancel in the accept test); `refine()`
  logs the computed/dropped tally. Profiled on fb237_v4: unguarded ~57 ms mean per proposal
  (1.2 s max on hub swaps) — the dominant per-swap cost once the cycle guard is active; at
  guard 200, 88 % of proposals still steer motif4 with a ~47 ms worst case.
- **5-/6-cycles** — these are **induced (chordless)** cycles (degree sequences `(2,2,2,2,2)` /
  `(2,2,2,2,2,2)`), matching the motif counters. Exact, incremental `_cycle_delta`: counts induced
  cycles through the swap endpoints before and after and diffs them (O(Δ⁴)/O(Δ⁵) per swap — best
  left off for hub-heavy graphs). A swap changes the count both by adding/removing a cycle edge
  **and** by adding a chord (destroys an induced cycle) or removing one (can create an induced
  cycle). Only active when a Block-E cycle target is > 0.
  The per-pair arc enumeration behind the delta is selected by the module-level
  `_cycles_through_pair` switch in `local_updates.py`. The default is
  `_induced_cycles_through_pair_mitm`, an **anchored meet-in-the-middle** scan that generates each
  arc's interior vertices by intersecting the two endpoints' neighbourhoods (C-speed `set` ops)
  instead of a recursive induced-path DFS; it returns identical cycle sets (parity-tested against
  the recursive DFS `_induced_cycles_through_pair` and the brute-force oracle in
  `tests/_brute_motifs.py`) and is **~2.1–3.2× faster** on random neighbourhoods (measured: sparse
  n20/p.15 2.1×, medium n30/p.25 2.4–2.8×, dense n40/p.40 2.6–3.2×, very dense n50/p.55 3.1×). The
  win is a **constant factor** — both remain O(Δ^(k-2)) worst case, so the degree guard below is
  still required on hub-heavy graphs. Point `_cycles_through_pair` at
  `_induced_cycles_through_pair` to fall back to the DFS.
  The delta is guarded by `CYCLE_DELTA_MAX_DEGREE` (a tuning constant at the top of `stage3.py`),
  applied to **every node the
  enumerator is about to branch over** — swap endpoints and arc-adjacent interiors alike (the MITM
  scan branches over fewer interior nodes than the DFS, so at a given guard it drops *slightly*
  less often; both still return exact counts whenever they compute). On the first
  node whose simple degree exceeds the guard, the whole delta for that swap is dropped
  (`_cycle_delta` returns `None`) and the cycle counts carry over unchanged, so the cycle loss
  terms cancel in the accept test and neither favour nor penalise the swap. Interior guarding
  matters because endpoint filtering alone is not enough: profiling on fb237_v4
  (`experiments/stage3_delta_profiling/`, `scripts/profile_stage3_deltas.py`) shows that with the
  guard off the 6-cycle delta alone is ≥ 94 % of per-swap delta time (median ≥ 2.6 s per
  proposal), and the DFS branches through high-degree *interior* vertices — even swaps whose
  endpoints all have degree < 50 average ~2 s. The tradeoff is that most searches encounter
  *some* node above the guard, so most swaps drop the delta and cycle steering is largely frozen —
  measured at guard 20: 100 % of fb237_v4 proposals dropped, and even sparse wn18rr_v4 (where the
  unguarded delta costs ~1 ms) drops 92 %. Speed over c5/c6 fidelity; raise the guard (or set
  `float("inf")`) on graphs whose unguarded delta is already cheap.
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
from a single `_error_terms(state)` helper, so each term is defined once: the loss is the weighted
sum of the unsigned magnitudes, the log is the unweighted dump of the **signed** errors
(`_error_terms(state, signed=True)`, i.e. `(current − target) / |target|`, negative = under target,
positive = over) so the direction of each miss is visible against the 0 reference line. The best
graph seen is returned, then components are re-bridged. When a `convergence_log` is given, each
active term writes a relative-error column every `CONVERGENCE_LOG_INTERVAL` *proposals* (accepted or
rejected — the `step` column is the proposal index and forms the plot x-axis; `accepted` is the
accepted-swap count so far): `tri_err`,
motif4 `*_err`, `c5/c6_err`, `cc_err`, `assort_err`, `tree_entropy_err`, `path_entropy_k3_err`, and
`star_k{k}_err` per tracked k). Setting `CONVERGENCE_LOG_GLOBAL_REMEASURE = True` additionally
re-measures the full graph each logged row and appends ground-truth `sig_*_err` columns (validates
the incremental deltas; expensive).

**Adaptive weights** (`adaptive_weights=True`, CLI `--adaptive-weights`). By default each term's
weight in the loss sum is the fixed `LOSS_WEIGHT_*` constant. In adaptive mode, `_term_weights` is
instead rescaled every accepted swap as `weight = base_weight * ADAPTIVE_WEIGHT_SCALE * error`
(linear, with a high fixed multiplier; `ADAPTIVE_WEIGHT_SCALE = 50.0`, tuned at the top of
`stage3.py`) from the *pre-swap* `current` state (never from the candidate being scored, so a swap
can't move its own weight mid-evaluation) — a term already at target drops toward weight 0, a term
still far off is pushed harder, proportionally to its error. This turns the loss from a
weighted-L1 sum (`Σ base_i·err_i`) into a weighted-L2-like sum
(`Σ base_i·ADAPTIVE_WEIGHT_SCALE·err_i²`) with an amplified overall scale, so loss magnitudes
between adaptive and fixed runs on the same graph are not directly comparable. When a
`convergence_log` is given in adaptive mode, one extra `weight_<name>`
column per active term is written alongside the `<name>_err` column so the weight trajectory can be
plotted with `scripts/convergence_plot.py --features weight_tri weight_cc ...`.

When a `swap_log` is given, `refine()` additionally writes one CSV row per *evaluated* swap
proposal (proposals discarded by the self-loop guard produce no row): context columns `step`,
`targeted`, `deg_s1`/`deg_o1`/`deg_s2`/`deg_o2`/`deg_max4` (pre-swap simple degrees), the per-motif
deltas `d_tri`, `d_c4`/`d_diamond`/`d_k4`/`d_paw` (only for active motif4 targets) and
`d_c5`/`d_c6` (only when steered), plus `d_loss` and `accepted`. Delta cells are left **empty**
when a degree guard dropped that delta (guards are respected, never force-computed for the log).
The log exists to measure per-swap motif *leverage* — how many proposals move each motif and by
how much, and whether the leverage concentrates in hub swaps — to assess whether an approximate
hub delta would be viable. Analyse with `scripts/swap_delta_viz.py`.

**Checkpoints** (`checkpoint_steps`/`checkpoint_callback`, threaded through
`Generator.sample(checkpoint_steps=…, checkpoint_callback=…)`). For tracing how the graph evolves
through the annealing walk rather than only inspecting the final output: `checkpoint_callback(step,
graph)` fires once per step in `checkpoint_steps` (sorted ascending) with an independent
`igraph.Graph` snapshot of the walk's *current* state at that point — step `0` is the pre-loop,
post-Stage-2 graph, before any swap is attempted. A step at or beyond where the loop actually
stopped (the requested `budget`, or earlier on a manual escape) fires with the same graph `refine()`
returns, so a trajectory's last point always matches the caller's final output. Both the checkpoint
snapshots and the final output share one `_materialize_graph` helper, so they're structurally
identical. Used by `scripts/signature_pca_trajectory.py` to plot a path through PCA space from the
post-Stage-2 graph toward the target as rewiring progresses.

> **Runtime note.** The incremental deltas are O(Δᵏ⁻¹) per *attempted* swap, so they dominate
> Stage-3 cost on hub-heavy graphs. Lower `--rewire-budget` or disable a steering term (set its
> `LOSS_WEIGHT_*` to 0). The two costliest deltas honour degree guards and log their
> computed/dropped tallies: the 5/6-cycle update via `CYCLE_DELTA_MAX_DEGREE` (node-level —
> checks every DFS-expanded node and drops the delta on the first hub encountered) and the
> 4-node motif update via `MOTIF4_DELTA_MAX_DEGREE` (endpoint-level, which is exact for its
> cost).
> Measured magnitudes (`experiments/stage3_delta_profiling/summary.md`): with the cycle guard off,
> fb237_v4's Stage-2 graph (deg max ≈ 1 383, mean ≈ 14) averages ≥ 2.9 s of delta work per
> proposal — ≥ 4 h for a 5 000-swap budget — while wn18rr_v4 (deg max 73, mean ≈ 5) averages
> ~1 ms, a ≳ 2 700× gap. The node-level guard bounds the cycle work to milliseconds per proposal,
> at the cost of freezing cycle steering on the (many) dropped proposals of dense graphs.

---

## Reduced-signature adapters (`_adapters.py`)

The reduced blocks store distribution *parameters*, not raw moments. These helpers reconstruct
the concrete quantities Stage 1/2 need (NaN-safe; NaN → neutral fallback):

- `_quantile_mean(fit)` — mean of a quantile-function fit via `np.trapezoid` (e.g. `cs_size_mean`).
- `_functionality_from_alpha(fit)` — `1/ζ(α)` = fraction of single-valued slots, from the
  median of a multiplicity-α quantile function (out-side `mean_functionality` fallback).
- `_reconstruct_singular_values(exp_fit, k)` — `scale·exp(−rate·k)` spectrum for `P(r|t)`.
- `sample_quantiles_trunc(fit, n, rng)` — inverse-transform draws (`np.interp`) naturally clipped
  to `[q@0, q@1]`; `None` when the fit is NaN.
- `sample_powerlaw(alpha, n, rng)` — `power-law(α>1)` draws via inverse-CDF; uniform ones when α
  is NaN/≤1 (the neutral fallback).

---

## Design notes — why the wiring is what it is

The load-bearing design choices behind the Stage-1/2 wiring:

- **`P(r|t)` from its own spectrum.** Stage 1 reconstructs `P(r|t)` from `type_rel_spectrum_exp`
  (the T×R spectrum), kept separate from the `M` co-occurrence spectrum — they are different
  quantities, so one cannot target the other.
- **Measured relation Zipf.** Relation weights use the measured `relation_zipf` exponent, not a
  fixed value.
- **Per-relation multiplicity + edge conservation.** Out-side wiring allocates each relation's
  edges by `multinomial` on per-relation α (`obj_alpha`) and the G2b `a_obj` CS-size offset,
  hitting each relation's edge budget `|edges_r| = freq(r)·E` exactly. In-side allocation draws
  per-relation **subject**-multiplicity from `subj_alpha` × preferential attachment.
- **Realizability cap + redistribute.** With `α_subj<2` (infinite mean) or superlinear PA
  (`in_pa>1`), the in-side `multinomial` would concentrate nearly all of `|edges_r|` on one
  object, leaving unplaceable duplicates and collapsing the edge budget. Each object is capped at
  `|S_r|` (the out-side symmetrically at `|O_r|`) and the overflow redistributed, so the realised
  edge count stays near the budget.
- **CS templating for `num_distinct_cs`.** Reaching the target distinct-CS count needs three
  floors working together: (a) distinct templates drawn by rejection + size-escape, (b) a floored
  entity→template assignment (≥1 entity per template) with a power-law reuse tail, and (c) an
  out-side floor (every subject of `r` gets ≥1 edge). Without them a too-steep rank-Zipf leaves
  most templates unused and out-side multinomial zeros shrink the realised CSs.
- **Symmetric inverse CS + `a_subj`.** Block D's `inv_num_distinct_cs` / `inv_cs_freq` and the
  inverse-CS templates (object side) let edges be matched within `S_r × O_r`, and the in-side
  weight gains the `inv_cs_size^a_subj` G2b offset — the object-side mirror of `a_obj`.
- **Tuning constants** for each stage are module-level at the top of
  `stage1.py` / `stage2.py` / `stage3.py`.

---

## Co-occurrence groups

### Why

`subj_cooc_exp` is the exp-decay fit of the V-normalised singular spectrum of M_subj (the R×R
relation co-occurrence matrix, where M_subj[r1,r2] = # entities using both r1 and r2). Without
explicit targeting, M_subj's spectrum, density, and row-entropy distribution are not reproduced.

Entity relation diversity is driven by the co-occurrence groups (below), not by `P(r|t)`:
`P(r|t)`'s spectrum (driven by `type_rel_spectrum_exp`, the T×R matrix) is a different quantity
from M_subj's spectrum, so it cannot be used to target M_subj's spectrum, density, or row entropy.

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

`per_type_entropy_exp` and class sizes emerge from the post-hoc type assignment — they are not
directly controlled.

---

## Path-length steering

**Block F's `shortest_path_mean` / `shortest_path_max` / `shortest_path_var` are measured but not
steered.** They are emergent — determined by density (Block A) and degree structure (Block B) —
and are reported by the roundtrip comparison as validation-only quantities.

Path lengths are not targeted because the only cheap post-hoc lever, injecting hub shortcuts, is
*one-sided*: shortcuts can shorten paths but never lengthen them, and the synthetic graph almost
always undershoots (its paths are already too short), so the lever has nothing to do. The
structural-undershoot analysis and a sketch for moving path steering into Stage 3's annealing loop
(if it is ever wanted) are in [archive/path_length_steering.md](archive/path_length_steering.md).

---

## Known limitations / open items

- **Block F path lengths are unsteered.** `shortest_path_mean` / `shortest_path_max` are
  measured and reported but no target drives them (see
  [§ Path-length steering](#path-length-steering)).
- **Co-occurrence density and row-entropy not analytically pinned.** These emerge from
  group prototype concentration and CS size but have no independent control knob. Remaining
  deviations require either softmax temperature tuning per group or Stage-3 obj-side steering.
- **Inverse `num_distinct_cs` after Stage 3.** The Stage 2 inv-CS template completion (step 6b)
  achieves the target `inv_num_distinct_cs` (e.g. 32) before Stage 3. Stage 3's degree-preserving
  swaps undo this (~55 after 200k swaps for WN18RR) because swaps that change in-edge endpoints
  freely modify each node's in-predicate set. Adding an inv-CS loss term to Stage 3 would require
  O(1) per-swap updates to a `frozenset` index — non-trivial but feasible. Until then, the path
  and tree template entropy errors remain ~47–60%.
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

Every auto-named output of a round-trip run is stamped with a single per-run timestamp
(`YYYYmmdd_HHMMSS`) so repeated runs — even with identical options — don't overwrite each other:
the synthetic graph (`<graph>_synth_<timestamp>.ttl`), the synthetic signature directory
(`signature_synth_<timestamp>/`), and the auto-named convergence / swap logs. An explicit `--out`
is used verbatim.

The re-measured synthetic signature is also dumped to a `signature_synth_<timestamp>/` directory
next to the source graph (`data/graphs/<name>/`, `data/test_graphs/<name>/`),
using the same layout as the measured `signature/` dir — per-block `block_<x>.png` plots and
`block_<x>.json` state, plus `summary.txt` and combined `signature.json`, via the shared
`signature.write_signature_outputs` helper. This mirrors what `measure_signature.py` writes for
real graphs (its own inline write logic produces the same on-disk layout), so a measured and a
generated signature are structurally identical and directly comparable for validation.

Pass `--convergence-log` with no value to record the Stage 3 convergence CSV; the file is
auto-named from the graph name and run options (e.g. `conv_<graph>_seed42_rb5000.csv`) and written
to `experiments/convergence_logs/`. An explicit `--convergence-log <path>` overrides both the name
and location. Plot the result with `scripts/convergence_plot.py`.

`--swap-log` works the same way for the Stage 3 swap-proposal log (one row per evaluated proposal
with per-motif deltas, Δloss and the accept decision): no value auto-names
`swaps_<graph>_seed<seed>_rb<budget>.csv` into `experiments/swap_delta_logs/`; an explicit path
overrides. Plot with `scripts/swap_delta_viz.py`.

Pass `--adaptive-weights` to switch Stage 3 to error-scaled loss weights (see "Adaptive weights"
above). It appends an `adaptive` token to the auto-named convergence/swap log filenames — e.g.
`conv_<graph>_seed42_rb5000_adaptive_<timestamp>.csv` — so a fixed-weight and an adaptive-weight run
of the same graph/seed/budget land as sibling files in `experiments/convergence_logs/` instead of
overwriting each other, ready to compare with `scripts/convergence_plot.py conv_..._TS1.csv
conv_..._adaptive_TS2.csv`.

`scripts/sweep_adaptive_weight_scale.py <graph>` finds the `ADAPTIVE_WEIGHT_SCALE` that minimises
the accumulated *unweighted* error across all active motifs/metrics after a fixed rewire budget
(default 100 000). Stage 1/2 run once to build a fixed pre-Stage-3 graph; each candidate scale then
runs Stage 3 from a fresh copy of that same graph via a monkey-patch of
`generator.stage3.ADAPTIVE_WEIGHT_SCALE`, so only the weighting scheme varies between candidates.
The comparison metric is `stage3_best_unweighted_error_sum` — `Σ|error|` over `_error_terms` at the
best snapshot, read straight off the graph attribute `refine()` now always sets — not
`stage3_best_loss`, since the loss itself is scaled by the candidate under test and isn't
comparable across scales. Pass `--scales`, `--rewire-budget`, `--seed`, `--out <csv>` to customise.
