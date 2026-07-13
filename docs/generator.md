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

`../_logging.py` (i.e. `src/kgsynth/_logging.py`, one level up from this package) supplies the
shared logger (`signature.*`/`generator.*`/`motif_counter.*`, INFO progress lines) — it is not
generator-specific.

Public API (re-exported from `__init__.py`):
```python
from kgsynth import Signature, Generator
g = Generator(Signature.from_file("target.ttl")).sample(seed=42, rewire_budget=5000)
```
`Signature` holds reduced blocks `a, b, c, d, f` (required — a real graph always measures them,
see [§ Target signature must be complete](#target-signature-must-be-complete)) and `e` (nullable —
`kgsynth.corpus.load_target_from_corpus(with_block_e=False)` legitimately skips it for callers that
only drive Stages 1-2, since it is the expensive block to measure and neither stage reads it).
`sample()` derives sub-seeds so the whole pipeline is reproducible from one integer: Stage 1 `seed`,
Stage 2 `seed+1`, Stage 3 `seed+2`.

A target signature can also come from a YAML file — `Signature.from_config(path)` and its inverse
`sig.to_config(path)` — instead of measuring a graph. The file holds one top-level key per block
letter (`a`..`f`), each mapping to that block's `to_serializable()` state (the same shape the
tracked corpus's `block_*.json` files use, just YAML instead of JSON — PyYAML round-trips `NaN`
natively, unlike stdlib `json`). All six blocks are required — a hand-edited config describes a
complete target signature, matching what any real graph measures. This is the backing for
`kgsynth generate --config <file>`, and for hand-editing or versioning a target signature
independent of any single measured graph.

### The flat feature dict: `as_features()` / `from_features()`

A third route in and out of a `Signature` — the **flat 127-key feature dict**, the same
`{name: value}` mapping stored under `"features"` in a measured `signature.json`:

```python
feats = sig.as_features()          # 127 public feature names -> float
feats["mean_degree"] *= 1.2
Generator(Signature.from_features(feats)).sample(seed=1)
```

Unlike `from_config`, which restores each block's full serialized state, `from_features` rebuilds the
blocks from their *named features alone*. That is sufficient because the generator only ever reads
values the feature vector carries: it dereferences `.alpha` off a six-field `PowerLawStats`, and
`out_degree_alpha` is a feature; a `QuantileFit` **is** its seven `*_q00`..`*_q100` features; and so
on for `ZipfFit`, `ExpDecayFit`, `TruncPowerLawFit` and the six `recip_symmetric_frac_bin*` bins. A
signature rebuilt this way produces a **bit-identical graph** — same `Schema`, same edge list at a
fixed seed (`tests/test_signature_from_features.py`).

What it does **not** restore is each block's raw sample arrays — degree lists, class sizes, singular
spectra — which exist only so `block.visualize()` can overlay a fit on the data it was fit to. So a
block rebuilt from features **cannot plot itself**, and must not be re-serialized with
`to_serializable()`: that would emit a file that looks measured but is missing everything the vector
does not carry. Persist the feature dict instead.

This is what lets a caller treat a signature as a **point in feature space** — perturb it, sample a
new one (`kgsynth.signature_sampler`), interpolate — and still generate from the result.

---

## Inputs — which signature fields drive generation

| Block | Field | Used for |
|---|---|---|
| A | `num_entities`, `num_relations`, `mean_degree`, `type_edge_frac` | `V`, `R`, edge budget `E = round(mean_degree·V)`, and its split into `n_type_edges = round(E·type_edge_frac)` + `content_E` |
| B | `relation_zipf` | relation-frequency weights (Zipf exponent) |
| B | `obj_alpha_q` | per-relation **object**-multiplicity tail α (out-degree shape), as a quantile function |
| B | `subj_alpha_q` | per-relation **subject**-multiplicity tail α (in-side shape), as a quantile function |
| B | `obj_mult_max`, `subj_mult_max` | upper bounds of those two multiplicity laws (the range their α was fitted over) |
| B | `a_obj` | G2b forward CS-size→multiplicity offset (`cs_size^a_obj`) |
| B | `a_subj` | G2b inverse CS-size→multiplicity offset (`inv_cs_size^a_subj`) |
| B | `in_degree_fit.alpha` | tail-shape input to the Stage-1-sampled target in-degree sequence (`_adapters.sample_degree_sequence`) |
| C | `num_classes`, `class_size_fit.alpha` | type count + type-size weights |
| C | `type_rel_spectrum_exp` | `P(r\|t)` low-rank reconstruction (used for post-hoc type scoring) |
| C | `subj_cooc_exp` | forward co-occurrence group prototypes (CS source) |
| C | `obj_cooc_exp` | inverse co-occurrence group prototypes (inverse-CS source) |
| D | `cs_size_q`, `num_distinct_cs`, `cs_freq_fit.alpha`/`.v_min`/`.v_max` | forward CS templates: size (quantile function), count, reuse skew + the reuse law's support |
| D | `inv_cs_size_q`, `inv_num_distinct_cs`, `inv_cs_freq_fit.alpha`/`.v_min`/`.v_max` | inverse CS templates (object side): size (quantile function), count, reuse skew + support |
| E | motif counts | Stage-3 targets (triangles, 4-cycle, diamond, k4, tailed) |
| F | `degree_assortativity` | Stage-3 target |
| F | `clustering_coefficient` | Stage-3 target (`CC_avg` loss term) |
| F | `num_components`, `largest_component_fraction` | Stage-2 connectivity target (nc, LCC fraction) |

Block F's `shortest_path_mean` / `shortest_path_max` are measured but **not** targeted (see
[§ Path-length steering](#path-length-steering)).

**Validation-only (measured, deliberately *not* used constructively):** C `subj/obj_cooc_density`,
`subj/obj_row_entropy_q`, `per_type_entropy_exp`; D `two_step_fit`; F
`shortest_path_var` (emergent — not steered). These are diagnostics
the brief marks "get near," not directly steered. (`clustering_coefficient` is *not* on this list
— Stage 3 steers it directly as the `CC_avg` loss term with an exact incremental delta; see
Stage 3 below.) Tuning constants for each stage
are module-level at the top of `stage1.py`/`stage2.py`/`stage3.py`.

---

## Target signature must be complete

Blocks A, B, C, D and F are **required** on `Signature` and `sample_schema` — a real graph always
measures them, so there is no degraded-mode ("Block absent") code path for Stage 1/2 to fall back
to. This was not always true: earlier versions carried a full second implementation for
"Block D absent" (per-entity/legacy CS sampling), "Block B absent" (no degree steering, no
functionality estimate), and "Block F absent" (assume fully connected) — dead weight, since no
production caller ever actually omitted them. That degraded-mode code has been deleted; see
`CHANGELOG.md` for the corpus-wide evidence (every NaN feature across the 9 measured KGs was
enumerated and classified).

`sample_schema` calls `_validate_target(a, b, c, d, f)` as its first step, which raises
`ValueError` naming the offending feature if any of these is NaN — quantities that are measurable
on **any** real graph, so a NaN here means the target signature is incomplete or corrupted, not a
legitimate edge case:

```
num_entities, mean_degree, num_relations,
cs_size_q, inv_cs_size_q, num_distinct_cs, inv_num_distinct_cs,
out_degree_fit.alpha, in_degree_fit.alpha,
out_degree_p90, out_degree_max, in_degree_p90, in_degree_max,
a_obj, a_subj,
subj_cooc_exp, obj_cooc_exp,
num_components, largest_component_fraction
```

A short list of features are **legitimately** allowed to be NaN, and are read directly by
`sample_schema` / `instantiate` with their own documented fallback rather than being validated
away — `_validate_target` deliberately does not check them:

| Feature | Reason | Fallback |
|---|---|---|
| `relation_zipf.exponent`, `cs_freq_fit.alpha`, `inv_cs_freq_fit.alpha` | small R (too few relations/CSs to fit a Zipf/power-law) | `DEFAULT_ZIPF_EXPONENT` |
| `obj_alpha_q[i]`, `subj_alpha_q[i]` (per-relation) | small R for that one relation | flat weights for that relation (`sample_powerlaw` returns uniform ones) |
| `class_size_fit.alpha` | small `T`, or **untyped KG** (the majority case — 7 of 9 corpus KGs) | uniform type weights |
| `type_rel_spectrum_exp` | untyped KG (no `P(r|t)` signal) | uniform per-type relation weights |
| `recip_symmetric_frac[bin]` | empty frequency bin (small R) | nearest non-empty bin |
| `recip_symmetric_value` | **no symmetric relation exists** in the target (a real measurement outcome, not missing data) | `0.9` |

Every other NaN feature in the reduced signature (`path_template_*_k*`, `class_size_*`,
`type_rel_spectrum_*`, `per_type_entropy_*`, etc.) is either a real measurement outcome the
generator was never meant to steer toward (e.g. "no path of that length exists") or is downstream
of one of the two tables above and requires no separate handling.

---

## Stage 1 — schema sampler (`sample_schema`)

Produces a `Schema`: relations, types, `P(r|t)`, and the per-relation/CS parameters Stage 2
needs. All randomness from one seeded `np.random.Generator`.

Every input this stage reads is validated up front by `_validate_target` — see
[§ Target signature must be complete](#target-signature-must-be-complete) — so the steps below
describe the single code path a real target always takes; the handful of quantities that can
legitimately still be NaN (small-R Zipf/CS fits, untyped-KG class stats) are called out inline.

1. **Counts & budget.** `R = max(1, num_relations)`, `T = max(0, num_classes)`,
   `E = round(num_entities · mean_degree)` (reduced Block A stores mean degree, not `|E|`).
2. **Relations.** `relation_weights = Zipf(R, exponent)` where the exponent is the **measured**
   `b.relation_zipf.exponent`, falling back to the `relation_zipf_exponent` param only when the
   measured exponent is NaN/non-positive (small R — too few relations to fit a Zipf curve).
3. **Types.** `type_weights = Zipf(T, class_size_fit.alpha)`, uniform fallback when α is NaN
   (untyped KGs — the majority case in the corpus — or `T` too small to fit).
4. **`P(r|t)`.** Reconstruct a singular-value spectrum from `type_rel_spectrum_exp`
   (`scale·exp(−rate·k)`) and feed it to a low-rank random factorisation
   (`_sample_type_relation_probs`). This uses `P(r|t)`'s **own** T×R spectrum, kept separate
   from the `M` co-occurrence spectrum. No types (`T=0`) → empty `(0,R)` table, and an empty
   spectrum (also an untyped-KG symptom) degenerates to uniform per-type relation weights.
5. **CS structure (Block D).** `cs_num_templates = num_distinct_cs`;
   `cs_template_zipf = cs_freq_fit.alpha` (CS reuse skew, small-R NaN fallback →
   `DEFAULT_ZIPF_EXPONENT`) and `cs_template_vmin`/`cs_template_vmax = cs_freq_fit.v_min`/`.v_max`
   — the support the reuse law was fitted over, which Stage 2 draws from directly (see the
   [truncated power-law contract](#the-truncated-power-law-contract)); `cs_size_q`
   passed through. `num_distinct_cs` is never zero on a real graph, so CS templates are always
   built — there is no per-entity fallback mode.
6. **Co-occurrence group prototypes (Block C).** `subj_cooc_exp` / `obj_cooc_exp` are never NaN
   on a real graph (`_validate_target` rejects it otherwise), so `COOC_NUM_GROUPS = 10` singular
   values are always reconstructed and fed to `_sample_type_relation_probs`, building one `(k, R)`
   group-prototype matrix per side plus a normalized weight vector from the singular values.
   Stage 2 always draws entity CSes from these prototypes (never the `P(r|t)` path) and assigns
   types post-hoc. See [§ Co-occurrence groups](#co-occurrence-groups) below.
7. **Multiplicity / degree (Block B).** `obj_alpha_q`, `subj_alpha_q`, `a_obj` and the two
   multiplicity bounds `obj_mult_max` / `subj_mult_max` passed through (per-relation NaN fits are a
   legitimate small-R outcome — Stage 2 falls back to flat weights for that one relation). Per-entity **target degree sequences** (`target_out_degrees`,
   `target_in_degrees`) are sampled purely from **signature-vector components** —
   never Block B's raw retained arrays (`_adapters.sample_degree_sequence`):
   the top 10% of nodes (the tail) draw from a power law truncated to `[p90, max]`
   whose exponent is **extreme-value matched** (`1 + ln(n_tail)/ln(max/p90)`, so
   the expected maximum of the tail draws lands on the measured max) rather than
   the fitted degree α — the global fit is too shallow for this range and would
   overshoot mid-tail mass; the remaining 90% (the body) draw from the *same*
   fitted-α power law truncated to `[1, p90]`, then repaired — up *or* down, via
   `repair_degree_sum` — so the sequence sums to exactly `content_E` (edge
   conservation). The repair is confined to the body: the tail carries p90/max and is
   never touched. The mean it targets is the **content** mean `content_E/V`, not `E/V`
   — Block B measures entity content degrees (rdf:type edges and class nodes excluded)
   and Stage 2 wires rdf:type edges outside this budget, so `E/V` would describe a
   different population than the fits do. Both sides sum to `content_E`, which is what
   a directed wiring needs (`Σ out = Σ in = E`). The whole distribution (body, p90,
   max) is targeted, not just a single hard bound; `out_degree_p90`/`max` and
   `in_degree_p90`/`max` are never NaN on a real graph, so degree steering is always active.

---

## Stage 2 — CS-first instantiation (`instantiate`)

Builds the graph by sampling characteristic sets first, then wiring edges per relation. This is
where most of the structural fidelity is established.

1. **Budget split.** `content_E = E − n_type_edges`, where `n_type_edges = round(E ·
   type_edge_frac)` (Block A), capped at `|V|` since an entity gets at most one type. Sized from the
   *measured* rdf:type share rather than assumed to be one per entity — a target graph may type only
   some of its entities. The `n_type_edges` best-scoring entities (by the CS→type score in §5b) are
   typed; the rest stay untyped and emit no `rdf:type` edge.
2. **CS size source.** Each CS's *size* is drawn from `cs_size_q` (`sample_quantiles_trunc`,
   inverse-transform of the stored quantile function) — never NaN on a real graph, so the
   quantile draw always applies (no budget-derived fallback). CS size sets **relation
   membership only** — the per-relation allocation (step 6) owns the edge budget.
3. **Entity-type map allocated** (all entities untyped, `-1`). Types are *not* an input to CS
   construction: the co-occurrence-group path (step 4) never reads a type, so types are derived
   from the realised CS post-hoc in step 5b. When `T=0` the entities simply stay untyped.
4. **Distinct CS templates (forward + inverse).** `subj_group_probs`/`obj_group_probs`
   (the co-occurrence group prototypes — [§ Co-occurrence groups](#co-occurrence-groups)) and
   `cs_num_templates`/`inv_cs_num_templates` are always populated by Stage 1, so this is the only
   CS-construction path — there is no per-type or per-entity fallback:
   - Assign each entity to a co-occurrence group drawn from the Zipf-weighted group distribution.
   - Build one template pool per group, sized ∝ group weight via `_allocate_quotas` (largest-remainder
     method: distributes `cs_num_templates` slots exactly, with floor 1 per group — a naive
     `max(1, round(w_g · total))` would inflate the total when many groups have small weights).
     Applied to both forward (`_fwd_quotas`) and inverse (`_inv_quotas`) pools.
   - Assign entities within each group to templates via `_assign_templates`.
   - (Inverse side mirrors this using `obj_group_probs`.)

5. **Entity → template assignment.** Per pool: **floor each template at ≥1 entity** (so every
   distinct CS is realised), then distribute the rest by a `power-law(reuse_zipf)` reuse tail
   with raw draws truncated at `reuse_vmax` (the measured max recurrence, mirroring the
   truncated-power-law fit) — for forward (`cs_template_zipf`/`cs_template_vmax`) and inverse
   (`inv_cs_template_zipf`/`inv_cs_template_vmax`) alike. This steers the
   distinct-CS counts *and* the reuse skews together.

   5b. **Post-hoc type assignment** (when `T>0`): once CSes are fixed, score each
   entity's CS against every type: `score(v, t) = Σ_{r ∈ CS(v)} log P(r|t)`. Assign
   `entity_type[v] = argmax_t score`. Entities with empty CSes fall back to sampling from
   `type_weights`. This makes type labels emerge from relation usage (the real causal direction)
   rather than being set independently of CS content.
5c. **Target degree assignment** (`target_out_degrees`/`target_in_degrees`, always sampled by
   Stage 1 — see Stage 1 §7): sampled target values
   are **rank-matched** to CS size (out-side; largest target → largest CS, floored at `|CS|` so the
   ≥1-edge-per-CS-relation floor stays feasible) and to inverse-CS size (in-side). Both sides are then
   repaired to the **same** quota budget `round(DEGREE_QUOTA_SLACK · content_E)` via
   `repair_degree_sum`, so `Σ tgt_out == Σ tgt_in`: a directed wiring needs matched stub counts, and
   the two sides are sampled independently. Allocation weight is ∝ remaining quota `(target − placed)⁺`
   ("capacity" weighting), plus a **hard per-node quota** via `_cap_redistribute(hard_cap=…)`.
   (An expected-degree "chunglu" alternative — weight ∝ target degree, no hard cap — was evaluated and
   rejected: tail overshoot ≈ +116% on max-out.)

   The repair is **two-sided** (it trims as well as tops up) and **skips the hub entries**, which is
   what an earlier one-directional multinomial top-up (`if shortfall > 0`, weight ∝ target) could not
   do. That top-up had two failure modes. It could not bring an *over*-budget side down — aids' in-side
   sat at ~3× `content_E`, so its quota never bound and in-degree steering was inert there — and
   because its weight was ∝ target it acted as a rescale, multiplying the hubs by the same factor and
   inflating the p90/max targets the tail was built to hit (up to 2.3× on max-out), undoing the
   extreme-value matching in `sample_degree_sequence`.

   `DEGREE_QUOTA_SLACK` (default **1.0**, i.e. no headroom) trades the deficit-recovery volume against
   degree fidelity. Slack lets the wiring loop place more edges through the main path — deficit on
   fb237_v4 falls 3229 → 1076 at slack 1.25, on aids 89413 → 23055 — but it costs more than it buys, on
   two counts:

   * a loose quota spreads edges evenly instead of feeding the hubs, so the realised max degree decays
     monotonically (fb237_v4 max-out 195 → 137 → 115 at slack 1.0 / 1.25 / 1.5, against a signature
     target of 195);
   * at high slack the main loop saturates the whole per-relation budget, and `_connect_components`
     then appends its bridging edges *on top* — so the graph **overshoots |E|** (aids at slack 1.5:
     804721 edges against a target of 802066). Only slack 1.0 leaves the main loop short enough that
     bridging + deficit recovery land the edge count exactly on budget.

   So 1.0 is not merely the best point on a tradeoff curve — it is the only value that currently
   conserves the edge count. Raising it means fixing the bridging-edge double-count first.
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
   - `|edges_r| = min(edge_budget[r], |S_r|·|O_r|)` (capacity bound), where `edge_budget` is a
     largest-remainder integer allocation of `content_E` across all present relations by
     `renorm_weight` (same technique as `_allocate_quotas` above), so the per-relation budgets
     sum to `content_E` exactly.
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
`TEMPLATE_ATTEMPT_*`) are module-level at the top of `stage2.py`.

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

  > **Current default.** `CYCLE_DELTA_MAX_DEGREE` defaults to `float("inf")`, i.e. the guard is
  > **off by default** — every swap gets the exact cycle delta computed, none dropped — the
  > opposite of the "small guard, most swaps dropped" default this section otherwise describes.
  > Pass an explicit finite `CYCLE_DELTA_MAX_DEGREE` to enable guarding on graphs where the
  > unguarded delta is too slow (see the runtime note below).
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
- **Induced k-star counts — currently disabled.** Earlier versions steered induced k-star counts
  (`k=2..5`) via an exact incremental `_star_count_delta` (still present in `local_updates.py`)
  and an optional `_targeted_star_swap` move that biased proposals toward breaking triangles among
  high-degree hubs. That path is no longer wired into `refine()`: `_star_count_delta` isn't
  imported into `stage3.py`, `_SAState` / `_error_terms` / the loss / the convergence CSV carry no
  star term at all, and `_targeted_star_swap` is dead code kept only "as a helper for future use"
  (per its own comment). Induced star counts are therefore purely emergent from the Stage-2
  out-degree distribution — not steered, not logged, by the current Stage 3.

The SA loss is a weighted sum of relative errors. Both the loss and the convergence log derive
from a single `_error_terms(state)` helper, so each term is defined once: the loss is the weighted
sum of the unsigned magnitudes, the log is the unweighted dump of the **signed** errors
(`_error_terms(state, signed=True)`, i.e. `(current − target) / |target|`, negative = under target,
positive = over) so the direction of each miss is visible against the 0 reference line. The best
graph seen is returned, then components are re-bridged. When a `convergence_log` is given, each
active term writes a relative-error column every `CONVERGENCE_LOG_INTERVAL` *proposals* (accepted or
rejected — the `step` column is the proposal index and forms the plot x-axis; `accepted` is the
accepted-swap count so far): `tri_err`,
motif4 `*_err`, `c5/c6_err`, `cc_err`, `assort_err`, `tree_entropy_err`, `path_entropy_k3_err`.
(No `star_k{k}_err` columns — induced-star steering is currently disabled, see
[Stage 3](#stage-3--refinement-refine).) Setting `CONVERGENCE_LOG_GLOBAL_REMEASURE = True` additionally
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

- `_reconstruct_singular_values(exp_fit, k)` — `scale·exp(−rate·k)` spectrum for `P(r|t)`.
- `sample_quantiles_trunc(fit, n, rng)` — inverse-transform draws (`np.interp`) naturally clipped
  to `[q@0, q@1]`; `None` when the fit is NaN.
- `sample_powerlaw_trunc(alpha, lo, hi, n, rng)` — `power-law(α)` draws **truncated to `[lo, hi]`**
  via inverse-CDF; constant `lo` (→ equal weights after normalisation, the neutral fallback) when
  α is NaN/≤1 or the range is empty. The single power-law sampler — see the contract below.
- `sample_degree_sequence(alpha, p90, d_max, mean_deg, n, rng)` — the per-entity target degree
  sequence behind Stage 1 §7: an extreme-value-matched tail on `[p90, max]` plus a fitted-α body on
  `[1, p90]`, body-repaired to sum to exactly `n · mean_deg`; `None` when `p90`/`max` are unavailable.
- `repair_degree_sum(seq, target_sum, rng, floor=…, adjustable=…)` — forces a degree sequence onto an
  exact sum. Two-sided (trims as well as tops up), and the caller marks the hub entries
  non-`adjustable` so the p90/max targets survive. Shared by `sample_degree_sequence` (Stage 1) and
  `_sample_target_degrees` (Stage 2), which is why both sides of the wiring balance.

### The truncated power-law contract

Every quantity the generator draws from a power law is **bounded**, and every power law in the
signature is fitted as a **truncated MLE over exactly those bounds** (`_fit_powerlaw` pins
`[1, max]`; `fit_truncated_powerlaw` pins `[v_min, v_max]`). So the generator draws from the
bounded law directly — `sample_powerlaw_trunc` — rather than drawing unbounded and clamping.
The distinction is not cosmetic: a clamp deposits an atom of probability mass exactly at the
bound instead of redistributing it across the support, so the realised distribution would have
a spike the target does not.

| Quantity drawn | Support | Bounds come from |
|---|---|---|
| CS-template reuse (forward / inverse) | `[v_min, v_max]` | `cs_freq_fit` / `inv_cs_freq_fit` (Block D) — the fit stores both |
| per-relation object / subject multiplicity | `[1, mult_max]` | `obj_mult_max` / `subj_mult_max` (Block B) |
| target degree — body | `[1, p90]` | `out/in_degree_p90` (Block B), fitted α |
| target degree — tail (top 10%) | `[p90, max]` | `out/in_degree_p90`, `out/in_degree_max`; exponent is extreme-value matched, **not** the fitted α (Stage 1 §7) |

The two *rank* laws — relation weights and type weights, via `_zipf_weights` — are bounded by
construction (`rank^(−α)` normalised over a finite `n`) and need no truncation.

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
  per-relation **subject**-multiplicity from `subj_alpha` × a capacity-weighting factor (remaining
  quota, from the Stage-1-sampled `target_in_degrees`). There is no
  preferential-attachment mechanism in the current code — an earlier PA-based in-degree lever was
  replaced by this target-degree-sequence approach (see Stage 1 §7 and Stage 2 §6's in-side
  wiring).
- **Realizability cap + redistribute.** With `α_subj<2` (infinite mean), the in-side
  `multinomial` would concentrate nearly all of `|edges_r|` on one object, leaving unplaceable
  duplicates and collapsing the edge budget. Each object is capped at
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
- ~~Aggregate vs per-relation in-degree~~ **(resolved).** Aggregate in-degree is now targeted
  directly: Stage 1 samples `target_in_degrees` from `in_degree_fit.alpha` + the `p90`/`max`
  scalars (`_adapters.sample_degree_sequence`), and Stage 2's capacity weighting
  enforces it as a hard per-node quota across the whole per-relation wiring
  loop (later relations see the shrinking remaining budget via the live `in_degrees` array).
  Per-relation multiplicity within one relation is separately shaped by `subj_alpha_q`. The old
  PA-based lever (`in_pa`) this bullet used to describe no longer exists in the code.
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
