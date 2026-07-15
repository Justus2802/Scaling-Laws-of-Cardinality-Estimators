# The Generator (`kgsynth`) — algorithms, step by step

The generator turns a measured **reduced signature** into a synthetic KG whose re-measured
signature lands near the target. It implements the project brief's three-stage procedure
([notes/generation_algorithm_fit.md](../developer_docs/notes/generation_algorithm_fit.md)) against
the reduced blocks ([signature.md](signature.md)).

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

A third route in and out of a `Signature` — the **flat 135-key feature dict**, the same
`{name: value}` mapping stored under `"features"` in a measured `signature.json`:

```python
feats = sig.as_features()          # 135 public feature names -> float
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
| B | `rel_freq_logq` | relation-frequency weights (quantile function of the log edge shares) |
| B | `obj_alpha_q` | per-relation **object**-multiplicity tail α (out-degree shape), as a quantile function |
| B | `subj_alpha_q` | per-relation **subject**-multiplicity tail α (in-side shape), as a quantile function |
| B | `obj_mult_max`, `subj_mult_max` | upper bounds of those two multiplicity laws (the range their α was fitted over) |
| B | `a_obj` | G2b forward CS-size→multiplicity offset (`cs_size^a_obj`) |
| B | `a_subj` | G2b inverse CS-size→multiplicity offset (`inv_cs_size^a_subj`) |
| B | `in_degree_fit.alpha` | tail-shape input to the Stage-1-sampled target in-degree sequence (`_adapters.sample_degree_sequence`) |
| B | `subject_frac`, `object_frac` | share of entities with a nonzero out- / in-degree; places the zero-degree entities (non-subjects / non-objects) |
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
| `cs_freq_fit.alpha`, `inv_cs_freq_fit.alpha` | small R (too few CSs to fit a power-law) | `DEFAULT_ZIPF_EXPONENT` |
| `rel_freq_logq` | a graph with no relations at all (never on a real KG — the fit needs only 2 relations) | `Zipf(relation_zipf_exponent)` |
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
2. **Relations.** `relation_weights` are rebuilt from Block B's `rel_freq_logq` — the quantile
   function of `log(E_r / Σ E_r)` — by evaluating it at `R` evenly-spaced levels (reconstructing the
   rank curve directly), exponentiating and renormalising, then shuffling so relation *indices* carry
   no implicit rank ordering. The evaluation is deterministic rather than an iid draw: with `R` small
   the rank curve *is* the signal, so sampling would only add variance to a quantity with almost no
   degrees of freedom left. The old `Zipf(R, relation_zipf.exponent)` path is gone — see
   [§ Relation frequency](#relation-frequency) — and `relation_zipf_exponent` survives only as the
   fallback for a graph with no relations at all.
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
   never Block B's raw retained arrays (`_adapters.sample_degree_sequence`).
   A `(1 − subject_frac)·V` share of entities draw degree **0** — the non-subjects (and
   symmetrically non-objects on the in-side). Not every entity emits an edge: on swdf only
   30% do, and spreading the whole out-budget over all `V` entities both flattens the
   distribution and, via Stage 2's ≥1-edge-per-CS floor, drives `Σ|CS|` far past the edge
   budget (swdf: 606 500 vs 242 256), collapsing the realised CS size from 6 to 1. Of the
   nonzero nodes, the top `10%` of *all* `V` (the tail) draw from a power law truncated to
   `[p90, max]` whose exponent is **extreme-value matched** (`1 + ln(n_tail)/ln(max/p90)`,
   so the expected maximum of the tail draws lands on the measured max) rather than the
   fitted degree α — the global fit is too shallow for this range and would overshoot
   mid-tail mass; the remaining active nodes (the body) draw from the *same* fitted-α power
   law truncated to `[1, p90]`, then repaired — up *or* down, via `repair_degree_sum` — so
   the sequence sums to exactly `content_E` (edge conservation). The repair is confined to
   the body, at a **floor of 1** so the active nodes keep degree ≥1 (a heavy in-hub would
   otherwise make the tail so heavy that the repair zeroes out the body, inventing
   non-objects the signature never had); the tail carries p90/max and is bent only as a last
   resort when the tail alone exceeds the budget. The zeros carry `subject_frac`/`object_frac`
   and are never disturbed. The mean it targets is the **content** mean `content_E/V`, not `E/V`
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
   - `_build_distinct` dedups against a `seen` set **shared across all groups of one family**
     (one for forward pools, one for inverse). `num_distinct_cs` is a *union-over-groups*
     property, but the group prototype rows are near-identical, so independent per-group
     `seen` sets made every group redraw the same high-probability sets — Σquotas templates but
     only ~half as many *distinct* ones (wn18rr_v4: 44 quota → 23 distinct → realised ~20).
     A shared `seen` forces later groups past the sets earlier groups already took, so the
     union approaches Σquotas (wn18rr_v4 realised ~20 → ~30, codex_l/swdf pool-union exact).
     A starved group (attempt cap hit with every drawable set already claimed) is floored to
     one duplicate draw so it still contributes ≥1 realised template.
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
   Stage 1 — see Stage 1 §7): sampled target values are **rank-matched** to CS size (out-side;
   largest target → largest CS, floored at `|CS|`) and to inverse-CS size (in-side). Both sides are
   then repaired by `repair_degree_sum` to the **same** budget, `content_E`.

   The entities that draw a **zero** degree are the non-subjects (out-side) / non-objects (in-side).
   The sorted samples put those zeros on the lowest-CS-size entities, and their forward (resp.
   inverse) **CS is then blanked** — a zero-out-degree entity emits nothing, so it must not sit in
   any relation's subject pool, or `fit_stubs` would floor it back to ≥1. This is what keeps `Σ|CS|`
   within the edge budget; without it Stage 2 gave every entity a CS, and on swdf (only 30% subjects)
   that asked for 606 500 CS relations against a 242 256-edge budget, forcing the ≥1-per-CS floor to
   be dropped and collapsing the realised CS size 6 → 1. The zeros keep floor 0 and are held out of
   the repair, so neither the floor nor a top-up revives them.

   `Σ tgt_out == Σ tgt_in == content_E` is asserted, not merely intended. These two vectors are the
   **row margins** of the IPF allocation in §6, and a transportation problem whose margins disagree
   has no solution at all — the wiring would silently inflate or starve by the difference rather than
   fail. (An early version of this overshot wn18rr_v4's edge count by 9% for exactly that reason.)

   The repair escalates rather than returning a residual: trim the body first, preserving the hub tail
   that carries p90/max; and if the budget is *still* unreachable, the CS-size floor itself
   over-determines it (`Σ|CS(v)| > content_E` — swdf's characteristic sets ask for 606 500 edges
   against a budget of 242 256), so **drop the floor** and fall back to the unfloored degree law, which
   already sums to the budget by construction. Trimming *towards* the floor instead is what a first
   version did, and it destroys the degree law: the trim is weighted by headroom above the floor, so
   the hubs — which have by far the most — absorb nearly all of it (swdf's max out-degree target
   collapsed 623 → 18). The floor is the softer constraint: an entity emitting no edge for one CS
   relation costs a little Block-D fidelity, whereas a flattened degree sequence costs the whole of
   Block B. `fit_stubs` still honours the floor per relation wherever that relation's budget affords it.

   `DEGREE_QUOTA_SLACK` is **gone**. It existed to trade deficit-recovery volume against degree
   fidelity; the IPF allocation leaves no deficit to trade against.
5d. **Pool overlap for reciprocal relations** (when `relation_reciprocity` is set):
   real graphs pack directed content edges onto **shared** node pairs (parallel/multi-relational
   overlap and bidirectional pairs), whereas the CS-first construction above assigns forward and
   inverse CS **independently**, so `S_r ∩ O_r` (entities eligible to both emit *and* receive `r`)
   is tiny — even for a relation Stage 1 marked symmetric (`ρ_r≈1`). This pass adds `r` to a `ρ_r`
   fraction of `r`'s emitters' inverse CS (swapping out one existing entry so inverse-CS *size* —
   and the `inv_cs_size_q` / degree-rank-matching above — is unaffected; only *which* relations an
   entity receives changes), enlarging `S_r ∩ O_r` before the wiring loop below needs it. See
   `developer_docs/notes/motif_reachability_and_edge_multiplicity.md` and
   `developer_docs/notes/relation_reciprocity_and_bidirectionality.md` for the diagnosis.
6. **Joint stub allocation (IPF), then per-relation pairing within `S_r × O_r`.** `S_r` = subjects
   whose forward CS contains `r`; `O_r` = objects whose inverse CS contains `r`. Every entity's
   out- and in-stub count **per relation** is decided for all relations at once, by iterative
   proportional fitting — see [§ The IPF stub allocation](#the-ipf-stub-allocation) for what that
   is and why. In outline:
   - Build the sparse (entity × relation) supports `Ω_out = {(v,r) : r ∈ CS(v)}` and
     `Ω_in = {(v,r) : r ∈ invCS(v)}`, seeded with the weights the loop used to compute inline:
     `power-law(α_obj_r) · cs_size^a_obj` and `power-law(α_subj_r) · inv_cs_size^a_subj`.
   - `solve_edge_budget` finds the per-relation edge counts `e_r` **both sides can actually
     realise**, starting from the relation-frequency allocation and shrinking any relation whose
     pools cannot absorb it, redistributing the surplus to relations with room (bounded above by
     `|S_r|·|O_r|`, since an edge needs a distinct pair).
   - `fit_stubs` fits each side to those margins: rows to the degree targets, columns to `e_r`. The
     column sums come out **exactly `e_r` on both sides**, so every relation's out-stubs and
     in-stubs match and can be paired. The out side carries the ≥1-edge-per-CS-relation floor,
     imposed by substitution (`X = 1 + X'`) rather than by post-hoc repair. Each entry is capped at
     the opposite pool's size (`X[v,r] ≤ |O_r|`, `Y[v,r] ≤ |S_r|`) — a relation cannot carry the same
     `(s,o)` pair twice, so an entity allocated more stubs of `r` than there are partners to spend
     them on is not merely hard to place but **unrealisable**.
   - **Stub reservation** (when `ρ_r > 0`): the two sides are fitted independently, so an entity
     eligible for both roles rarely gets a stub on *both* by chance. For up to
     `round(ρ_r·edges_r/2)` entities in `S_r ∩ O_r`, force their out- and in-stub counts to ≥1,
     taking the stub from an entity with a surplus. The donor is drawn **uniformly among entities
     with a surplus** — taking it from `argmax` instead (as this originally did) robs the same
     entry repeatedly, and that entry is by definition the hub carrying the max/p90 degree target.
     It was harmless when the allocation came from `ones + multinomial` (no zeros, nothing to
     reserve), but with real hub mass in the allocation it flattened the peak completely: aids'
     realised max out-degree collapsed to 4 against a target of 11.
   - **Pair** the stubs within `S_r × O_r`, **object by object, tightest first**, taking each
     object's subjects **without replacement** — see
     [§ Why the pairing draws without replacement](#why-the-pairing-draws-without-replacement) for
     why proportional stub-drawing systematically undershoots the in-hubs and cannot be repaired
     after the fact. Three passes, in this order:
     - **B1 — the tight objects** (needing `≥ |S_r|/4` of the subject pool), with an unbounded scan.
       They run **before** Phase A, which would otherwise burn exactly the stubs they need on random
       mutual pairs.
     - **A — mutual-pair construction** (reciprocal relations): draws entities *with replacement*
       from `S_r ∩ O_r` (an entity stays available across multiple mutual pairs until either its
       out-stub or in-stub supply is exhausted) and places `e1→e2` + `e2→e1`, up to the reserved
       target — this is what actually realises bidirectional pairs.
     - **B2 — everything else**, still object-centric and without replacement, but bounded at
       `MAX_PAIR_RETRY` misses.
     **Multi-relational biasing** (`edge_multiplicity`/`bidirectional_ratio` targets, independent of
     per-relation reciprocity) runs inside each object's fill: when behind the parallel target, first
     take subjects that already point at this object via another relation.
   Reciprocity attainment is **capped by the entity pool and average stub multiplicity**, not just the
   mechanism: e.g. on wn18rr_v4 the biggest relation needs ~3.4 stubs/entity on average from its
   shared pool to hit its reciprocity target but only has ~2.85 available, so a real shortfall
   remains even with reservation + full stub reuse.
6b. **Inv-CS template completion** (step 4b, when `entity_inv_cs` is assigned): after the main
   wiring loop, detects object nodes whose actual in-predicate set is a strict subset of their
   assigned inverse-CS template. For each missing predicate `r`, finds an existing edge
   `(s', o', r)` where `o'` already receives `r` from ≥2 edges (so removing one doesn't break
   its template) **and is over its in-degree target**, and **redirects** it to `(s', o, r)`. No net
   edge-count change. The over-target condition is load-bearing: the object holding the *most* edges
   of a relation is the in-hub, so without it this pass raids precisely the node the wiring worked
   hardest to build — it stripped fb237_v4's hub from 117 edges of relation 178 back to 71, and the
   guard alone took that graph's max-in from 1 141 to 1 442 and dbpedia100k's from 5 676 to 14 767.
   Donors are drawn from one shuffled per-relation queue consumed across all gaps; re-scanning the
   relation's whole edge list per gap made this pass `O(gaps · |E_r|)`, which cost aids ~110 s. This enforces
   that every template predicate is realised on each object, reducing `inv_num_distinct_cs`
   from ~63 to ~32 (target) before Stage 3. Stage 3 degree-preserving swaps partially undo this
   (ending around 55), but Stage 3 initial loss improves (49.6 vs 56.2 without this pass).
7. **Connect components** — bridge isolated components into the giant, *selectively*: keeps up
   to `target_nc − 1` satellite components unbridged (chosen so their combined size is closest to
   `(1 − target_lcc) · V`), bridges the rest. Runs **first** among the connectivity-affecting
   steps and returns an `is_satellite` mask that the two passes below must respect — neither may
   place an edge touching a satellite node, or it would silently reconnect a component this step
   chose to leave isolated, undoing the `target_nc` / `target_lcc` guarantee.
7a. **Pairing residual.** The IPF allocation leaves no *budget* deficit — every stub the degree law
   asks for has a relation to be spent on. What can still fail is the **pairing**: a bounded (B2)
   object's last few stubs may find only self-loops or already-used `(s,o)` pairs within
   `MAX_PAIR_RETRY` misses. The residual is now tiny (0–0.15% of the budget, zero on four of the nine
   graphs) and is placed by sampling `(subject, object)` pairs weighted by remaining quota.
   Subject/object pools are filtered to non-satellite nodes first.
7b. **Trim to the edge budget.** `_connect_components` appends its bridging edges *on top* of
   whatever the main loop placed. That used to be invisible, because the old wiring always finished
   short and the bridges landed in the shortfall; the IPF allocation saturates the budget, so those
   bridges now push `|E|` over it (aids overshot by 456). An equal number of edges is removed to make
   room. Only **non-bridge** edges (undirected sense, via `igraph.bridges()`) are eligible — removing
   one cannot split a component, so the `target_nc` / `target_lcc` structure just established
   survives — and among those, the edges whose endpoints most *exceed* their degree targets go first,
   so the trim improves degree fidelity rather than degrading it.
8. **`rdf:type` edges** for typed entities; assemble the `igraph.Graph` with the `kg_io.load_kg`
   attribute contract (so `compute_reduced_signature` can read it back).

Stage-2 tuning constants (`MAX_PAIR_RETRY`, `SIZE_ESCAPE_FAILS`, `TEMPLATE_ATTEMPT_*`) are
module-level at the top of `stage2.py`; the IPF constants (`IPF_ITERS`, `OUTER_ITERS`) at the top of
`_ipf.py`.

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
- **Measured relation frequency.** Relation weights are rebuilt from the measured `rel_freq_logq`
  quantile function, not from a fitted Zipf exponent (which lost on every corpus graph — see
  [§ Relation frequency](#relation-frequency)).
- **Per-relation multiplicity + edge conservation.** Out-side wiring allocates each relation's
  edges by `multinomial` on per-relation α (`obj_alpha`) and the G2b `a_obj` CS-size offset,
  hitting each relation's edge budget `|edges_r| = freq(r)·E` exactly. In-side allocation draws
  per-relation **subject**-multiplicity from `subj_alpha` × a capacity-weighting factor (remaining
  quota, from the Stage-1-sampled `target_in_degrees`). There is no
  preferential-attachment mechanism in the current code — an earlier PA-based in-degree lever was
  replaced by this target-degree-sequence approach (see Stage 1 §7 and Stage 2 §6's in-side
  wiring).
- **Realizability.** With `α_subj<2` (infinite mean), an unconstrained in-side draw would concentrate
  nearly all of `|edges_r|` on one object, leaving unplaceable duplicates and collapsing the edge
  budget. The IPF allocation bounds this at the source: each entity's in-stub count across all
  relations is its degree target (a *row margin*, not a post-hoc cap), and `solve_edge_budget` bounds
  each relation at `|S_r|·|O_r|` distinct pairs. The old `_cap_redistribute` clamp is gone — see
  [§ The IPF stub allocation](#the-ipf-stub-allocation) for why clamping was the bug rather than the
  fix.
- **CS templating for `num_distinct_cs`.** Reaching the target distinct-CS count needs three
  floors working together: (a) distinct templates drawn by rejection + size-escape, (b) a floored
  entity→template assignment (≥1 entity per template) with a power-law reuse tail, and (c) an
  out-side floor (every subject of `r` gets ≥1 edge, imposed inside `fit_stubs` by substitution).
  Without them a too-steep reuse tail leaves most templates unused and out-side allocation zeros
  shrink the realised CSs.
- **Symmetric inverse CS + `a_subj`.** Block D's `inv_num_distinct_cs` / `inv_cs_freq` and the
  inverse-CS templates (object side) let edges be matched within `S_r × O_r`, and the in-side
  weight gains the `inv_cs_size^a_subj` G2b offset — the object-side mirror of `a_obj`.
- **Tuning constants** for each stage are module-level at the top of
  `stage1.py` / `stage2.py` / `stage3.py`.

---

## The IPF stub allocation

Stage 2 must decide, for every entity `v` and every relation `r` it is eligible for, how many
out-stubs `X[v,r]` and in-stubs `Y[v,r]` it gets. Two families of constraint bear on that at once:

* **rows** — `Σ_r X[v,r] = tgt_out[v]`: the per-entity degree targets (Block B's degree law);
* **columns** — `Σ_v X[v,r] = Σ_v Y[v,r] = e_r`: the per-relation edge budget, which must be the
  *same on both sides* or the relation's stubs cannot be paired at all.

### What was wrong before

The old loop drew each relation's stubs independently, `m_obj ~ Multinomial(edges_r, w)` and
`m_in ~ Multinomial(edges_r, w)`. That hits the **column** margin and says nothing about the rows, so
the degree target was bolted on afterwards as a *cap* (`_cap_redistribute(hard_cap=…)`). Capping each
side independently against the global remaining quota of its own pool is precisely what broke the
column margin again: the two sides ended up with **different stub counts**, the surplus could not be
paired, and the difference fell into a uniform-random deficit-recovery pass.

Measured per-relation stub imbalance under the old scheme: fb237_v4 **1 843**, wn18rr_v4 **790**,
aids **184 992** — and that imbalance *was* the deficit. On aids it meant a third of the content edges
were placed with no multiplicity law, no preferential attachment and no degree law, at a cost of
~185k `rng.choice` calls over 70–200k-element pools. The diagnosis is in
`developer_docs/plan/per_relation_stub_balance.md`.

### What IPF does

Seed a matrix `W` on the sparse support with the same weights the loop already computed
(`power-law(α_r) × cs_size^a`), then alternately rescale rows to hit `tgt_out` and columns to hit
`e_r` until both margins hold — Sinkhorn's algorithm, a.k.a. Deming–Stephan iterative proportional
fitting. Every operation multiplies a whole row or column by one positive scalar, so the fitted
matrix has the form `A[v,r] = W[v,r]·u[v]·w[r]`: the search is over `V + R` numbers, not `nnz`.

That form is why this is the *right* tool and not merely a working one. All cross-ratios of `W`
survive exactly — `A[v,r]·A[v',r'] / (A[v,r']·A[v',r]) == W[v,r]·W[v',r'] / (W[v,r']·W[v',r])`, since
the `u`/`w` factors cancel. The multiplicity law and the G2b CS-size coupling come out untouched;
only the margins are forced. Formally `A` is the I-projection of `W` onto the transportation polytope:
the closest matrix to `W` in KL divergence with the required margins.

Cost is trivial — the support is sparse (`nnz = Σ_v |CS(v)|`: ~21k on fb237_v4, ~570k on aids), and
each sweep is two `bincount` passes over it.

### Infeasibility is an output, not a failure

Convergence holds iff a matrix with that support and those margins exists. Ending each fit on a **row**
step means every entity's degree quota is spent exactly, and the achieved *column* sums then report
what the supports could actually deliver — and they automatically sum to `Σ tgt_out = content_E`, so
**the budget is fully spendable and there is no deficit by construction**. Where a relation's pool
cannot absorb its budget, its column comes out short and the surplus has already flowed to relations
with room. That is the "shrink `e_r`, redistribute" policy, obtained for free rather than as a separate
mechanism, and `solve_edge_budget` logs it so a shrunk relation is visible.

### Two numerical traps, both hit in practice

* **Do not accumulate `u`/`w`.** Algebraically identical, numerically fatal: a column whose target far
  exceeds what its support can supply drives `w` to `inf`, the next product is `nan`, and the whole
  allocation is silently destroyed (swdf came out with a 170-edge budget against a target of 242 256).
  `ipf()` rescales the value vector in place instead — each row sweep renormalises it back to the row
  margins, so it stays bounded.
* **`_largest_remainder` rounds; it does not scale.** It can move each entry by at most one, so it only
  reaches the total when its input already sums to within `len(values)` of it. Call `_fill_to_total`
  first. (Handing it a badly-scaled vector returns a sum of exactly `len(values)` — which is where that
  170 came from: swdf has 170 relations.)

### Results

Against the previous wiring, across all 9 corpus graphs, `|E|` still lands exactly on target and:

| graph | deficit before | deficit after | max-out (target) | max-in (target) | Stage 2 |
|---|---|---|---|---|---|
| `wn18rr_v4` | 841 | **0** | 27 (26) | **64 (64)** | 0.1 s |
| `wn18rr_v4_ind` | — | **0** | 28 (29) | **21 (21)** | 0.2 s |
| `fb237_v4` | 2 284 | **13** (0.04%) | **195 (195)** | 1 435 (1 442) | 0.3 s |
| `fb237_v4_ind` | — | **22** (0.15%) | 147 (146) | **318 (318)** | 0.2 s |
| `aids` | **184 992** | **0** | 10 (11) | **11 (11)** | 8.4 s |
| `codex_l` | — | **150** (0.02%) | 267 (265) | **18 433 (18 433)** | 8.2 s |
| `swdf` | — | **0** | 499 (623) | 9 128 (9 148) | 19.9 s |
| `hetionet` | — | **1 781** (0.08%) | 22 592 (23 866) | 1 957 (2 718) | 10.1 s |
| `dbpedia100k` | — | **26** (0.004%) | 117 (115) | **14 767 (14 767)** | 13.0 s |

aids' Stage 2 drops from **15+ minutes to ~8 seconds** (the deficit pass it no longer runs was
`O(deficit × |pool|)`); codex_l 88 s → 8 s, hetionet 220 s → 10 s, dbpedia100k 138 s → 13 s.

The in-hubs — the thing the old wiring missed worst — now land **exactly** on target on `codex_l`
(18 433), `dbpedia100k` (14 767, up from 5 676), `fb237_v4_ind`, `wn18rr_v4` and `wn18rr_v4_ind`; see
[§ Why the pairing draws without replacement](#why-the-pairing-draws-without-replacement).

Still short: `hetionet`'s max-in (1 957 / 2 718) and `swdf`'s max-out (499 / 623). Both graphs have
in-hubs whose demand exceeds what their pools can supply even in principle; the per-entry cap in
`fit_stubs` clips the allocation to what is realisable, and the shortfall is then genuine rather than
lost in the wiring. `fb237_v4`'s max-in (1 435 / 1 442) and `aids`' max-out (10 / 11) are 7 and 1 short
of exact respectively — both are recoverable by protecting the degree tail in the reciprocity stub
reservation (measured), which is left off pending a check of what it costs reciprocity.

---

## Why the pairing draws without replacement

The subject reservoir is a shuffled multiset with each subject repeated once per remaining out-stub,
so drawing from it is a draw **∝ remaining stubs** — the ordinary configuration model. That is
unbiased for a *multigraph*. But a relation may not carry the same `(s, o)` pair twice, so when a
subject with several stubs picks the same object twice, the duplicate is discarded.

The loss falls hardest on exactly the objects whose stub share is large enough to collide with
themselves — the in-hubs. This is the **erased configuration model** bias, and it is not a small
correction. Measured on `fb237_v4`'s in-hub:

| relation | hub needs (distinct subjects) | hub's share of in-stubs | proportional draw yields | shortfall |
|---|---|---|---|---|
| 58 | 178 of 258 | 42.0% | ~137 | −41 |
| 178 | 117 of 117 | 58.3% | ~82 | −35 |
| 193 | 275 of 832 | 18.6% | ~246 | −29 |

Note relation 58: the hub must be reached by **69% of every subject of that relation** while holding
only 42% of its in-stubs. *No proportional rule can turn 42% into 69%* — the probability has to exceed
the stub share to compensate for the collisions. Sampling proportionally is the one thing guaranteed
to undershoot a hub.

So the pairing fills **one object at a time**, taking its subjects **without replacement**: an object
with `k` in-stubs needs `k` *distinct* subjects, and drawing them without replacement makes
distinctness structural rather than a rejection test, so the collision loss is zero by construction.
The draw stays random and stays weighted by remaining stubs — "without replacement" simply moved from
the test into the sampler. (The textbook alternative is the max-entropy *simple*-graph model,
`p_so = x_s·y_o / (1 + x_s·y_o)`, where the `1/(1+xy)` **is** the collision correction; it needs a
dense `|S_r|×|O_r|` fit, which is 1.6×10¹⁰ entries for aids' biggest relation, so it is not used.)

Objects are served in **decreasing tightness** (= decreasing stub count, `|S_r|` being fixed within a
relation), and the tight ones are served **before Phase A**. Neither is optional:

* Tightness cannot be *detected late*. By the time a random pairing "gets hard", the distinct subjects
  a hub needed have already spent their stubs elsewhere, and no endgame repair can un-spend them. The
  ordering is a statement about the *beginning* of the process, not the end.
* Phase A (mutual pairs) otherwise burns exactly those stubs on random reciprocal pairs. On
  `fb237_v4`'s relation 178 — where the hub needs *every* subject in the pool — that alone cost it 46
  of 117.

Objects needing a large share of the pool get an unbounded scan; the rest are bounded at
`MAX_PAIR_RETRY`, since an object needing a handful of subjects out of thousands has ample freedom and
walking the whole reservoir for it would cost far more than it buys. Tight objects are few — needing
`≥ |S_r|/4` stubs bounds their number at `4·edges_r/|S_r|` — so the unbounded scan is affordable
exactly where it is needed.

### Three passes that funded themselves out of the degree tail

The same bug appeared three times, in three different repair passes, and each one silently undid the
wiring's work on the hubs. The pattern is worth naming: **a repair pass that needs a spare edge will
find the most spare edges at the hub, which is precisely the node whose degree is hardest to hit.**

* **The reciprocity stub reservation** (`_reserve`) took its donor stub from `np.argmax(m)`, over and
  over — the hub by definition. It flattened aids' max out-degree to **4** against a target of 11.
  Donors are now drawn uniformly among entities with a surplus.
* **The inv-CS template completion** (step 4b) redirects an edge away from any object holding ≥2 edges
  of a relation — and the object holding the *most* is the in-hub. It correctly wired `fb237_v4`'s hub
  to all 117 subjects of relation 178 and then stripped it back to **71**, costing ~300 in-edges in
  total. A donor edge may now only be taken from an object that is **over** its in-degree target: the
  pass spends surplus, never target-critical mass. This single guard took `fb237_v4`'s max-in from
  1 141 to 1 442 and `dbpedia100k`'s from 5 676 to 14 767.
* **The budget trim** (step 5b) removes the excess from the edges whose endpoints most *exceed* their
  degree targets, for the same reason — designed with the trap in mind rather than fixed after it.

---

## Relation frequency

Block B stores relation usage as `rel_freq_logq` — the empirical quantile function of
`log(E_r / Σ E_r)` — and Stage 1 rebuilds the rank curve from it. It used to store a Zipf
exponent (`relation_zipf`) and Stage 1 built `relation_weights = Zipf(R, exponent)`. That path was
replaced because it was wrong in three separate ways.

**It was a unit error.** `fit_zipf` called `_fit_powerlaw(counts)`, which fits the *count
distribution* `P(count = x) ∝ x^(−α)` with `xmin` pinned to 1. Stage 1 then consumed that α as a
*rank-frequency* exponent (`ranks ** (−α)`). These are different laws — a rank-Zipf with exponent
`s` has `α = 1 + 1/s`.

**The fitted exponent carried no information.** It came out pinned at ≈1.0 on all six corpus graphs
where it could be fitted at all (1.000, 1.068, 1.000, 1.000, 1.000, 1.014), so `relation_weights ≈
rank⁻¹` on *every* graph regardless of shape — which is why `fb237_v4`'s top relation was handed
16.8% of the edge budget against a real share of 5.9%. On the other three (`aids` R=5, `wn18rr_v4`
R=9) the fit returned NaN and the generator fell back to a hard-coded `Zipf(2.0)`.

**These curves are frequently not Zipf-shaped at all**, so no exponent — however fitted — can
represent them. `aids`' shares are `.337 / .284 / .230 / .060 / .003`: a flat head, then a cliff. A
`Zipf(2.33)` (its own log-log slope) over R=5 puts 0.75 on the top relation against a real 0.337.

Reconstruction error against the true share vector, all 9 corpus graphs (log-share RMSE; `top err` is
the relative error on the largest relation's share):

| graph | R | Zipf (old) | top err | OLS rank exp. | top err | **log-quantile** | top err |
|---|---|---|---|---|---|---|---|
| `aids` | 5 | 1.199 | 103% | 1.185 | 122% | **0.000** | 0.0% |
| `codex_l` | 69 | 2.511 | 32% | 2.435 | 145% | **0.534** | 28% |
| `dbpedia100k` | 470 | 2.429 | 97% | 4.091 | 708% | **0.480** | 37% |
| `fb237_v4` | 219 | 0.597 | 184% | 0.798 | 455% | **0.152** | 6.7% |
| `fb237_v4_ind` | 200 | 0.581 | 145% | 0.743 | 378% | **0.189** | 9.4% |
| `hetionet` | 24 | 1.804 | 6.5% | 1.326 | 184% | **0.252** | 11% |
| `swdf` | 170 | 2.806 | 94% | 3.839 | 740% | **0.149** | 6.0% |
| `wn18rr_v4` | 9 | 1.371 | 11% | 1.029 | 45% | **0.287** | 7.3% |
| `wn18rr_v4_ind` | 9 | 0.804 | 0.0% | 0.400 | 21% | **0.119** | 4.0% |

Three design consequences, each measured rather than assumed:

- **No goodness-of-fit gate.** One was built, to pick the Zipf where it fits and the quantile
  function where it doesn't. It degenerates to a constant — the quantile fit wins on every graph —
  so the Zipf is simply gone rather than gated.
- **Log space is mandatory.** The shares are heavy-tailed; a *linear*-space quantile fit reconstructs
  them worse than the Zipf did.
- **Fixing the units is not enough.** An OLS rank exponent — the natural repair — loses on 8 of 9
  graphs, for the "not Zipf-shaped" reason above. It was evaluated and rejected.

`min_samples=2` on the fit: a relation's edge share is a directly-reliable statistic, not a noisy
per-item power-law fit, so the Clauset small-sample caveat that justifies `MIN_SAMPLES_FOR_FIT=10`
elsewhere does not apply (the same reasoning already used for the per-relation reciprocity fractions).
This is what makes `aids` (R=5) and `wn18rr_v4` (R=9) measurable at all.

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
(if it is ever wanted) were recorded in an earlier investigation that has since been pruned from
this tree; git history has it if this is revisited.

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
