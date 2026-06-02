# Signature Distribution Observations

Observations drawn from component-wise histogram plots of the graph signatures
measured across: lubm, freebase, AIDS, codex, fb237, hetionet.

---

## General notes

### Across-graph distributions are not yet known — must be measured

- Applies to all the stage-1 / stage-2 notes below.
- Where properties of the distribution **over real-world graphs** were noted, these
  are **not assumed** — they still have to be measured again.
- Distinguish two levels: the distribution **inside a single graph** (what the
  signature describes) vs. the distribution of those parameters **across real-world
  graphs** (what stage 1 would sample from, once measured).

### Two stage-2 strategies (sampling vs. convergence)

- **Stage-2 sampling** is one strategy to converge to a target within-graph
  distribution: draw per-element values directly from the described distribution
  (works for per-relation multiplicity α, functionality, class size, CS size).
- For **emergent properties** direct sampling is **not possible** — the value falls
  out of the topology, not from a per-element draw (row entropy, singular values,
  per-type relation entropy, shortest path length).
- For those, a different strategy is needed: build the sampleable structure first,
  then **iteratively adjust the topology toward the target** (e.g. degree-preserving
  edge rewiring driven by the target statistic / optimisation loop). The target
  distribution is the objective, not a source to sample from.
- ⚠️ This alternative strategy still has to be worked out per emergent property.

### Sampling contradictions between properties

- A bigger structural problem: not all properties can be sampled independently —
  sampling one property can contradict another.
- Example: if we sample class sizes for certain classes, the resulting mapping
  from entities to classes could contradict other properties.
- Open question: which things can actually be sampled freely, and which are
  constrained by already-sampled properties?

---

## Stage 1 — sampling a signature from the real-world population

Stage 1 draws a **novel target signature** from the distribution of real-world
graphs (this is the part **not yet implemented** — the current generator always
consumes one measured signature directly). Sampling each signature feature
independently from its own marginal would produce **incoherent signatures**
(combinations no real graph exhibits). The dependencies must be handled, and they
come in two distinct kinds:

### 1. Algebraic dependence → derive, do not sample

Some features are exact functions of others within a single signature and must be
**computed, never sampled** (sampling them is the contradiction source):

- `density = E / V²`, `triples_per_entity = E / V`, `relation_reuse = E / R`
- `star_count_k = Σ C(deg(v), k)` (function of the degree sequence)
- `functionality ≈ f(multiplicity)`; inverse CS = f(forward CS + wiring)

First step: reduce the signature to a **minimal set of free parameters** and
derive the rest. This shrinks the signature to its real degrees of freedom.

### 2. Statistical correlation → dependent (joint) sampling

The remaining free parameters still **co-vary across real graphs** (e.g. graph
size correlates with degree exponent, clustering, CS sizes). These need
**dependent / joint sampling** — a joint distribution over the free parameters
(multivariate normal on transformed marginals, a copula, or a conditional model),
not independent marginals.

### The practical catch: only ~6–10 real graphs measured

A full covariance over a high-dimensional free-parameter space is **not estimable**
from a handful of graphs. So dependent sampling has to be **low-dimensional and
structured**:

- Reduce to as few free parameters as possible (step 1 already helps).
- **Condition on size:** sample a size variable (e.g. `num_entities`) first, then
  sample everything else *conditioned on size* — captures the dominant correlation
  cheaply.
- Model only the pairwise / hierarchical dependence that can actually be estimated;
  treat the rest as conditionally independent given size.

> The biggest win is collapsing the algebraic dependencies (step 1) **first**, so
> the joint distribution that has to be estimated from few graphs is small enough
> to be learnable.

---

## Block B — Degree Structure

### Per-relation power-law alphas (`obj_multiplicity_alpha`, `subj_multiplicity_alpha`)

- Distributions are mostly skewed normals.
- Cutoff at approximately **α ≈ 1.4** (lower bound) and **α ≈ 3.0** (upper bound).

**Implication for the generator (two-stage design):**

- **Stage 1 — Signature sampler:** samples the distribution parameters (center,
  scale, skew) and the cutoff bounds for this skewed normal. These across-graph
  distributions are **not yet measured** — fitting them over multiple real-world
  graphs is still to be done.
- **Stage 2 — Graph generator:** for each individual generated graph, samples a
  per-relation α from the stage-1 skewed normal; this α then parameterises the
  power-law used to sample subject/object multiplicity for each relation.
  (Directly sampleable — see general note.)
- 🔧 **Signature change needed:** currently only
  `obj/subj_multiplicity_alpha_{mean,std,median}` are stored — replace with the
  skew-normal params (center, scale, skew) + lower/upper cutoff (~1.4 / ~3.0).

---

### Aggregate out-degree & in-degree (`out_degree_alpha`, `in_degree_alpha`)

- Both follow a power-law.
- The α values need to be sampled from a distribution still to be measured on
  real-world graphs; observed range is roughly **0 – 2000**.
- The absolute range of out/in-degree is bounded by the number of nodes, so the
  observable α range scales with graph size — it cannot be treated as a fixed
  constant across graphs of different sizes.
- **Hypothesis:** handling this size-dependence through the power-law α directly
  may be sufficient — for a fixed α, sampling degrees for n nodes from a power-law
  naturally produces a maximum that grows with n, so the scaling is implicit.
  An explicit **upper-limit guard (degree ≤ num_nodes)** still needs to be applied.
  This needs to be verified.

---

### Functionality & inverse functionality (`functionality`, `inverse_functionality`)

- Both distributions resemble truncated power-laws on [0, 1].
- In some cases the power-law is **mirrored around x = 0.5**, producing strongly
  skewed distributions with a peak at 1 instead of 0.
- **`inverse_functionality`** most commonly has its peak at **0** (standard power-law).
- **`functionality`** most commonly has its peak at **1** (mirrored power-law).

**Implication for the generator (two-stage design):**

- **Stage 1 — Signature sampler:**
  - Samples the power-law α for both `functionality` and `inverse_functionality`.
    The across-graph distribution of these αs is **not yet measured** — still to be
    fitted over real-world graphs.
  - Additionally samples, independently for each of the two, a **mirror
    probability** — the probability that the power-law is mirrored (peak at 1
    rather than 0).
- **Stage 2 — Graph generator:** uses the sampled α and mirror flag to construct
  the truncated power-law distribution on [0, 1] from which per-relation
  functionality and inverse-functionality values are drawn. (Directly sampleable.)
- 🔧 **Signature change needed:** currently `functionality_{mean,std,median}` and
  `inverse_functionality_{mean,std,median}` — replace with the truncated power-law
  α on [0, 1] + a mirror flag/probability, per side.

---

## Block C — Schema & Co-occurrence

### Singular values of co-occurrence matrices (`subj_singular_value_01` … `_10`, `obj_singular_value_01` … `_10`)

- The singular values as a function of rank appear to follow **exponential decay**
  on both the subject-side and object-side co-occurrence matrices.
- **Stage 1:** sample the decay parameter and a magnitude scale factor. The
  across-graph distribution of these is **not yet measured** — still to be fitted
  over real-world graphs.
- **Stage 2 (emergent — not directly sampleable):** singular values fall out of the
  co-occurrence structure; converge to the scaled exponential via the alternative
  strategy (see general note), letting the top-k approach the target curve.
- 🔧 **Signature change (optional):** the 10 raw singular values per side could be
  replaced by (decay rate + magnitude scale); the current 10-value form is usable
  but redundant.
- ⚠️ Needs to be checked again.

### Row entropy of co-occurrence matrices (`subj_row_entropy_mean`, `obj_row_entropy_mean`, …)

- The row entropy distributions appear to follow a **skewed normal** on both the
  subject and object side.
- Centers fall roughly between **1 and 3** (nats); the **object side appears to
  have a lower center than the subject side**.
- **Stage 1:** assume skewed normal; sample skew, center, scale. The across-graph
  distribution of these params is **not yet measured** — still to be fitted over
  real-world graphs.
- **Stage 2 (emergent — not directly sampleable):** entropy falls out of the
  co-occurrence structure; converge to the target distribution via the alternative
  strategy (see general note).
- 🔧 **Signature change needed:** currently `subj/obj_row_entropy_mean` and `…_std`
  — add **skew** (a skewed normal needs 3 params).
- ⚠️ Needs to be checked.

### Class size distribution (`class_size_zipf_exponent`, `num_classes`)

- The number of entities per class (class size) is **power-law fitted**.
- **Stage 1:** sample the power-law α; the across-graph distribution of α is **not
  yet measured** — still to be done.
- **Stage 2:** unlike row entropy, **class size can be sampled directly** from the
  power-law distribution.
- 🔧 **Signature:** `class_size_zipf_exponent` covers the α; `num_classes` is stored
  but **how the class count itself is chosen is not yet covered** (see coverage).

---

### Per-type relation entropy (`mean_type_relation_entropy`)

- Follows **exponential decay** again.
- **Stage 2 (emergent — not directly sampleable):** converge via the alternative
  strategy (see general note), like the singular-values case — describe the
  rank→entropy curve as exponential decay and steer the topology toward it.
- 🔧 **Signature change needed:** currently only a single scalar
  `mean_type_relation_entropy` — store the decay-curve params (decay rate + scale)
  instead of just the mean, so the curve can be targeted.

---

## Block D — Characteristic Sets

### CS size distributions (`cs_size_mean`, `cs_size_median`, `cs_size_p90`, inverse variants)

- The CS size distribution **inside a single graph** is **skewed normal**.
- **Stage 1:** the across-graph distribution of the skewed-normal params is **not
  yet measured** — still to be done.
- **Stage 2:** samples per-entity CS sizes directly from the described skewed
  normal.
- 🔧 **Signature change needed:** currently `cs_size_{mean,median,p90}` (+ inverse
  variants) — replace with the skew-normal params (center, scale, skew) so the
  within-graph distribution can be reconstructed.

---

### Two-step pair frequencies (`pair_freq_top_01` … `_20`)

- The top-20 two-step pair frequencies look like **exponential decay with rank**
  again, but quite flat.
- **Approach (same as singular values / per-type relation entropy):** describe the
  rank→frequency curve as exponential decay. **Stage 1:** sample decay rate +
  magnitude scale — across-graph distribution **not yet measured**. **Stage 2
  (emergent — not directly sampleable):** converge the top-k toward the scaled
  curve via the alternative strategy (see general note).
- **Alternative framing:** instead check the **distribution of the frequencies**
  (this does not have to be restricted to the top 20).
- 🔧 **Signature change (optional):** `pair_freq_top_01..20` could be replaced by
  (decay rate + scale); current top-20 form is usable.

---

## Block F — Connectivity

### Shortest path length (`avg_shortest_path_length`, …)

- The shortest path length distribution **inside a graph** is **skewed normal**.
- **Stage 1 is the same** as before — sample the within-graph skewed-normal params;
  their across-graph distribution is **not yet measured**, still to be done.
- **Stage 2 (emergent — not directly sampleable):** shortest path length is an
  emergent property of the topology; converge toward the target via the alternative
  strategy (see general note).
- 🔧 **Signature change needed:** currently `avg_shortest_path_length` + `…_se` —
  store the skew-normal params (center, scale, skew) to describe the distribution.
- ⚠️ Open question: how do we reach a target shortest-path-length distribution
  during graph generation?

---

## Coverage — signature parts still uncovered

Components with an approach above: per-relation multiplicity α, functionality /
inverse functionality, out/in-degree α, singular values, row entropy, class size,
per-type relation entropy, CS size, two-step pair frequencies, shortest path length.

**No approach yet:**

- **Block A — entirely:** `num_entities`, `num_triples`, `num_relations`, `density`,
  `triples_per_entity`, `relation_reuse`. Global size/shape scalars — presumably
  fixed first, but no note describes how.
- **`num_classes`** — the class count itself (the class-size note covers only the
  size distribution).
- **Block C co-occurrence density:** `subj_cooc_density`, `obj_cooc_density`.
- **Block D CS counts & frequency fits:** `num_distinct_cs`, `inv_num_distinct_cs`,
  `cs_freq_alpha`/`cs_freq_ks`, `inv_cs_freq_alpha`/`inv_cs_freq_ks`.
- **Block E — entirely (36 features):** motif counts (triangle, 4-/5-/6-cycle,
  diamond, k4, tailed-triangle), star counts, path/tree template zipf & entropy.
  Not even measured yet (skipped → all NaN); no generation approach.
- **Block F connectivity (besides shortest path):** `num_components`,
  `largest_component_fraction`, `clustering_coefficient`, `degree_assortativity`.
- **KS goodness-of-fit fields throughout** (`out/in_degree_ks`,
  `*_ks_{mean,std,median}`, `cs_freq_ks`, `pair_freq_ks`): measurement diagnostics
  — open whether they belong in the generation signature at all.
