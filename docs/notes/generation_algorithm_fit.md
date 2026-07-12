# Generation algorithm vs the reduced signature — fit & reconciliations

Assessment of the proposed three-stage generation algorithm against the reduced,
non-over-determined signature ([../signature.md](../signature.md)) and the
guaranteed-vs-joint analysis. Companion to
[../archive/signature_measurement_plan.md](../archive/signature_measurement_plan.md).

## Stage ↔ bucket mapping

The three stages map almost exactly onto the three buckets from the derivability
analysis — the architecture is a **strong fit**.

| Algorithm stage | Our concept |
|---|---|
| **Stage 1** — schema sampler (|R|, |T|, P(r\|t), co-occurrence spectrum, per-relation params) | Sample the **free constructive params** (G0–G3) |
| **Stage 2** — CS-aware wiring (types → CS → edges via multiplicity + preferential attachment) | **Instantiate to hit the marginals** (P3 = CS-first primitive) |
| **Stage 3** — Maslov–Sneppen degree-preserving rewiring + simulated annealing | **Steer the free-emergent targets** without disturbing marginals |

Pattern = **sample marginals → realise → steer emergent.** Stage 3 being
degree-preserving correctly preserves the constructive params and moves only the
emergent ones.

The algorithm also implicitly settles two open decisions:
- **Primitive = P3** (CS-first). ✓ one of our options.
- **Schema = both** (spectrum in Stage 1, CS distributions in Stage 2) = **S-C**. ✓

## What aligns well (keep as-is)

- P3 CS-first instantiation.
- Degree-preserving simulated annealing for motifs = the free-emergent steering loop.
- Local `O(deg)` triangle-count deltas; stratified 10⁵ sample for 5/6-node motifs updated
  incrementally on swaps.
- "Start with very small graphs first."
- Validation: KS (continuous), absolute/relative error (scalar counts), KL (categorical),
  PCA/UMAP manifold.
- Explicit acknowledgement that **not all constraints are simultaneously satisfiable —
  "get near"** (matches the edge-conservation web + guaranteed/emergent split).

## Reconciliations needed

### 1. Multiplicity representation — make multiplicity-α primary, derive functionality
Stage 1 samples a per-relation **cardinality archetype {1-1, 1-N, N-1, N-N}** and
**expected functionality / inverse-functionality**. But functionality is the head
`P(count = 1)` of the multiplicity distribution → **derived**, not independently sampled
(sampling both can contradict). **Fix:** make the per-relation **multiplicity-α
(skew-normal)** the primary handle; derive archetype + functionality from it. Use the
archetype only as a coarse routing label, not an independent sampled quantity.

### 2. Degree — PA steers in-degree; out-degree is the residual
Stage 2 uses **preferential attachment** (objects ∝ `in_degree^exponent`, exponent tuned
to the target Zipf), which **does steer the in-degree** distribution. So:

- **In-degree:** steered by the PA exponent → matched at the **tail exponent** level.
  (PA is a one-parameter family — it matches the power-law tail, not an arbitrary
  in-degree *shape*.) ✅
- **Out-degree:** PA selects objects, so it does **not** touch out-degree.
  `out_degree(v) = Σ_{r∈CS(v)} m_obj(v,r)`. **Largely resolved by G2b** — the
  CS-size→multiplicity offset `cs_size^a` makes Stage 2 draw multiplicity *conditioned on
  CS size* instead of independently, so the dominant CS-size↔multiplicity correlation is
  reproduced at construction. Residual (relation-identity-within-CS) remains; out-degree
  stays a diagnostic/target, but Stage 3 still can't move it.
- **Per-relation subject-multiplicity:** Block B has a target subject-multiplicity α
  *per relation*, but PA shapes only the **aggregate** in-degree with a single exponent —
  it doesn't independently hit each relation's fan-in (the bipartite-realisability point,
  Investigation §4). → Best-effort; check it lands near the Block B target, or add a
  correction.

### 3. Stars — resolved: non-induced (CS-fixed), the algorithm is correct
The project uses the **non-induced** star (the spec: *"stars of degree k, already fixed by
characteristic sets"*) = `Σ C(deg,k)`. So Stage 2's claim that CS **pins stars without
further work is correct**, and stars are **dropped** from the signature (degree-derivable),
not steered in Stage 3. No reconciliation needed here.

### 4. Stage-3 objective omits kept schema-side targets
The distance function steers **motifs + path-templates** only. But the same-relation swap
also moves the object side — **inverse-CS size, two-step pairs, object co-occurrence /
row entropy** — all of which we **keep as targets** (not pinned by the lossy spectrum).
They are currently **unsteered**. Either add them to the Stage-3 objective or explicitly
mark them best-effort.

## Smaller fixes
- **Inputs** are stated as "blocks A and C"; Stage 1/2 also use multiplicity (**B**) and
  CS distributions (**D**). Real inputs: A, B, C, D.
- **Edge budget:** the entry point passes `num_triples`; our handle is **mean degree
  `E/V`** (size-stable). Fine to pass E, but parameterise/condition on mean degree.
- State explicitly which targets are **hit exactly** (counts, CS marginals), which are
  **approximate** (out-degree — largely fixed by the G2b CS-size offset, small residual),
  and which are **steered** (motifs).

## Scope clarification
The algorithm's "Stage 1" is **schema construction from a given target signature** — i.e.
our **doc-Stage-2** (signature → graph). The preliminary "load KG / compute features /
save / UMAP the manifold" steps are the **measurement phase** (the measurement plan). The
actual **doc-Stage-1 — sampling a *novel* signature from the real-graph population** (the
conditional-on-size model) is **not** in this algorithm; the manifold viz is its
groundwork. So this is the generator, not the population sampler.

## Net
- **Architecture: adopt it** — the 3-stage structure is the right one.
- **Must reconcile:** (1) multiplicity-α primary, functionality derived; (2) out-degree —
  use the **G2b CS-size offset** in Stage 2 (sample multiplicity conditioned on `cs_size`),
  plus the per-relation subject-multiplicity check. Stars resolved: non-induced / CS-fixed
  → dropped, not steered.
- **Good as-is:** P3 CS-first, PA for in-degree, degree-preserving SA for motifs, local
  motif deltas + stratified sampling, "start tiny," KS/KL/relative-error validation.

---

## Future work & best-effort limitations

Things the document (and our design) leave **best-effort** or unaddressed — to revisit
once the core pipeline runs on small graphs. None block a first version; each is a known
fidelity gap.

### Targets the generator does *not* steer (set at Stage 2, drift afterwards)
The Stage-3 objective steers only **motifs + path/templates** (per the doc). The
degree-preserving swap **changes the object side**, so these **kept targets drift
uncontrolled**:
- **inverse-CS size**, **row entropy**, **co-occurrence density** — not in any objective;
  whatever Stage 2 produces stands. *Future:* add them to the Stage-3 distance, or a
  schema-aware move, if their drift proves material.

### Per-relation subject-multiplicity is only approximated
PA shapes **aggregate** in-degree with a single exponent; the **per-relation**
subject-multiplicity (Block B target) emerges from PA rather than being hit, and the
degree-preserving swap can't fix it (it preserves multiplicities). *Future:* a
bipartite/configuration-style wiring per relation, or a correction pass, to target the
subject-multiplicity marginal directly. (Investigation §4.)

### Out-degree residual after G2b
The **CS-size offset (a)** captures the dominant CS-size↔multiplicity correlation, but
the **relation-identity-within-CS** residual remains (which specific relations a subject
bundles). *Future:* the type-conditioned multiplicity **option (b)** (`mult(r|t)`,
low-rank like `P(r|t)`), worthwhile when schema regularity / per-type fidelity matters.

### Type-light schema
We store `P(r|t)`'s spectrum but the generator's types are **synthetic co-occurrence
clusters**, not semantic types (random `U`, `V`). Fine for aggregate co-occurrence;
**per-type query selectivity** is only approximate. *Future:* tie types to real profiles
if per-type cardinality fidelity is needed.

### Measurement gap — depth-3 tree templates
The spec wants rooted trees of **depth 2 and 3**; the code computes **depth-2 only**.
*Future:* add the depth-3 tree-template walk/stats.

### Other deferred / empirical knobs
- **G2c per-relation joint** — default marginals; promote only if measured
  `(freq, obj-α, subj-α)` correlations are material.
- **`I(R;T)` scalar** — omitted (not in the doc); add as the "is option (b) worth it?"
  indicator if/when type-conditioning is considered.
- **Per-type relation entropy as a rank curve** — kept, but **over-specified** under the
  chosen spectrum-only `P(r|t)` construction (the rank is not consumed); revisit if the
  construction changes to entropy-direct rows, else it can fall back to a value
  distribution.
- **Generator rewrite scope** — full refactor vs compatibility shim (implementation).
