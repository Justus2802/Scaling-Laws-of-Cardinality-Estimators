# Plan: implement the three-stage sampler against the reduced signature

> **Status (implemented, differs from the original plan below).** Rather than a separate
> `src/generator_reduced.py`, the existing generator was converted in place to consume the
> reduced blocks and split into a package: `src/generator/` (`schema`, `stage1`, `stage2`,
> `stage3`, `pipeline`, plus `_adapters` for the reduced-signature reconstructions). The
> public API (`Schema`, `sample_schema`, `instantiate`, `refine`, `Signature`, `Generator`)
> is unchanged and re-exported from `generator/__init__.py`. **Stage 3 is now in scope**:
> reduced Block E exists (the 7 raw motif counts + path/tree templates), so `refine()` steers
> triangle / 4-node-motif / assortativity targets as before. The reduced-block reads that
> lacked a direct attribute are reconstructed in `_adapters.py`: `num_triples = V·mean_degree`,
> functionality/inverse-functionality from the multiplicity-α skew-normal (`1/ζ(α)`),
> P(r|t) singular values from the `type_rel_spectrum_exp` exp-decay fit (its own T×R
> spectrum, no longer conflated with `M`), and `cs_size_mean` from the CS-size skew-normal.
> The inputs table and stage notes below remain accurate as reference.

Implements the project spec's three-stage generation algorithm
([../notes/generation_algorithm_fit.md](../notes/generation_algorithm_fit.md)) consuming a
reduced signature ([../signature.md](../signature.md),
[../notes/signature_measurement_plan.md](../notes/signature_measurement_plan.md)).

## Scope & decisions

- **Stages 1 + 2 now; Stage 3 deferred.** Stage 3 (Maslov–Sneppen degree-preserving
  rewiring + simulated annealing) steers **motif counts**, which the reduced signature
  defers (no Block E). It is left as a documented future pass; the kept reduced targets
  (degree, inverse-CS, two-step, row entropy) become **validation diagnostics**, not
  steered. (Confirmed.)
- **Full Stage-2 reconciliation:** per-relation multiplicity from the **skew-normal α**
  (scale fixed by edge conservation), the **`cs_size^a` offset** (G2b) for out-degree, and
  **preferential attachment** for in-degree. (Confirmed; matches
  generation_algorithm_fit.md §2.)
- This is the **generator** (signature → graph = the algorithm's "Stage 1/2"), **not** the
  population sampler (doc-Stage-1, sampling a *novel* signature). Literals/datatypes are
  out of scope (G6). Types are synthetic co-occurrence clusters (type-light, the known gap).

## Inputs — reduced signature → generation parameters

| Reduced field | Block | Role in generation |
|---|---|---|
| `num_entities` V, `num_relations` R, `mean_degree` | A | V (root), R (vocab), **E = mean_degree · V** (edge budget) |
| `relation_zipf_exponent` | B | relation-frequency weights (replaces the hard-coded Zipf 2.0) |
| `obj_mult_alpha_*` / `subj_mult_alpha_*` skew-normal | B | per-relation multiplicity **shape** α (sampled per relation, truncated to [lo,hi]) |
| `a_obj`, `a_subj` | B | **G2b** CS-size→multiplicity offset (out/in-degree shaping) |
| `num_classes` T, `class_size_alpha` | C | type count + type-size weights |
| `type_rel_spectrum_rate/scale` | C | **P(r\|t)** singular spectrum → low-rank factorisation |
| `num_distinct_cs`, `cs_freq_alpha`, `cs_size_*` skew-normal | D | CS template pool: how many distinct CSs, how skewed their reuse, how big |
| out/in-degree, inverse-CS, two-step, row-entropy, connectivity | B/C/D/F | **validation targets only** (not constructive) |

Derived, **not** sampled: functionality (head of the multiplicity law), per-relation
multiplicity **scale** (edge conservation, below), aggregate degree (emerges from PA +
G2b).

## Module layout (`src/generator_reduced.py`)

Reuse from `generator.py` by import where identical: `_zipf_weights`,
`_sample_type_relation_probs` (the low-rank P(r|t) factoriser — fed reconstructed singular
values), and the igraph-assembly idiom. New small helpers:
`reconstruct_exp_decay(fit, n)` (σ_k = `scale · exp(−rate·k)`), `sample_skewnorm_trunc(fit,
n, rng)` (scipy `skewnorm.rvs` clipped to `[lo, hi]`).

```
ReducedSchema            # dataclass: Stage-1 output
sample_schema_reduced(a, b, c, d, *, seed) -> ReducedSchema
instantiate_reduced(schema, *, v_noise, e_noise, pa_exponent, seed) -> igraph.Graph
ReducedGenerator(target: ReducedGraphSignature).sample(seed=…) -> igraph.Graph   # Stage1→Stage2
```

## Stage 1 — schema sampler (`sample_schema_reduced`)

Samples the free constructive params (G0–G3) into a `ReducedSchema`:

1. **Relations** — R synthetic URIs; `relation_weights = _zipf_weights(R, relation_zipf_exponent)`.
2. **Types** — T synthetic URIs; `type_weights = _zipf_weights(T, class_size_alpha)` (uniform
   fallback when α is NaN / T tiny).
3. **P(r\|t)** — reconstruct singular values from `type_rel_spectrum_exp`
   (`reconstruct_exp_decay`, length `min(T,R)`), then `type_relation_probs =
   _sample_type_relation_probs(T, R, relation_weights, σ_reconstructed, rng)`. This is the
   spectrum-only construction (resolves the old M-conflation: feed P(r\|t)'s **own** spectrum).
4. **Per-relation multiplicity α** — `alpha_obj[r] = sample_skewnorm_trunc(obj_mult_alpha, R)`
   and `alpha_subj[r]` likewise; one exponent per relation, truncated to the stored cutoffs.
5. **CS pool params** — keep `cs_size` skew-normal, `cs_freq_alpha`, `num_distinct_cs`,
   and `a_obj`/`a_subj` for Stage 2.
6. **Budget** — `E = round(mean_degree · V)`.

## Stage 2 — CS-first instantiation (`instantiate_reduced`)

P3 primitive: CS first, then per-relation multiplicity given the CS.

1. **Sample actual V, E** with Gaussian noise (as in the current `instantiate`); split off
   rdf:type edges; `content_E = E − n_type_edges`.
2. **Assign a type** to each entity via `type_weights`.
3. **Sample CS templates** — build `num_distinct_cs` templates per the type's P(r\|t); CS
   **size** per template ~ `cs_size` skew-normal (rounded, ≥1); assign entities to templates
   with Zipf(`cs_freq_alpha`) reuse weights (reuses the current template-mode logic, but
   sizes/reuse now come from the reduced params instead of mean/`cs_freq.alpha`).
4. **Wire content edges — multiplicity-then-PA, with edge conservation:** for each relation r,
   - subjects of r = entities whose CS contains r → `S_r`; `|edges_r| = round(relation_weight[r]
     · content_E)`.
   - base weight per subject `w_s = powerlaw_draw(alpha_obj[r]) · cs_size(s)^a_obj`  (G2 shape ×
     **G2b offset**); allocate `|edges_r|` edges across `S_r` by `multinomial(w/Σw)` →
     `m_obj(s,r)`. *This makes the per-subject multiplicity have tail shape α, reproduces the
     CS-size↔multiplicity correlation, and hits the exact edge budget — i.e. the scale is
     **derived** from `|edges_r|/|S_r|`, not stored.*
   - for each of the `m_obj(s,r)` edges, pick the **object** by preferential attachment
     (`in_degree^pa_exponent`, Laplace-smoothed) → shapes the **in-degree** toward the target.
5. **rdf:type edges** for typed entities; assemble the igraph graph (vertex/edge attributes
   matching `kg_io.load_kg`, so `compute_reduced_signature` can read it back).
6. **Throttle** content edges to `content_E` if over budget.

Functionality / aggregate out-degree are **not** set directly — they fall out of steps 3–4.

## Stage 3 — deferred (documented)

Degree-preserving Maslov–Sneppen swaps + SA toward **motif** targets need a Block E
measurement; out of scope here. `generator.py`'s `refine()` already implements the move and
can be reused later once Block E is added to the reduced signature (or composed alongside).
Until then `ReducedGenerator.sample` returns the Stage-2 graph. The unsteered kept targets
(inverse-CS, two-step, row entropy) are the best-effort gaps already logged in
[../notes/generation_algorithm_fit.md](../notes/generation_algorithm_fit.md) §"Future work".

## Tests (`tests/test_generator_reduced.py`)

- **Schema:** `sample_schema_reduced` on a measured reduced signature → R relations, T types,
  `type_relation_probs` rows sum to 1, per-relation α within `[lo, hi]`.
- **Instantiate:** output is a valid `igraph.Graph` with the `load_kg` attribute contract;
  |V|, |E| within noise of targets; rdf:type edges present iff T>0.
- **Round-trip / fidelity:** `compute_reduced_signature(generated)` is finite and lands near
  the target on the **constructive** params — relation Zipf exponent, num_classes, CS-size
  location, P(r\|t) spectrum rate — within tolerance (KS / relative error). Degree/inverse-CS
  are checked as *diagnostics* (looser bounds, since Stage 3 is deferred).
- **Edge conservation:** per relation, `Σ_s m_obj(s,r) == |edges_r|` after allocation.
- Use a small synthetic target (the `tests/` TTL fixture) and "start tiny" V.

## Verification

1. `.venv/bin/python -m pytest tests/test_generator_reduced.py -q`.
2. Measure a small real graph's reduced signature, generate, re-measure, and diff the
   `signature.json`s (constructive params close; degree diagnostics reasonable).
3. Confirm `generator.py` + the old signature still run unchanged (coexistence).

## Suggested build order

`reconstruct_exp_decay` / `sample_skewnorm_trunc` helpers → `ReducedSchema` +
`sample_schema_reduced` → `instantiate_reduced` (CS pool → multiplicity allocation → PA
objects → assembly) → `ReducedGenerator` → tests → fidelity check. One stage per commit.
