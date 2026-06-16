# Documentation

Map of this project's docs. **Implemented** documentation lives at the top level;
**future plans** in [`plan/`](plan/); **notes** (analyses, observations, assumptions,
build records) in [`notes/`](notes/).

## Implemented

- **[signature.md](signature.md)** — the reduced, non-over-determined signature
  (`src/signature_reduced/`): the implemented module reference **and** the reasoning behind
  it (non-over-determination, the derivability criterion, the multiplicity↔degree
  investigation, and the per-group G0–G6 justification). Start here.
- **[generator.md](generator.md)** — the `kgsynth` generator (`src/generator/`): the three-stage
  algorithm (schema sampler → CS-first instantiation → Maslov–Sneppen refinement) step by step,
  which signature fields drive each step, the reduced-signature adapters, and the evolution/fixes
  (P(r\|t) de-conflation, per-relation multiplicity-then-PA with edge conservation, the
  realizability cap, and the `num_distinct_cs` fixes).
- **[block-refactoring-guide.md](block-refactoring-guide.md)** — the `SignatureBlock` class
  pattern shared by every block (lifecycle methods, the `_NOT_CALCULATED` sentinel,
  property guards, `visualize` split, logging conventions, selective block computation).

Two signatures coexist: the original full signature (`src/signature/`,
`scripts/measure_signature.py` → `sig_out/`) and the reduced one
(`src/signature_reduced/`, `scripts/measure_signature_reduced.py` → `data/graphs/<name>/signature/`).
`scripts/measure_all_raw.py [--reduced]` runs either over all raw KGs.

## Plans (future)

- **[plan/generation_implementation_plan.md](plan/generation_implementation_plan.md)** —
  the three-stage sampler against the reduced signature. **Implemented** in the
  `src/generator/` package (Stage 1 schema, Stage 2 CS-first instantiation, Stage 3 motif
  refinement — all in scope now that reduced Block E exists); see the status note atop that plan.
- **[plan/stage1_population_sampler.md](plan/stage1_population_sampler.md)** — the
  **doc-Stage-1 population sampler**: sampling a *novel* signature from the real-graph
  population (conditional-on-size). Evaluates the data-expansion proposals (more KGs, WCC
  splitting, subgraph cutting, size conditioning, component grouping), the p ≫ n reality of
  the current 6 measurements, and a recommended scaling-law pipeline. Blocked on acquiring
  more (esp. typed) real KGs.

## Notes

- **[notes/signature_observations.md](notes/signature_observations.md)** — empirical
  observations from the signature distribution plots; the basis for which distribution
  family each quantity is fit with.
- **[notes/assumptions.md](notes/assumptions.md)** — per-block measurement assumptions and
  the reasoning behind each measurement choice (literal handling, CS/co-occurrence
  definitions, sampling, …).
- **[notes/generation_algorithm_fit.md](notes/generation_algorithm_fit.md)** — analysis of
  how the spec's three-stage generation algorithm maps onto the reduced signature, the
  reconciliations needed, and the best-effort gaps (future work).
- **[notes/signature_measurement_plan.md](notes/signature_measurement_plan.md)** — the
  build record / plan for the reduced signature (completed; realised as a coexisting
  module). Its realised-module summary is mirrored in [signature.md](signature.md).
- **[notes/lod_laundromat_acquisition.md](notes/lod_laundromat_acquisition.md)** —
  evaluation of LOD-a-lot / LOD Laundromat as a doc-Stage-1 data source (plan §3b):
  splits the merged single-graph LOD-a-lot from the ~650 K-document LOD Laundromat, the
  per-document meta-dataset find, and the case against using it for the population fit
  (document ≠ KG, laundered structure, wrong population). Recommends meta-dataset for
  exploration/validation only; acquire fit rows from the named typed §3b sources.
- **[notes/data_source_evaluation.md](notes/data_source_evaluation.md)** — evaluation of the
  *named* §3b sources (Bio2RDF, DBpedia, YAGO, DBLP, GeoNames, OGB, PrimeKG, …) as
  population draws, split by the two sub-goals they serve — the type-block gate (needs
  rich-typed sources) vs the non-type spread of 58 features (any real KG, typed or not).
  Tiered verdict + acquisition order; Bio2RDF is the top typed-gate source.
