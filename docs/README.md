# Documentation

Map of this project's docs — every file under `docs/` is listed here.

| Section | What lives there |
|---|---|
| **Reference** (top level) | The implemented system: what it measures, how it generates. |
| [`notes/`](notes/) | Investigations: analyses, empirical observations, corpus surveys. |
| [`plan/`](plan/) | Forward-looking work that is **not** implemented yet. |
| [`archive/`](archive/) | Superseded / historical records. Do not treat as current. |

## Reference — the implemented system

- **[signature.md](signature.md)** — the reduced, non-over-determined signature
  (`src/kgsynth/signature/`): the implemented module reference **and** the reasoning behind
  it (non-over-determination, the derivability criterion, the multiplicity↔degree
  investigation, and the per-group G0–G6 justification). Start here. Includes a
  [Deviations from the proposal](signature.md#deviations-from-the-proposal) table.
- **[generator.md](generator.md)** — the `kgsynth` generator (`src/kgsynth/generator/`): the three-stage
  algorithm (schema sampler → CS-first instantiation → Maslov–Sneppen refinement) step by step,
  which signature fields drive each step, the reduced-signature adapters, and the design rationale
  behind the wiring (P(r\|t) kept separate from the co-occurrence spectrum, per-relation
  multiplicity with edge conservation, the realizability cap, and the `num_distinct_cs` templating).
- **[transform.md](transform.md)** — signature transforms (`src/kgsynth/transform/`): seeded maps over
  the flat feature dict that perturb a measured signature before generating from it. Covers the
  **perturbation surface** (only 74 of the 124 features are read by the generator; the rest are
  no-ops), the features that are read but *inert* or *pinned constants*, the coupled quantile groups
  that must move together, and why clamped perturbations are reported rather than swallowed.
- **[block-refactoring-guide.md](block-refactoring-guide.md)** — the `SignatureBlock` class
  pattern shared by every block (lifecycle methods, the `_NOT_CALCULATED` sentinel,
  property guards, `visualize` split, logging conventions, selective block computation).
- **[report_outline.md](report_outline.md)** — outline for the written report: pipeline
  architecture, the signature concept, and the evaluation/validation methodology. Stage 1/2/3
  algorithmic detail is deliberately out of scope there and lives in [generator.md](generator.md).

Measuring a graph writes a `signature/` directory next to it (`data/graphs/<name>/signature/`),
via the `kgsynth measure` CLI or its script wrapper. See [`scripts/README.md`](../scripts/README.md)
for the measurement, sweep and plotting tooling.

## Notes — investigations

- **[notes/signature_observations.md](notes/signature_observations.md)** — empirical
  observations from the signature distribution plots; the basis for which distribution
  family each quantity is fit with.
- **[notes/assumptions.md](notes/assumptions.md)** — per-block measurement assumptions and
  the reasoning behind each measurement choice (literal handling, CS/co-occurrence
  definitions, sampling, …).
- **[notes/signature_size_dependence.md](notes/signature_size_dependence.md)** — which of the
  124 features are **extensive** (scale with graph size) and which are **intensive** (size-free).
  The split the Stage-1 conditional-on-size model needs.
- **[notes/counter_benchmark.md](notes/counter_benchmark.md)** — the exact-vs-colour-coding
  motif-counter comparison per motif size, and the CC counter's adaptive sample-size feature.
  Data collected by `scripts/cc_variance.py`.
- **[notes/generation_algorithm_fit.md](notes/generation_algorithm_fit.md)** — analysis of
  how the spec's three-stage generation algorithm maps onto the reduced signature, the
  reconciliations needed, and the best-effort gaps (future work).
- **[notes/stage3_steering_analysis.md](notes/stage3_steering_analysis.md)** — why Stage 3
  is slow on hub-heavy graphs (`fb237_v4`) and why per-swap motif steering barely moves the
  loss. Delta-cost profiling (6-cycle delta ≥94 %), the node-level/endpoint degree guards and
  MITM cycle enumerator, the SA-schedule retune (was a random walk; now anneals) and its
  per-graph caveat, per-proposal swap logging + hub leverage, the rejected "approximate hub
  delta" idea, and the central finding: small `|d_loss|` is **both** scale (million-size
  targets cap the move at ~3e-4) **and** cancellation (Stage-2 overshoots paw/c5 ~2× while
  undershooting c4/k4, so correlated motif deltas oppose; median alignment 0.48). Points to
  the Stage-2 paw/c5 overshoot as the highest-value upstream lever.
- **[notes/motif_reachability_and_edge_multiplicity.md](notes/motif_reachability_and_edge_multiplicity.md)** —
  the follow-through on that lever: why Stage-3 *cannot* reach the fb237-class motif targets.
  The per-swap motif coupling is nearly 1-D (clustering ↔ chordless-cycle axis); the targets
  live on the original degree sequence but Stage-3 is locked to Stage-2's (27.5 % L1 off); and
  the root cause is that Stage-2 produces ~zero **edge multiplicity** (pair overlap) — ρ≈1 vs
  originals 1.03–2.0 across the whole corpus — inflating the simple graph +26 % and driving the
  paw/c5 overshoot. The signature never encodes pair overlap (feature audit); proposed fix is a
  third, pair-level relation co-occurrence (`pair_cooc`) whose density is the missing handle.
  Corpus survey via `scripts/edge_multiplicity.py`.
- **[notes/relation_reciprocity_and_bidirectionality.md](notes/relation_reciprocity_and_bidirectionality.md)** —
  characterises *where* bidirectionality comes from (the harder half of the multiplicity gap) and
  documents the implemented fix. Reciprocity is nearly **bimodal** per relation (symmetric ≈1 /
  asymmetric ≈0, ~0% in between — aids fully symmetric, hetionet fully asymmetric, dbpedia the
  partial exception, swdf cross-relational) and strongly tied to relation **frequency**. Realising
  it needed four independent Stage-2 factors fixed together (CS-pool overlap, stub reservation,
  mutual-pair reuse, frequency-binned reciprocity assignment) — a same-relation swap provably
  cannot fix this post-hoc, since it preserves per-relation degree exactly. Measured result:
  simple-edge inflation cut roughly in half on fb237/wn18rr/aids (e.g. wn18rr +45%→+24%);
  bidirectional attainment ~45–50% of target, capped by a genuine stub-multiplicity ceiling
  (documented, not further chased). Survey via `scripts/relation_reciprocity.py`.
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

## Plans — not implemented

- **[plan/stage1_population_sampler.md](plan/stage1_population_sampler.md)** — the
  **doc-Stage-1 population sampler**: sampling a *novel* signature from the real-graph
  population (conditional-on-size). Evaluates the data-expansion proposals (more KGs, WCC
  splitting, subgraph cutting, size conditioning, component grouping), the p ≫ n reality of
  the current 6 measurements, and a recommended scaling-law pipeline. Blocked on acquiring
  more (esp. typed) real KGs.

## Archive — historical, superseded

Kept as a record of how the work was scoped and what was tried. **Their numbers, file paths
and APIs are out of date**; the live references are [signature.md](signature.md) and
[generator.md](generator.md).

- **[archive/signature_measurement_plan.md](archive/signature_measurement_plan.md)** — the
  original build plan for the reduced signature (69 features, skew-normal fits, Block E deferred).
  Superseded: the signature now has 124 features with quantile functions.
- **[archive/generation_implementation_plan.md](archive/generation_implementation_plan.md)** —
  the original implementation plan for the three-stage sampler, now realised in
  `src/kgsynth/generator/`.
- **[archive/path_length_steering.md](archive/path_length_steering.md)** — the Stage-2
  path-length steering analysis. The implementation it describes was **removed** (one-sided and
  disabled); retained for its root-cause analysis and the Stage-3-loss-term option, which remains
  the way forward if path targeting is ever revisited.
- **[archive/submission_cleanup_plan.md](archive/submission_cleanup_plan.md)** — the plan that
  packaged the repo as installable `kgsynth` (pyproject, CLI, examples, README). Essentially
  complete; the one open item is the deferred corpus re-measurement.
