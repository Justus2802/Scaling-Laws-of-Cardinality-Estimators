# Developer docs

Internal documentation: how to extend the package, the empirical investigations and design
decisions behind the implementation, forward-looking plans, and the course-submission artifacts
this project started as. None of this is required reading to *use* `kgsynth` — see
[`../user_docs/`](../user_docs/) for that; start there unless you're contributing to the package or
digging into why it works the way it does.

| Section | What lives there |
|---|---|
| **Contributing** | How to extend the package (add a signature block, a transform, …). |
| [`notes/`](notes/) | Investigations: analyses, empirical observations, corpus surveys. |
| [`plan/`](plan/) | Design plans — some implemented (kept as the diagnosis that motivated the change), some not; each file states its own status. |
| **Course artifacts** | The academic-submission material this package was built for. Not user documentation. |

## Contributing

The `SignatureBlock` class-pattern guide (lifecycle methods, the `_NOT_CALCULATED` sentinel,
property guards, `visualize` split, logging conventions, selective block computation) has since
been pruned from this tree; git history has it.

## Notes — investigations

- **[notes/signature_observations.md](notes/signature_observations.md)** — empirical
  observations from the signature distribution plots; the basis for which distribution
  family each quantity is fit with.
- **[notes/assumptions.md](notes/assumptions.md)** — per-block measurement assumptions and
  the reasoning behind each measurement choice (literal handling, CS/co-occurrence
  definitions, sampling, …).
- **[notes/counter_benchmark.md](notes/counter_benchmark.md)** — the exact-vs-colour-coding
  motif-counter comparison per motif size, and the CC counter's adaptive sample-size feature.
  Data collected by `scripts/cc_variance.py`.
  (The size-dependence split — which of the 135 features are extensive vs. intensive — and the
  generation-algorithm-fit analysis have since been pruned from this tree as course scratch work;
  git history has them.)
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
- **[notes/data_source_evaluation.md](notes/data_source_evaluation.md)** — evaluation of the
  *named* §3b sources (Bio2RDF, DBpedia, YAGO, DBLP, GeoNames, OGB, PrimeKG, …) as
  population draws, split by the two sub-goals they serve — the type-block gate (needs
  rich-typed sources) vs the non-type spread of features (any real KG, typed or not).
  Tiered verdict + acquisition order; Bio2RDF is the top typed-gate source. The companion
  LOD Laundromat / LOD-a-lot evaluation has since been pruned from this tree — see
  [`plan/stage1_population_sampler.md`](plan/stage1_population_sampler.md) §"Diversity at
  scale" for its summary.

## Plans — design rationale for each change, implemented or not

Each file states its own status (`Status: ...` near the top) — treat that line as authoritative,
not the section heading here.

- **[plan/per_relation_stub_balance.md](plan/per_relation_stub_balance.md)** — **implemented.**
  Replaced the per-relation stub allocation's greedy quota (which drew each side's stubs
  independently and reconciled the mismatch with an ad hoc cap) with a joint IPF allocation that
  fits both sides to the same row/column margins by construction. As-built design and results now
  live in [`user_docs/generator.md`](../user_docs/generator.md#the-ipf-stub-allocation); this file
  is kept as the diagnosis that motivated the change.
- **[plan/powerlaw_truncation.md](plan/powerlaw_truncation.md)** — **executed.** Makes every power
  law in the signature consistently truncated, both at fit time and at sample time, instead of
  fitting unbounded and sampling bounded (or vice versa).
- **[plan/remove_unnecessary_fallbacks.md](plan/remove_unnecessary_fallbacks.md)** — **planned, not
  started.** Removing the generator's degraded-mode code paths (e.g. "Block D absent") that handle
  signature data which is in practice always measurable on a real graph.
- **[plan/degree_budget_and_type_edges.md](plan/degree_budget_and_type_edges.md)** — **implemented**
  (see `CHANGELOG.md`'s "degrees: rdf:type out of the measurement" entry). Excludes rdf:type edges
  and the class nodes they point at from the degree measurement, and makes the Stage-1 degree-sum
  repair two-sided so `Σ out == Σ in` by construction.
- **[plan/cs_freq_concentration.md](plan/cs_freq_concentration.md)** — **proposed** (diagnosis
  complete, fix not yet implemented). The `cs_freq` / `num_distinct_cs` concentration defect that
  the degree-W1 truncation fix (above) had been masking as a measurement artifact.
- **[plan/stage1_population_sampler.md](plan/stage1_population_sampler.md)** — **planned, blocked
  on data.** The **doc-Stage-1 population sampler**: sampling a *novel* signature from the
  real-graph population (conditional-on-size). Evaluates the data-expansion proposals (more KGs,
  WCC splitting, subgraph cutting, size conditioning, component grouping), the p ≫ n reality of the
  current measurements, and a recommended scaling-law pipeline. Blocked on acquiring more
  (especially typed) real KGs.

## Course artifacts

The academic-submission scaffold this package was built for. Not user documentation and not
guaranteed to reflect the current implementation. The academic proposal PDF, the report outline,
the scratch TODO list, and the raw report draft notes have since been pruned from this tree ahead
of making the repo public; git history has them.
