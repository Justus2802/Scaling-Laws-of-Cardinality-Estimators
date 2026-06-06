# Documentation

Map of this project's docs. **Implemented** documentation lives at the top level;
**future plans** in [`plan/`](plan/); **notes** (analyses, observations, assumptions,
build records) in [`notes/`](notes/).

## Implemented

- **[signature.md](signature.md)** — the reduced, non-over-determined signature
  (`src/signature_reduced/`): the implemented module reference **and** the reasoning behind
  it (non-over-determination, the derivability criterion, the multiplicity↔degree
  investigation, and the per-group G0–G6 justification). Start here.
- **[block-refactoring-guide.md](block-refactoring-guide.md)** — the `SignatureBlock` class
  pattern shared by every block (lifecycle methods, the `_NOT_CALCULATED` sentinel,
  property guards, `visualize` split, logging conventions, selective block computation).

Two signatures coexist: the original full signature (`src/signature/`,
`scripts/measure_signature.py` → `sig_out/`) and the reduced one
(`src/signature_reduced/`, `scripts/measure_signature_reduced.py` → `sig_out_reduced/`).
`scripts/measure_all_raw.py [--reduced]` runs either over all raw KGs.

## Plans (future)

- **[plan/generation_implementation_plan.md](plan/generation_implementation_plan.md)** —
  implementing the three-stage sampler (`src/generator_reduced.py`) against the reduced
  signature: Stage 1 (schema) + Stage 2 (CS-first instantiation) now, Stage 3 (motif
  refinement) deferred.

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
