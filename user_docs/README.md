# User docs

Documentation for using `kgsynth` as a Python library or CLI: what it measures, how it generates,
and how to call it. If you're contributing to the package itself, or want the design rationale,
empirical investigations and forward-looking plans behind these decisions, see
[`../developer_docs/`](../developer_docs/) instead — nothing here is required reading to just use
the package.

## Start here

- **[api.md](api.md)** — API reference for `kgsynth` as a Python library and CLI: every
  public class/function by subpackage (`Signature`, `Generator`, blocks, transforms,
  `kgsynth.dataset`, `kgsynth.corpus`, `kg_io`, `motif_counter`, `signature_sampler`), with
  signatures, parameters and minimal call examples. Start here for "how do I call X"; the
  other docs below cover the reasoning behind each subsystem.

## Reference — the implemented system

- **[signature.md](signature.md)** — the reduced, non-over-determined signature
  (`src/kgsynth/signature/`): the implemented module reference **and** the reasoning behind
  it (non-over-determination, the derivability criterion, the multiplicity↔degree
  investigation, and the per-group G0–G6 justification). Includes a
  [Deviations from the proposal](signature.md#deviations-from-the-proposal) table.
- **[generator.md](generator.md)** — the `kgsynth` generator (`src/kgsynth/generator/`): the three-stage
  algorithm (schema sampler → CS-first instantiation → Maslov–Sneppen refinement) step by step,
  which signature fields drive each step, the reduced-signature adapters, and the design rationale
  behind the wiring (P(r\|t) kept separate from the co-occurrence spectrum, per-relation
  multiplicity with edge conservation, the realizability cap, and the `num_distinct_cs` templating).
- **[dataset.md](dataset.md)** — the perturb-and-generate pipeline (`src/kgsynth/dataset/`,
  `kgsynth dataset`): generate many synthetic KGs from one measured signature, one process per graph.
  Covers the two designs (**joint** jitter for corpus-building, **OFAT** for a sensitivity analysis
  over the signature's features), the YAML config, the resumable output layout, and the two effects
  that will mislead a naive reading of the results (clamped perturbations, and Block E's estimator
  variance).
- **[transform.md](transform.md)** — signature transforms (`src/kgsynth/transform/`): seeded maps over
  the flat feature dict that perturb a measured signature before generating from it. Covers the
  **perturbation surface** (only 87 of the 135 features are read by the generator; the rest are
  no-ops), the features that are read but *inert* or *pinned constants*, the coupled quantile groups
  that must move together, and why clamped perturbations are reported rather than swallowed.

Measuring a graph writes a `signature/` directory next to it (`data/graphs/<name>/signature/`),
via the `kgsynth measure` CLI or its script wrapper. See [`scripts/README.md`](../scripts/README.md)
for the measurement, sweep and plotting tooling, and [`examples/`](../examples/) for minimal
runnable scripts covering the measure → generate → compare loop.
