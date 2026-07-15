# Report Outline — Approach & Evaluation (excl. generator internals)

Scope note: this outline covers everything **around** the graph generator's internal
rewiring mechanics — the pipeline architecture, the signature concept, how the signature
is calculated, and the evaluation/validation methodology. Stage 1/2/3 algorithmic detail
(schema sampling formulas, CS-first wiring, Maslov–Sneppen SA loss terms) is deliberately
out of scope here and belongs in a separate "generator algorithm" section/report, backed
by `user_docs/generator.md`.

---

## 1. Introduction / Motivation

- Research question this repo serves: scaling laws of learned KG cardinality estimators
  (project tasks T1–T4; this repo = T1+T2). Source: `README.md` §"The research question",
  `MEMORY.md` project overview.
- The core problem the *approach* solves: real labeled KG corpora are small and hard to
  get in bulk → need a way to generate *numerous, diverse, statistically realistic*
  synthetic KGs on demand.
- The central idea, stated once, used everywhere after: **measure → generate → compare**
  (`README.md` diagram). Introduce the vocabulary here: *signature*, *roundtrip*,
  *target* vs *synthetic*.

## 2. The Signature Concept

- What a "signature" is in this project's domain language: a compact, **non-redundant**
  numerical fingerprint of a KG (134 features) — not a full copy of the graph, not a
  naive dump of every possible statistic.
- Why not "just record every statistic": KG statistics are highly inter-dependent (e.g.
  degree = sum of per-relation multiplicities), so a naive feature set is
  **over-determined**. Explain the two kinds of redundancy the project explicitly
  designs against (`user_docs/signature.md` §Goal):
  1. Algebraic redundancy (`density = E/V²`, etc.)
  2. Cross-statistic redundancy (e.g. `functionality` = head of the multiplicity law)
- The **derivability criterion** used to decide what to keep vs. drop (`user_docs/signature.md`
  §"Derivability criterion"): a quantity may be dropped only if it's an *exact* function
  of already-stored values with no unstored joint/correlation entering — "derivable under
  an independence assumption" is explicitly *not* sufficient. Worth a figure/example:
  out-degree as a compound sum over CS × per-relation multiplicity.
- Consequence: some quantities that *look* derivable (aggregate degree, inverse-CS size,
  row entropy) are **kept as free targets** because they cross into unstored joint
  structure — this is the conceptual core that justifies the whole Block A–F design.
- Representation conventions (`user_docs/signature.md` §"Convention"): the signature does not
  store raw moments (mean/std/median) but the **parameters of a distribution family**
  chosen per-quantity from empirical observation (`developer_docs/notes/signature_observations.md`):
  quantile functions, power-laws, Zipf, exponential-decay rank curves, truncated
  power-laws, and a few plain summary scalars (Block F shortest-path max/mean/var).
  Explain *why* (regenerates the shape; moments don't) and give
  the reading-guide for each family (what a large vs. small exponent/rate means).

## 3. Signature Calculation — the `src/kgsynth/signature/` Module

- Architecture: six independent measurement **blocks**, `BlockA`…`BlockF`, each one class
  in `block_<x>.py`, sharing a common `SignatureBlock` ABC (`src/kgsynth/signature/_block_base.py`):
  lifecycle (`calculate` → `as_vector`/`as_dict` → `visualize`), the `_NOT_CALCULATED`
  sentinel guarding uncomputed access, and a serialization round-trip
  (`to_serializable`/`from_serializable`) preserving numpy arrays and fit tuples through
  JSON. Reference: `developer_docs/block-refactoring-guide.md`.
- Table of blocks, what each measures, and vector length (134 total) — pull directly from
  `user_docs/signature.md`'s block table:
  - **A** (G0, 3 features) — size & vocabulary root parameters: `num_entities`,
    `num_relations`, mean degree `E/V`.
  - **B** (G1/G2/G2b, 33 features) — relation-usage Zipf skew; per-relation
    object/subject multiplicity tail shape as **quantile functions**; CS-size↔multiplicity
    offsets (`a_obj`, `a_subj`); degree-tail steering targets (`out/in_degree_max/p90`);
    per-relation reciprocity (6 frequency-binned fractions + a symmetric-mode value).
  - **C** (G3, 29 features) — schema/type structure: class-size power-law; subject/object
    relation co-occurrence spectra (exp-decay) + density; edge-multiplicity and
    bidirectional-ratio scalars; row-entropy quantile functions; `P(r|t)` spectrum;
    per-type entropy.
  - **D** (G3, 25 features) — characteristic-set (CS) structure: `num_distinct_cs` +
    CS-frequency (truncated power-law) + CS-size (quantile function), mirrored for the
    inverse (object) side; two-step path-count truncated power-law.
  - **E** (G5, 27 features) — motif counts (triangle, 4-/5-/6-cycle, diamond, K4, tailed
    triangle), path-template Zipf+entropy (k=2..10), tree-template Zipf+entropy.
  - **F** (G4, 7 features) — connectivity: component count, LCC fraction, average local
    clustering, degree assortativity, shortest-path max/mean/var summary.
- The compute entry point: `compute_reduced_signature(path, blocks=...)`
  (`src/kgsynth/signature/__init__.py`) loads a `.ttl`/`.nt` file via `kg_io.load_kg`, runs the
  selected blocks, returns a `ReducedGraphSignature` dataclass; NaN-fills any block not
  computed so the feature vector is always fixed-length (`as_vector`/`as_dict`).
- Output artifacts: `write_signature_outputs` — per-block plot (`block_<x>.png`),
  per-block full-state JSON, a combined `signature.json` (name→value), and a text
  `summary.txt`. Same layout used for both measured real graphs and re-measured synthetic
  ones, which is what makes them directly comparable (§5).
- Worth a short example walkthrough of one simple block (Block A,
  `src/kgsynth/signature/block_a.py`) end-to-end — calculate → vector → visualize — as a concrete
  illustration of the shared pattern before summarizing the more complex blocks (B–F) at a
  higher level.
- Distribution-fitting machinery: `_fits.py` (quantile fits, exponential-decay-rank fits,
  truncated power-law, Zipf — via `scipy.stats` and the `powerlaw` package) and
  `_plot_helpers.py` (overlay functions pairing each fit back onto its raw pre-fit data
  for the diagnostic plots).
- Motif-counting backend feeding Block E: `src/kgsynth/motif_counter/` — `_base.py` (ABC),
  `exact_motif_counter.py` (exact triangle/4-node/ESCAPE 5-node counting),
  `cc_motif_counter.py` (color-coding sampler, Bressan et al. 2021, for large/dense
  graphs), `hybrid_motif_counter.py` (exact for k≤3, CC for k≥4 — the `MOTIF_COUNTER`
  Block E actually uses, so every motif count above a triangle is an estimate). Worth a short
  "accuracy vs. tractability" note here, expanded on in §5.5's counter-benchmark item —
  since every motif feature the signature and the evaluation depend on is only as
  trustworthy as this counting layer.
- Design rationale deep-dives worth citing/summarizing (not full detail, but the headline
  findings): `developer_docs/notes/assumptions.md` (per-block measurement choices, e.g. literal
  exclusion), `developer_docs/notes/signature_size_dependence.md` (which of the 134 features scale
  with graph size vs. are size-free — relevant to any cross-graph comparison).

## 4. The Generation Pipeline (architecture only, not algorithm internals)

- Public API surface and orchestration: `src/kgsynth/generator/pipeline.py` —
  `Signature` (target-loading wrapper around Blocks A–F) + `Generator.sample()`. One
  seeded call (`Generator(Signature.from_file(...)).sample(seed=42, rewire_budget=...)`)
  derives sub-seeds for each stage (`seed`, `seed+1`, `seed+2`) so the whole pipeline is
  reproducible from one integer (`user_docs/generator.md` intro).
- Module map (table form, from `user_docs/generator.md`):
  - `schema.py` — `Schema` dataclass, the Stage-1→Stage-2 handoff object.
  - `stage1.py` / `stage2.py` / `stage3.py` — the three stages (mention *what* each stage
    is responsible for at the architecture level only: abstract schema → concrete wiring
    → motif/connectivity refinement; do not detail the sampling formulas or SA loss terms
    here — reference `user_docs/generator.md` for that).
  - `_adapters.py` — reduced-signature reconstruction helpers (turning stored distribution
    *parameters* back into samples/means the stages need), since the signature stores
    fitted parameters, not raw arrays.
  - `_logging.py` — package-wide progress logging.
- Which signature block feeds which stage — the input-routing table
  (`user_docs/generator.md` §"Inputs — which signature fields drive generation") is the key
  diagram for this section: it shows the pipeline is *driven end-to-end by the signature*,
  not by hand-picked heuristics. Also list the **validation-only** fields (measured but
  deliberately not used constructively: co-occurrence density/row-entropy,
  `clustering_coefficient`, `shortest_path_var`) — these matter for §5 (evaluation),
  since they're diagnostics the roundtrip checks but the generator never targets.
- Required vs. optional blocks: `Signature` requires A, C, E; B/D/F are optional and each
  unlocks more faithful structure when present (graceful degradation story).

## 5. Evaluation Methodology

### 5.1 The roundtrip — the core validation loop

- `scripts/signature_roundtrip.py`: load a cached/measured target signature → generate a
  synthetic graph → re-measure its signature with the same Block code → print a
  per-block, per-feature relative-error comparison. This is described in `README.md` as
  *the* main way progress is validated — frame it as the evaluation backbone the rest of
  the toolchain builds on.
- Practical details worth reporting: synthetic re-measurement uses a reduced Block-E
  sampling budget (`_FINAL_SAMPLE_BUDGET = 20_000`) vs. the 100k default, to keep
  roundtrips fast; every output is timestamped so repeated runs don't clobber each other;
  outputs mirror the real-graph `signature/` directory layout
  (`signature_synth_<timestamp>/`) so measured and generated signatures are structurally
  identical and diff-able.
- The aggregate metric used for reporting: **median relative error** across features
  (mean/max are explicitly noted as inflated by near-zero-target features — worth a
  methodological callout: don't just report a mean).

### 5.2 Distribution-level comparison — Wasserstein distance

- `src/kgsynth/signature/_distance.py`: comparing fitted distribution *parameters* directly is
  misleading (an unstable shape parameter can explode relative error even when the
  underlying distributions agree) — so the roundtrip instead reconstructs a representative
  sample from each fit (via inverse-CDF / shared common random numbers so identical fits
  give exactly 0 distance) and computes the **Wasserstein-1 distance** between target and
  synthetic samples. Also computes `reconstructed_iqr` to scale-normalize W1 across
  features with different units/scales. This is a nice, self-contained "we thought
  carefully about the right comparison metric" subsection.

### 5.3 Per-feature error visualization

- `scripts/signature_error_boxplot.py` — boxplot of per-feature/per-distribution relative
  errors grouped by signature block; the artifact behind
  `data/graph_population/signature_error_boxplot_wn18rr_v4.png` seen in the working tree.
- `scripts/convergence_plot.py` (incl. `--grid`) — plot Stage-3's
  per-swap-logged relative-error trajectory for chosen features (or a fixed 2×2 grid of
  triangle/diamond/c6/paw) over the SA run; used to visually confirm convergence and
  compare fixed- vs. adaptive-weight runs. Backing data: the `convergence_log` CSVs
  produced by `stage3.refine(...)`, stored under `experiments/convergence_logs/`.
- `scripts/swap_delta_viz.py` — visualizes the per-swap-proposal delta log (one row per
  evaluated swap: pre-swap degrees, per-motif deltas, Δloss, accept/reject) — used to
  study *where* the optimizer's leverage comes from, not just whether it converges.

### 5.4 Population-level / PCA comparison

- The problem PCA analysis addresses: a single signature vector (134 numbers) has no
  meaningful "shape" in isolation — comparing target vs. synthetic only makes sense
  relative to the spread of real KGs (`scripts/plot_signature_pca.py` docstring).
- `scripts/plot_signature_pca.py`: fits a 2D PCA basis on the **corpus** of real measured
  KGs (`_fit_pca_2d` — mean-impute, z-score, SVD), then projects a target graph and its
  roundtrip synthetic counterpart into that space for a visual "did we land in the right
  neighborhood" check. Includes a `--size-agnostic` mode that fits PCA on scale-free
  structural features only (excludes raw counts/degree extrema), since size otherwise
  dominates the corpus variance and swamps structural differences. Artifacts:
  `data/graph_population/signature_pca.png`, `signature_pca_size_agnostic.png`.
- `scripts/signature_pca_trajectory.py`: runs one roundtrip, snapshots the graph at
  intervals *during* Stage-3 refinement, and plots the **trajectory** through PCA space
  from the raw Stage-2 output toward the target — a visual story of convergence, not just
  an endpoint comparison. Artifacts: `signature_pca_trajectory_wn18rr_v4.png` (+
  `_size_agnostic` variant).
- `scripts/sweep_collect.py` + `scripts/plot_sweep_pca.py`: run many seeds/budgets and
  project the whole cloud of synthetic runs plus the target into the same PCA space —
  answers "how much does seed variance matter, and does the cloud surround the target or
  sit systematically off to one side" (bias vs. variance, visually). Artifacts:
  `signature_pca_sweep_wn18rr_v4.png`, `..._swdf.png`, `..._fb237_v4.png`.

### 5.5 Structural-gap investigations (corpus-wide surveys)

Frame these as targeted "why does the roundtrip still show error X" investigations, each
producing its own survey script + note — good material for a "known limitations /ongoing
work" subsection:

- **Edge multiplicity / pair overlap** — `scripts/edge_multiplicity.py`, note
  `developer_docs/notes/motif_reachability_and_edge_multiplicity.md`: measures the directed→simple
  edge-multiplicity gap between originals and Stage-2 synthetics across the corpus; traces
  motif-error root cause to near-zero pair overlap in the generator's output.
- **Relation reciprocity / bidirectionality** — `scripts/relation_reciprocity.py`, note
  `developer_docs/notes/relation_reciprocity_and_bidirectionality.md`: surveys per-relation
  reciprocity and forward/inverse-CS symmetry across the corpus; documents the
  bimodal reciprocity finding and the measured ~45–50% attainment ceiling.
- **Motif-counter accuracy/runtime** — `scripts/cc_variance.py`,
  `scripts/estimator_variance.py`, note `developer_docs/notes/counter_benchmark.md`: exact vs.
  color-coding (CC) motif counter comparison (accuracy + wall-clock) across motif sizes,
  and the CC estimator's variance as a function of endpoint degree — establishes *how
  much to trust* the motif features the signature and evaluation depend on.
- **Stage-3 steering limits** — note `developer_docs/notes/stage3_steering_analysis.md` +
  `scripts/profile_stage3_deltas.py`: profiling of why per-swap steering barely moves the
  loss on hub-heavy graphs (delta-cost blowup, degree guards, SA schedule retuning).
  Useful as an evaluation-methodology example (per-swap delta logging + profiling as a
  diagnostic technique), without needing to re-derive the SA algorithm itself.

### 5.6 Parameter sweeps as an evaluation tool

- `scripts/sweep_adaptive_weight_scale.py`: holds Stage 1/2 output fixed, varies only the
  Stage-3 `ADAPTIVE_WEIGHT_SCALE` constant, and compares candidates via
  `stage3_best_unweighted_error_sum` (explicitly *not* the raw loss, since the loss's
  magnitude is scale-dependent — a good "getting the comparison metric right" example
  worth walking through, since it was itself a design correction — see `CHANGELOG.md`
  2026-07-06 entries).
- General pattern worth naming explicitly: whenever a tuning knob is introduced, this
  project evaluates it via (a) a fixed pre-knob starting graph to isolate the knob's
  effect, (b) a comparison metric that is invariant to the knob itself, (c) a sweep script
  + plot rather than a single manually-tried value.

## 6. Test Data / Corpus Organization

- `data/graphs/` — curated real-KG corpus, each with a cached `signature/` directory
  (measured once, reused as roundtrip targets and PCA corpus points).
- `data/test_graphs/<name>/` — held-out graphs used for roundtrip runs; contains both the
  `.ttl` source and (after a roundtrip) `signature_synth_<timestamp>/` directories — the
  many `wn18rr_v4/signature_synth_*` directories in the working tree are exactly these
  artifacts from repeated roundtrip runs.
- `experiments/convergence_logs/`, `experiments/swap_delta_logs/`,
  `experiments/stage3_delta_profiling/` — per-run diagnostic CSVs backing §5.3/§5.5,
  named by graph/seed/budget (and an `adaptive` token when applicable) so repeated runs
  don't overwrite each other.
- `data/graph_population/` — the PCA-plot output directory (§5.4); "population" here
  refers to the corpus of real KGs the PCA basis is fit on.

## 7. Development History / Design Evolution (optional, for a "lessons learned" section)

Note: `CHANGELOG.md` only covers the last ~2 weeks (2026-06-28 → 2026-07-06) of a project
whose `git log` goes back much further (138 commits) — for full coverage, draw on
**all three** of `CHANGELOG.md`, `git log`, and `user_docs/generator.md` §"Evolution & fixes" /
`developer_docs/notes/*.md`, not the changelog alone.

- The reduced/non-over-determined signature itself was a redesign of an earlier
  over-determined ("full") signature — `user_docs/signature.md` documents the before/after
  and why the change was made (coexisting module, not an in-place replace; the original
  build plan for the pre-redesign signature has since been pruned from this tree); git
  history shows an intermediate `signature_reduced` package that
  was later merged into the single current `signature` package.
- Motif-counting evolution (visible only in git log, not the changelog): independent
  rewrites of 4-node motif counting, adoption of color-coding sampling for large graphs,
  exact 5-node (ESCAPE) and 6-cycle counting, then the hybrid exact/CC counter with degree
  guards, and finally a `MotifCounter` package split — i.e. the counting backend itself
  went through its own accuracy/performance iteration before the signature could be
  trusted to build on it.
- Selected fix-by-fix narrative beats (each is a small case study in
  hypothesis → change → measured result): CS-template quota fix (largest-remainder
  allocation), inverse-CS template completion and its Stage-3 interaction, adaptive
  Stage-3 loss weights and the scale-sweep that tuned them, the disabled star-targeting
  experiment (kept as a documented negative result), the Stage-2 out-degree global cap.
- Recurring theme across the later commits: **iterative diagnosis of Stage-3's limits** —
  profiling per-swap delta cost on hub-heavy graphs, adding degree guards
  (`CYCLE_DELTA_MAX_DEGREE`, `MOTIF4_DELTA_MAX_DEGREE`, `STAR_CENTER_MAX_DEGREE`) to keep
  swaps tractable, and ultimately tracing motif-count overshoot back to an **upstream
  Stage-2 deficiency** (near-zero edge multiplicity/pair-overlap vs. real graphs) rather
  than a Stage-3 problem — a good example of evaluation tooling (per-swap logging,
  profiling) leading to a correct root-cause diagnosis instead of a local patch.
- Common thread worth calling out explicitly: nearly every fix is paired with a **measured
  before/after** (specific error percentages), not just an implementation change — this is
  itself a methodological point about how the project evaluates its own iteration.

## 8. Known Limitations / Open Items

Pull directly from `user_docs/generator.md` §"Known limitations / open items" and the notes in
§5.5 — a concise bullet list is enough here since the detail lives in the notes:

- Co-occurrence density / row-entropy not analytically pinned (no independent control
  knob).
- Inverse-CS structure degraded by Stage-3's degree-preserving swaps (path/tree entropy
  errors ~47–60%).
- Motif over-shoot when Stage 2's starting point is already far from target and the
  rewire budget is small.
- Edge-multiplicity / bidirectionality ceiling (~45–50% attainment), traced to entity-pool
  stub-supply limits, not an algorithmic shortcoming that can be fixed with more compute.
- Out of scope by design: literals/datatypes, semantic (non-synthetic) types, tree
  templates beyond depth 2, path-template entropy for k≥4 (cost grows too fast).

## 9. Suggested Figures/Tables to Include

- Block table (§3) reproduced from `user_docs/signature.md` — good as a report table.
- The measure→generate→compare diagram from `README.md` (recreate as a figure).
- One PCA scatter plot (`signature_pca.png` or the sweep variant) as the headline
  "does the generator work" visual.
- One convergence-grid plot (`convergence_plot.py --grid` output) showing Stage-3 error
  trajectories.
- Signature-error boxplot (`signature_error_boxplot_*.png`) as the headline per-feature
  accuracy figure.
- A condensed version of the "which signature field drives which stage" input-routing
  table from `user_docs/generator.md` (trimmed to illustrate the point, not full detail).
