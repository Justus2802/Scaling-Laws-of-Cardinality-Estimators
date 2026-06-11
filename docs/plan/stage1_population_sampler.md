# Plan: the Stage-1 population sampler (sample a *novel* signature)

Status: **planned / blocked on data**. This is the design and the reasoning; no code
yet (decided). It plans **doc-Stage-1** — sampling a *novel* `ReducedGraphSignature`
from the population of real-world knowledge graphs, conditioned on size — which is the
input that feeds the generator
([generation_implementation_plan.md](generation_implementation_plan.md)).

## What this is — and what it is *not*

Two different things are both called "Stage 1" in this project. Keep them apart:

- **The generator's "Stage 1/2"** ([generation_implementation_plan.md](generation_implementation_plan.md))
  takes **one given** signature and builds a graph (schema sampler → CS-first wiring).
- **This document — doc-Stage-1** — samples a **new** signature vector from the
  distribution of real KGs, so the generator has something to instantiate. The
  [generation_algorithm_fit.md](../notes/generation_algorithm_fit.md) §"Scope clarification"
  flags exactly this: "the actual doc-Stage-1 — sampling a *novel* signature from the
  real-graph population (the conditional-on-size model) is **not** in [the generator]; the
  manifold viz is its groundwork."

So: **measure real KGs → fit a model over their 69-feature signatures → draw novel,
size-conditioned signatures**. The 69 features are defined in [signature.md](../signature.md)
(Blocks A, B, C, D, F).

## Decisions resolved

| Decision | Resolution | Consequence |
|---|---|---|
| Scope of the first deliverable | **Design doc only** (this file) | data acquisition gates code |
| Target population | **Real KGs only** | synthetic generators (LUBM, future WatDiv/LDBC) are **excluded from the fit** — `lubm` drops out of the current corpus, leaving 6 usable (after the `59621618` duplicate is also dropped) |
| Type block (only 1 typed graph) | **Acquire typed KGs first** | the type-side features stay unmodelled until several typed KGs exist (see §"Data acquisition") |

## Reality check on the current corpus

The eight measured reduced signatures (the corpus lives in `data/graphs/`; `lubm` and the
`59621618` duplicate are measured but excluded) are still thinner than their row count.
`swdf` and `dbpedia100k` were added by measuring already-on-disk graphs (no download) — see
[`graphs/GRAPH_SIZES.md`](../../graphs/GRAPH_SIZES.md):

| graph | V | R | mean-deg | T | #comp | LCC frac | usable? |
|---|---:|---:|---:|---:|---:|---:|---|
| codex_l | 77,951 | 69 | 7.86 | 0 | 2 | 1.000 | ✅ |
| dbpedia100k | 99,604 | 470 | 7.00 | 0 | 66 | 0.998 | ✅ (new) |
| swdf | 76,711 | 170 | 3.16 | 0 | 3 | 1.000 | ✅ (new) |
| hetionet | 45,158 | 24 | 49.83 | 0 | 1 | 1.000 | ✅ |
| fb237_v4 | 4,707 | 219 | 7.21 | 0 | 11 | 0.989 | ✅ |
| aids | 254,207 | 5 | 3.16 | **51** | 4 | 1.000 | typed but R=5 → relation blocks degenerate |
| lubm | 664,048 | 18 | 4.05 | 0 | 2 | 0.999 | **synthetic → excluded** |
| raw/59621618 | 4,707 | 219 | 7.21 | 0 | 11 | 0.989 | **duplicate of fb237_v4** |

Four facts dominate the design — two **relaxed** by the new graphs, two **unchanged**:

1. **Six real, distinct, usable graphs — up from four.** `swdf` and `dbpedia100k` add two
   independent non-type draws. Still excluded: `59621618` (byte-identical in signature to
   `fb237_v4`), `lubm` (synthetic), and the FB237 / WN18RR `_v4`/`_v4_ind` entries in
   `GRAPH_SIZES.md` (train/inductive **splits of the same KG**, not independent graphs).
2. **Still only `aids` is typed** (T=51) — *unchanged*. `swdf` and `dbpedia100k` carry no
   `rdf:type` (they are anonymised, structural-only), so the type-side features —
   `class_size_*`, `type_rel_spectrum_*`, `per_type_entropy_*` (≈11 of 69) — remain NaN on
   every graph but one. The two new graphs do **nothing** for the type block; the gating
   "acquire typed KGs first" decision stands.
3. **That one typed graph is still degenerate on the relation side** — *unchanged* (R=5 →
   `relation_zipf`, `obj/subj_mult_alpha`, `row_entropy`, `cs_freq` fail to fit → 24 NaNs).
4. **p ≫ n — now in two regimes.** For the non-type blocks (A,B,D,F ≈ 58 features) the count
   improved from ~4 to ~6, and `swdf`/`dbpedia100k` are relation-rich (R=170 / 470), so they
   populate Block B *cleanly* rather than degenerately like `aids`. But the **type side is
   still n = 1**: its marginal is undefined, let alone a joint. The headline problem is no
   longer uniform across the signature — it is concentrated in the type block.

The conceptual issue persists: **these graphs are not draws from one distribution.** The
corpus is a heterogeneous mixture — encyclopedic (`codex_l`, `dbpedia100k`, `fb237_v4`),
biomedical (`hetionet`, `aids`), scholarly (`swdf`). Fitting one distribution over the
mixture and sampling yields an "average" graph resembling none of them. **Defining the
population is prerequisite to fitting it** — resolved here as "real KGs only," but domain
heterogeneity remains and is handled by conditioning (below).

## Organizing principle — independent information, not row count

Every data-expansion idea adds **rows** to the measurement table, but they differ in
whether they add **independent information about the population**, which is what actually
relieves p ≫ n:

- **More distinct real KGs** → new independent draws. Real signal.
- **WCC fragments / cut subgraphs of existing graphs** → conditionally dependent on their
  parent; they inflate nominal `n` but add ≈0 independent information about the
  *population*. They *do* add information about one thing — **how a single graph's
  signature scales with its size** — so they belong in the conditional-on-size model, not
  the population fit.

Every proposal below is judged on that axis.

## Proposal evaluation

### 3b — find more distinct real KGs — **highest leverage; do first**
The only lever that directly attacks p ≫ n. Candidate sources (RDF/`.nt`, real, diverse);
**evaluated and tiered** in [notes/data_source_evaluation.md](../notes/data_source_evaluation.md)
(Bio2RDF first for the typed gate; untyped sources still feed the 58 non-type features):

- **Encyclopedic:** DBpedia (full + per-language chapters = genuinely different KGs),
  YAGO 3/4, Wikidata Truthy subsets, Freebase.
- **Biomedical (mostly typed):** Bio2RDF suite (~30 datasets — DrugBank, KEGG, ChEBI…),
  PharmKG, PrimeKG, OGB `ogbl-biokg`, `ogbl-wikikg2`.
- **Other domains:** LinkedGeoData, GeoNames, MusicBrainz-RDF, DBLP-RDF.
- **Diversity at scale:** LOD Laundromat / LOD-a-lot (thousands of crawled real KGs) — the
  apparent best single source of *spread* across the population. **But evaluated and largely
  rejected** in [notes/lod_laundromat_acquisition.md](../notes/lod_laundromat_acquisition.md):
  LOD-a-lot is one *merged* graph (a single point, not a population); LOD Laundromat's
  documents are crawl artifacts (document ≠ KG), structurally *laundered* (blank-node
  Skolemization), and skew to the dirty long tail of the web — wrong unit, wrong population.
  Its per-document **meta-dataset** is still useful for population cartography and as an
  external validation set, but the fit rows should come from the named, curated, typed
  sources above.

Caveats: (i) **do not count splits of one KG as separate graphs** (the fb237 duplicate is
the warning); (ii) **exclude synthetic generators** (decided) — they cluster tightly and
bias the population toward their generator.

### 3a — split on weakly-connected components — **drop (near-useless here)**
Every current graph has **LCC fraction 0.989–1.000**. Splitting `fb237_v4` (11 components)
yields one giant component (98.9 %) plus ~10 fragments of ~5 nodes — too small to fit any
block, so discarded; net **zero usable datapoints**, and fragments are not independent of
the parent. Only worth revisiting if a future acquisition is genuinely a union of
*several large* components (rare for real KGs).

### 5a — cut representative subgraphs to manufacture size-varied datapoints — **only for the scaling curve, never the population fit**
Two hazards, both load-bearing:

- **Method-dependent sampling bias.** Per Leskovec & Faloutsos (KDD'06) and follow-ups,
  random-walk and forest-fire preserve structure best (down to ~15 % of the graph), but
  **RW is biased toward high-degree nodes and misses the degree distribution;
  forest-fire misses clustering and path length;** BFS distorts degree badly. A cut
  subgraph's signature is therefore a *biased* estimate of "what a real graph that size
  looks like," and that bias would be baked into the generator.
- **Non-independence.** 20 subgraphs of hetionet are 20 correlated views of hetionet.

Verdict: never feed subgraphs into the population fit; use them only as in §5b.

### 5b — condition on size — **yes; the key that makes few graphs almost workable**
Reframe "condition on size" not as *fit a distribution at each size* (hopeless at n≈6) but
as a **scaling law**: `feature = f(log V) + residual`. Regress each component against
`log V`, store the trend plus residual spread. A trend line needs far fewer points than a
per-size distribution — and it is the project's own theme (cf.
`scaling_laws_student_project.pdf`, the FICE paper). This is where subgraph cutting
(§5a) earns its place: **subsample each real graph at 10–90 % (forest-fire / RW) to trace
that graph's own signature-vs-size trajectory**, fit the scaling exponents *within* each
graph (where the parent is the control, so bias is tolerable), and use the few distinct
full graphs to fit the *scatter across* graphs.

### 4 — group components (joint vs independent) — **right instinct, but impose it; don't learn it at n≈6**
A full joint over 69 features is unidentifiable; even a 3×3 empirical covariance is
unreliable from 6 points. So the grouping must be **imposed**, then *confirmed/extended*
as data grows:

- **Use the signature's own consistency web** ([signature.md](../signature.md), "Consistency
  web ≠ redundancy"): edge conservation already couples
  `meanCS_size · mean_mult = E/V`, `|edges_r| = freq(r)·E`, `Σ CS-freq = V`. These are
  *known* joint constraints — encode them as **derived relations**, not free sampled
  axes. That removes redundant dimensions for free.
- **Low-rank / copula, not full joint.** A factor model / PCA (the "manifold" the docs
  already earmark as Stage-1 groundwork) or a Gaussian copula with a *sparse* dependency
  graph captures the dominant couplings with O(n) parameters.

## Feature taxonomy for the scaling model

Not all 69 entries are free things to sample. Three kinds (cf. the feature list in any
`data/graphs/*/signature/signature.json`):

- **Constants / derived bounds — do not sample.** `obj/subj_mult_alpha_lo|hi` (fixed
  ≈1.4/3.0), `cs_size_lo|hi`, `inv_cs_size_lo|hi`, the `*_row_entropy_lo|hi`,
  `shortest_path_lo|hi` — fixed cutoffs or data-derived support bounds. Restore them
  post-hoc, don't model them.
- **Size-dependent — the scaling-law targets, regressed on `log V`.** `num_relations`,
  `num_classes`, `num_distinct_cs`, the `*_xmin` (`out/in_degree`, `relation_zipf`,
  `cs_freq`), the co-occurrence magnitude scales `subj/obj_cooc_scale`, `two_step_vmax`,
  `num_components`, and **`mean_degree`** (a densification law — Leskovec–Kleinberg–
  Faloutsos: edges grow super-linearly in nodes, so mean degree drifts with V).
- **Size-stable shapes — weak/no `V` dependence, fit as residual scatter.** the tail
  exponents (`*_alpha`, `relation_zipf_exponent`), skew-normal `loc/scale/shape` triples,
  `a_obj/a_subj`, densities, `clustering_coefficient`, `degree_assortativity`,
  `largest_component_fraction`, the exp-decay `rate`s.

Reducing the 69 to the genuinely free + size-stable subset is the first concrete step of
proposal 4 and shrinks p before any fitting.

## Recommended Stage-1 pipeline

1. **Population = real KGs only** (decided); tag each graph's domain.
2. **Acquire ~20–40 distinct real KGs** (§3b), prioritising **typed** ones (§"Data
   acquisition") — the real fix for p ≫ n.
3. **Measurement table:** one row per graph of the 69 features via
   `scripts/measure_all_raw.py --reduced`; dedup (drop `59621618`), drop `lubm`, flag
   NaN/typed.
4. **Reduce dimensionality** (taxonomy above): drop constants/derived-bounds, fold the
   consistency-web couplings into derived relations.
5. **Per-feature scaling law** `f(log V) + residual` (§5b); trace within-graph scaling
   curves by **forest-fire / RW subsampling** of each real graph (§5a) — used *only* here.
6. **Residual coupling** via the imposed groups + a low-rank/copula model (§4).
7. **Sample:** draw or condition on `V` → evaluate scaling laws → add correlated residuals
   → enforce consistency constraints + restore fixed bounds → emit a novel
   `ReducedGraphSignature` → hand to the generator.
8. **Drop WCC-splitting** (§3a).

## Data acquisition — the gating prerequisite

Per the "acquire typed KGs first" decision, before the type block can be modelled at all:

- Target **several typed KGs with rich relation vocabularies** (so they are not degenerate
  like aids/R=5): Bio2RDF datasets, DBpedia (typed via `rdf:type`), YAGO (typed). Aim for
  ≥5 typed graphs spanning a range of T before fitting `class_size_*`,
  `type_rel_spectrum_*`, `per_type_entropy_*`.
- Until then the sampler emits **untyped** signatures only (T=0), type features left NaN —
  consistent with the G6 literal-exclusion stance ([signature.md](../signature.md) §G6): a
  documented, honest gap, not a silent zero.

## Implemented — v0 uniform-range sampler

`src/signature_sampler.py` provides the sampler class hierarchy:

- **`SignatureSampler` (ABC)** — loads the corpus
  (`UniformRangeSampler.load_corpus()` reads `data/graphs/*/signature/signature.json`),
  exposes per-feature finite values, and runs the shared
  sample → post-process → emit pipeline. `FEATURE_ORDER` is derived from the block
  classes' `feature_names()`, so the 69-key schema never drifts (and a future reduced
  Block E would flow in automatically). Subclasses implement only `_sample_one`.
- **`UniformRangeSampler`** — v0: each feature ~ `Uniform(min − 0.1·r, max + 0.1·r)`
  over its finite corpus range `r = max − min`. Constants (`r=0`) reproduce exactly.
- **Post-processing** (shared): type block forced untyped (`num_classes = 0`, type
  params NaN); features with < 2 finite values → NaN; integer rounding for counts;
  domain clamps (`[0,1]` densities/clustering/LCC-fraction, `[−1,1]` assortativity,
  counts/thresholds floored at 1).
- **Output** is the 69-key `{"source", "features"}` dict — drop-in compatible with
  `load_signatures`. CLI: `scripts/sample_signature.py [--seed N] [--out path]`.
  Tests: `tests/test_signature_reduced_sampler.py`.

Carried-over v0 limitations (the successor samplers' job): independent marginals
(no consistency web, can yield implausible continuous values when an outlier like
hetionet's mean-degree widens a range), no size conditioning, untyped only, no motifs
(Block E out of scope — see [signature.md](../signature.md)).

## Open questions / next steps

- **Acquisition target & budget:** how many graphs, and is automated fetching+measuring
  (a harness over the §3b sources) in scope as the next coding task?
- **Domain conditioning:** condition only on `V`, or also on a domain/typed label (mixture
  components) once enough graphs exist?
- **Subsampling method:** forest-fire vs induced random-walk for the within-graph scaling
  curves (§5b) — pick once §5 is implemented; both beat BFS, neither is unbiased.
- These are recorded here, not blocking the doc.

## Sources

- Leskovec & Faloutsos, *Sampling from Large Graphs*, KDD 2006 —
  <https://cs.stanford.edu/~jure/pubs/sampling-kdd06.pdf>
- *Empirical comparison of network sampling techniques* — <https://arxiv.org/pdf/1506.02449>
- *Subgraph Sampling Methods for Social Networks: The Good, the Bad, and the Ugly* —
  <https://www.researchgate.net/publication/228150793>
- Leskovec, Kleinberg & Faloutsos, *Graphs over Time: Densification Laws…*, KDD 2005
  (densification / shrinking-diameter scaling laws).
