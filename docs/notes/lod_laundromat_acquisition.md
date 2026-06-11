# Note: LOD-a-lot / LOD Laundromat as a doc-Stage-1 data source

Status: **research note, no code**. Evaluates whether the LOD-a-lot / LOD Laundromat
corpus can supply the *distinct real KGs* that
[plan/stage1_population_sampler.md](../plan/stage1_population_sampler.md) §3b calls the
highest-leverage fix for `p ≫ n`. The plan lists "LOD Laundromat / LOD-a-lot (thousands of
crawled real KGs)" as one bullet — **the best single source of spread**. This note splits
that bullet, because the two artifacts are different things, and records the case *for* and
*against* leaning on them. Conclusion up front: **use the meta-dataset for exploration and
validation only; acquire the actual fit rows from the named, curated, typed §3b sources.**

## The core distinction the plan conflates

LOD-a-lot and LOD Laundromat are not the same artifact, and the difference is load-bearing
for a *population* sampler (which needs independent draws, not row count — cf. the plan's
§"independent information, not row count").

| | **LOD-a-lot** | **LOD Laundromat** |
|---|---|---|
| What it is | **One** self-indexed HDT file: the *merge* of every document into a single graph (28.36 B triples, 3.21 B distinct subjects, 3.17 B distinct objects, 1.17 M properties) | ~650 K *separate* cleaned documents (gzipped, lexicographically-sorted, dedup'd canonical N-Triples), each its own file |
| Granularity | Document boundaries **destroyed** by the union | Document = the natural unit |
| As input to the fit | **A single signature point** | **Up to ~650 K rows** (but see objections) |
| Footprint | 524 GB disk / 15.7 GB RAM to query | Per-doc files, mostly tiny |

**LOD-a-lot is the wrong granularity for the population fit.** Measuring it yields *one*
row, not thousands — and the `rdflib.parse → igraph` loader ([src/kg_io.py](../../src/kg_io.py))
already OOMs on yago at 5 GB; 28 B triples / 304 GB is categorically out of reach. Treating
LOD-a-lot as "a KG to measure" is a category error. Its only defensible role is as a
**queryable HDT index** for materializing slices on a laptop — but a slice of the merged
union is a *cut subgraph* with destroyed provenance, i.e. the plan's §5a territory (biased,
non-independent), never the population fit. LOD Laundromat's native per-document files are
cleaner for that purpose anyway.

So the rest of this note is about **LOD Laundromat documents**, the only candidate at the
right granularity.

## The find: LOD Laundromat ships a per-document meta-dataset

A companion artifact the plan does not mention: the **LOD Laundromat meta-dataset**
(Rietveld, Beek, Hoekstra, Schlobach, *Meta-Data for a lot of LOD*, Semantic Web J. 2017) —
~110 M triples of *structural* descriptions, one record per document, SPARQL/`CONSTRUCT`-
queryable, uniformly and algorithmically computed (so cross-document-comparable, which the
paper stresses VoID/LODStats are not). Per document (their Table 1) it records:

- `Triples` (= E), `Entities`, `Distinct Subjects`, `Distinct Objects`,
  **`Distinct Properties`** (≈ `num_relations`/R), **`Distinct Classes`** (≈ T),
  distinct/total IRIs, blank nodes, literals;
- **`Degree`, `Indegree`, `Outdegree` as full `DescriptiveStatistics`** (mean/median/min/max/std);
- IRI / literal length statistics.

Mapped onto the [69-feature signature](../signature.md): this covers essentially all of
**Block A** (`num_entities`, `num_relations`, `mean_degree = E/V`) plus the degree moments —
for ~650 K graphs, at zero parsing cost. It does **not** cover anything in Blocks B/C/D/F
that is not a count or a degree moment: the tail exponents (`*_alpha`, `relation_zipf`),
characteristic sets (`num_distinct_cs`, `cs_freq`, CS-size), the co-occurrence / `P(r|t)`
spectra, `clustering_coefficient`, `degree_assortativity`, `largest_component_fraction`,
shortest-path shapes. Those require *loading each document* and running the project pipeline.

## Two tiers it could feed

- **Tier 1 — meta-dataset (cheap, population-scale, partial signature).** Query per-document
  Block-A + degree moments. Use it to (a) fit the **size scaling laws**
  `feature = f(log V) + residual` (plan §5b) on real data at scale rather than n≈4, and
  (b) **map the population's shape** — stratify by domain/namespace, detect the mixture,
  inform the §4 conditioning structure — *before* expensive measurement.
- **Tier 2 — download + measure a curated sample (full 69-d signature).** Laundromat docs
  are dedup'd sorted N-Triples → after `gunzip` they drop straight into the `.nt` loader,
  no new I/O code. Measure ~30–300 curated documents with
  `scripts/measure_signature_reduced.py` for full-signature rows — the literal §3b fix. The
  meta-dataset is the **sampling frame** that makes the curation principled.

## The case against — ordered strongest first

1. **A "document" is not a knowledge graph (the unit is wrong).** A document is whatever
   file sat at a crawled URL — a publishing/crawl artifact, not a coherent KG. The mapping
   is many-to-many: one KG (DBpedia, Bio2RDF) is split across thousands of dump files →
   a cloud of correlated rows; one file may bundle unrelated graphs. This is the plan's
   fb237-duplicate hazard **at 650 K scale and worse** — even non-duplicate documents from
   the same source are not independent draws. "650 K real KGs" is really "650 K files"; the
   number of distinct, KG-scale, *independent* sources is plausibly only hundreds, which
   barely moves `p ≫ n` while adding a large dependency-modeling burden. The unit you sample
   (file) is not the unit the generator instantiates (KG).

2. **The cleaning alters the structure being measured.** LOD Laundromat does **not**
   republish as-published: it **Skolemizes blank nodes** (→ well-known IRIs), removes
   duplicate triples, recovers-or-drops syntax errors, and lexicographically sorts.
   Skolemization changes node identity, V, component structure, and the characteristic sets
   (Block D); the loader keeps blank nodes as `_:b`, so a Laundromat doc is measured under a
   *different node-identity convention* than the existing raw corpus (aids, hetionet,
   codex). Syntax-error recovery silently drops an unknown, source-dependent number of
   triples. Net: you would fit the generator to *laundered* graphs, not real KGs.

3. **Wrong population for the goal.** The project endpoint is scaling laws of cardinality
   estimators (FICE), benchmarked on curated KGs (codex, hetionet). LOD Laundromat is
   explicitly *"other people's **dirty** data"* — the messy long tail of the crawlable web
   (FOAF scraps, RDFa snippets, half-published dumps). Fitting the population model to that
   tail makes the *novel* signatures resemble web detritus, not the quality KGs the
   estimators are evaluated on. Defining the population is prerequisite to fitting it (plan
   §"Reality check") — Laundromat may be the wrong population.

4. **It does not serve the gating prerequisite (typed KGs).** The one hard decision is
   "acquire typed KGs with rich relation vocabularies first." Laundromat is dominated by
   untyped / lightly-typed web data; rich-schema typed KGs (Bio2RDF-style) are a thin
   minority, better acquired directly and by name. Laundromat leaves the type block
   (`class_size_*`, `type_rel_spectrum_*`, `per_type_entropy_*`) as unmodelled as it is
   today — it does not unblock the actual blocker.

5. **The size distribution guts the effective sample.** The meta-data paper's Fig. 1 (avg
   out-degree over 650 K docs) collapses almost immediately — the overwhelming majority of
   documents are tiny, far below the V ∈ [4.7 k, 664 k] band. After filtering to KG-scale
   and deduping by source, the usable distinct count shrinks by one–two orders of magnitude.
   The "thousands of draws" is, for this purpose, far fewer.

6. **Definitional batch-effects in the "free" Tier-1 rows.** The paper itself flags that
   `Entities` is subjective and VoID-style counts are computed differently across tools.
   Their `Distinct Subjects/Objects` and `Degree` stats use *their* node definition
   (Skolemized, literal-inclusive), which does not match V (literals excluded per **G6**,
   blank nodes as `_:b`). Mixing those rows with the project's own measurements risks a
   scaling law that fits a measurement-convention artifact, not real densification.
   Reconciliation is non-trivial and partly impossible (you cannot re-derive the project's
   convention from their aggregates).

7. **Stale and possibly offline.** The major crawl is ~2015 (657,902 datasets, May 2015);
   the live endpoints probed during this research failed (`lod-a-lot` TLS error; no
   current-status confirmation for the wardrobe / SPARQL). For a project where data
   acquisition is the gate, a harness against a decade-old, possibly-offline service is poor
   ROI versus the stably-hosted named §3b sources.

## Bottom line / recommendation

- **Tier 2 (measure individual docs)** is undermined by objections 1–3: wrong unit,
  laundered data, possibly wrong population. **Do not fold Laundromat documents into the
  population fit.**
- **Tier 1 (meta-dataset)** survives best, but only as **exploratory population
  cartography** — seeing the mixture / size structure and sanity-checking scaling-law
  *shapes* — **not** as rows folded into the fit (objection 6).
- **Acquire the actual fit rows from the named, curated, typed §3b sources** (DBpedia
  chapters, Bio2RDF, YAGO), where the unit really is a KG, the data is as-published, and the
  typed-graph blocker is addressed.
- **One strong residual use survives all seven objections: an external *validation* set.**
  Once the sampler emits a novel signature at size V, check whether real graphs near V in
  the meta-dataset have Block-A features in the same range — a held-out reality check that
  does not require trusting Laundromat structure as training data.

## Verify before relying on this (acquisition is the gate)

- Can individual cleaned documents still be downloaded from the wardrobe?
- Is the meta-dataset (SPARQL endpoint or its ~110 M-triple dump) reachable, or is there an
  archived / Zenodo copy? (`lod-a-lot.lod.labs.vu.nl` returned a TLS certificate error
  during this research; liveness of the LOD Laundromat service was not confirmed.)
- If both are dead, this degrades to "find a mirror, or fall back to the §3b named sources."

## Sources

- LOD-a-lot (ISWC 2017) — <https://link.springer.com/chapter/10.1007/978-3-319-68204-4_7>
- Meta-Data for a lot of LOD (Semantic Web J. 2017) —
  <https://www.semantic-web-journal.net/system/files/swj1393.pdf>
- LOD Laundromat: A Uniform Way of Publishing Other People's Dirty Data (ISWC 2014) —
  <http://laurensrietveld.nl/pdf/LOD_Laundromat_-_A_Uniform_Way_of_Publishing_Other_Peoples_Dirty_Data.pdf>
- RDF HDT datasets — <https://www.rdfhdt.org/datasets/>
- LOD Laundromat GitHub org — <https://github.com/LOD-Laundromat>
