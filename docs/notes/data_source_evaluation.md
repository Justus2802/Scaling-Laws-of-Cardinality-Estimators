# Note: evaluation of the named §3b real-KG sources

Status: **research note, no code**. Companion to
[lod_laundromat_acquisition.md](lod_laundromat_acquisition.md) (which covers the
"diversity at scale" bullet). This note evaluates the *named* candidate sources in
[plan/stage1_population_sampler.md](../plan/stage1_population_sampler.md) §3b — the
encyclopedic, biomedical, and other-domain KGs — as draws for the doc-Stage-1 population
fit.

## Two sub-goals, not one — and untyped graphs serve the larger one

A source is useful if it feeds *either* of two distinct needs, which were previously
conflated:

- **(T) The type-block gate** — the ~11 type features (`class_size_*`,
  `type_rel_spectrum_*`, `per_type_entropy_*`). The plan's hard decision ("acquire ≥5 typed
  KGs with rich relation vocabularies, spanning T") applies *only here*. This needs
  **rich-typed** sources; "typed but coarse" (the aids R=5 failure mode) does not qualify.
- **(N) The non-type population spread** — Blocks A, B, D, F (degree, multiplicity,
  characteristic sets, components, clustering, paths ≈ **58 of 69 features**). These are
  measured on **any real KG, typed or not**. p ≫ n bites here too (still only ~4 usable
  points), so **untyped sources are first-class draws for (N)**; their type features simply
  stay NaN — the documented G6 / honest-gap stance in [signature.md](../signature.md), not a
  disqualification.

So "Typed?" is **not** a gate on a source's usefulness — only on *which blocks* it fills. A
distinct untyped KG in a new domain is valuable spread for 58 features.

## Evaluation axes

1. **Distinct & independent** — a new draw, not a split/duplicate (the fb237 warning) and
   not the same domain/lineage as a graph already in the corpus (the plan's
   §"independent information, not row count").
2. **Rich-typed** — required for sub-goal (T) only; irrelevant to (N).
3. **Size** — an external **HPC cluster** handles the measurement, so graphs **up to a few
   GB are fine** (the laptop's `rdflib → igraph` 5 GB OOM is no longer the ceiling). Only
   the multi-billion-triple / 100s-GB dumps (Wikidata Truthy, full Freebase) still need
   subsetting — and only *those* inherit the §5a slicing-bias concern.
4. **Native RDF** — `.nt`/`.ttl` drops into the loader; other formats need a converter that
   respects the G6 literal/blank-node conventions.
5. **As-published** — a coherent KG, not an ML *extract* (sampled/filtered splits carry the
   §5a non-independence/bias concern, like aids / codex / fb237).
6. **Available & fresh** — stably hosted and reachable.

**Independence baseline (what the corpus already covers):** codex_l (Wikidata lineage) +
fb237 (Freebase) = encyclopedic; hetionet + aids = biomedical. Sources sharing those
lineages add little *independent* information even though they still add a row.

## Per-source assessment

| Source | Distinct/indep. | Fills (T)? | Fills (N)? | Size (HPC) | Native RDF | As-published | Verdict |
|---|---|---|---|---|---|---|---|
| **Bio2RDF** (~35 datasets) | ✅ each sub-dataset distinct | ✅ rich-typed | ✅ | ✅ pick ≤ few-GB ones | ⚠️ n-quads → strip context | ✅ | **Tier 1** |
| **DBpedia chapters** | ✅ per-language = distinct KGs | ✅ DBpedia ontology | ✅ | ✅ | ✅ | ✅ | **Tier 1** (repo has `dbpedia100k`) |
| **YAGO 4.5** | ⚠️ Wikidata lineage | ✅ richest taxonomy | ✅ | ✅ a few-GB split | ✅ Turtle | ✅ | **Tier 1** |
| **DBLP-RDF** | ✅ new domain (bibliographic) | ✅ publication types | ✅ | ✅ | ✅ | ✅ maintained | **Tier 1–2** — top diversity-adder |
| **GeoNames RDF** | ✅ new domain (geographic) | ❌ coarse types | ✅ strong (N) draw | ✅ | ✅ | ✅ | **Tier 2** — now valued for (N) |
| **PrimeKG** | ⚠️ biomedical (≈hetionet) | ✅ 10 node / 30 edge types | ✅ | ✅ ~4 M rels | ❌ CSV → nt | ✅ Harvard 2023 | **Tier 2** |
| **ogbl-wikikg2** | ❌ Wikidata lineage | ❌ untyped | ✅ large (N) draw | ✅ 2.5 M ent. | ❌ int-ID → nt | ❌ benchmark extract | **Tier 2–3** |
| **LinkedGeoData** | ✅ geographic | ⚠️ light | ✅ | ✅ | ✅ | ✅ | **DEAD** — server refused (2026-06-09) |
| **MusicBrainz-RDF** | ✅ new domain (music) | ⚠️ light | ✅ | ✅ | ✅ | ✅ | **DEAD as RDF** — unmaintained, JSON-LD only |
| **Wikidata Truthy** | ❌ ≈codex lineage | ✅ | ✅ | ⚠️ 100s GB → subset | ✅ | ✅ | **Tier 3** — subset only |
| **Freebase** | ❌ ≈fb237 lineage | partial | ✅ | ⚠️ large → maybe subset | ✅ | ✅ (Google archive) | **Tier 3** — lineage dup |
| **ogbl-biokg** | ⚠️ biomedical (≈hetionet) | ❌ 5 coarse categories | ✅ | ✅ | ❌ int-ID → nt | ❌ benchmark extract | **Tier 3** |
| **PharmKG** | ⚠️ biomedical | ⚠️ curated subset | ✅ | ✅ ~500 k | ❌ tabular → nt | ⚠️ heavily filtered | **Tier 3** |

## Availability — verified 2026-06-09

Liveness and download format checked by probing each download endpoint:

| Source | Status | Evidence / files |
|---|---|---|
| **DBLP-RDF** | ✅ live, **current** | dump updated 2026-06-09: `dblp.nt.gz` 4.7 G / `dblp.ttl.gz` 2.1 G / `dblp.rdf.gz` 2.4 G |
| **Bio2RDF** | ✅ live | `download.bio2rdf.org/files/release/{1,2,3,4}/` — Release 4 (Feb 2021); SPARQL at `/sparql` |
| **DBpedia** | ✅ live | Databus (`databus.dbpedia.org`), ~140 languages, monthly releases |
| **YAGO 4.5** | ✅ live | `yago-knowledge.org/data/yago4.5/` — Turtle (Schema/Taxonomy/Facts/Facts-beyond/Meta) |
| **GeoNames RDF** | ✅ live | `download.geonames.org/all-geonames-rdf.zip` resolves (large file) |
| **PrimeKG** | ✅ live | Harvard Dataverse v2.1 RELEASED, CC0 — `kg.csv` 982 M, `edges.csv` 387 M, `nodes.tab` |
| **OGB** biokg / wikikg2 | ✅ live | `snap.stanford.edu/ogb/data/linkproppred/` — `biokg.zip` 919 M, `wikikg-v2.zip` 4.1 G |
| **Wikidata Truthy** | ✅ live | Wikimedia dumps (truthy `.nt`, 100s GB) |
| **Freebase** | ⚠️ archive-only | Google archive / academic mirrors (project discontinued 2016) |
| **LinkedGeoData** | ❌ **DEAD** | `downloads.linkedgeodata.org` → connection refused |
| **MusicBrainz-RDF** (LinkedBrainz) | ❌ **DEAD as RDF** | unmaintained; RDFa removed 2014; only JSON-LD today |

**Consequence:** the geographic-spread slot now rests on **GeoNames** alone (LinkedGeoData
gone), and the **music** domain drops out entirely (no maintained RDF dump). Everything in
Tiers 1–2 except those two is confirmed live, native-or-convertible, and within the
HPC-feasible size range.

## Reasoning behind the ranking

**Bio2RDF still single-handedly addresses the (T) gate.** ~35 *separately published,
individually distinct* typed life-science KGs (DrugBank, KEGG, ChEBI, …), every URI typed,
rich relation vocabularies. Selecting several mid-sized sub-datasets yields ≥5 typed draws
spanning T from one acquisition. With HPC the size filter is gentle (skip only the
multi-billion-triple sub-datasets); friction is just the n-quads → `.nt` strip and the dated
(Release 4, ~2014) but stably-archived data.

**DBpedia chapters, YAGO 4.5 fill (T) and (N)** but partly overlap the Wikidata/Freebase
lineage already in the corpus, so they extend more than they diversify. With the size ceiling
relaxed, YAGO needs only a normal Turtle split (not a tiny one) and DBpedia full per-language
chapters are usable as-is. DBpedia's per-language chapters are genuinely distinct KGs;
`dbpedia100k` is already a working template in `graphs/`.

**DBLP and GeoNames are the real diversity-adders for (N).** Under the corrected framing these
rise: they bring *new domains* (bibliographic, geographic) absent from the corpus, and
contribute clean draws for the 58 non-type features regardless of type richness. DBLP is the
standout — native RDF, **updated daily** (verified current 2026-06-09), and it also helps (T)
via publication types. GeoNames moves up too: its coarse typing only means the type block is
NaN, which is fine; it remains a strong, distinct, native-RDF (N) draw. **The two legacy
music/geo sources I had earmarked here — LinkedGeoData and MusicBrainz-RDF — are now confirmed
dead** (see Availability), so the music domain is unavailable and geographic spread rests on
GeoNames alone.

**The ML-benchmark sources (OGB, PrimeKG, PharmKG) are second-choice for structural reasons,
not typing.** They need format conversion, and the OGB graphs are *benchmark extracts*
(sampled/filtered splits → the §5a bias the plan warns of) and biomedical (≈hetionet).
ogbl-wikikg2 is a usable large (N) draw but carries both the extract bias and Wikidata
lineage. PrimeKG is the best of this group (coherent, well-typed, well-hosted).

**Lineage-duplicate giants (Wikidata Truthy, Freebase) stay low** — not because of size now,
but because they re-add the codex/fb237 lineage already present (low *independent*
information), and the truly huge ones still require a subset (→ §5a bias). Enter only to fill
a specific large-V gap.

## Cross-cutting caveats

- **Slicing bias now applies only to the genuine giants.** With HPC, few-GB graphs are
  measured whole (clean draws). Only Wikidata Truthy / full Freebase must be cut, and those
  cut rows carry the §5a method-dependent bias → treat as scaling-curve points (§5b), not
  pristine population draws.
- **Format conversion is real work.** Native RDF: DBpedia, YAGO, GeoNames, DBLP, MusicBrainz,
  LinkedGeoData, Freebase, Wikidata. Needs an adapter to `.nt`: Bio2RDF (n-quads→triples),
  OGB (int-ID triples), PrimeKG/PharmKG (CSV/tabular). The adapter must honour the **G6**
  literal-exclusion and blank-node conventions of [signature.md](../signature.md) so rows are
  comparable to the existing corpus.
- **Benchmark-extract bias.** OGB (and the already-present codex/fb237/aids) are ML
  splits/extracts of larger KGs, not as-published — flag such rows; do not treat them as
  pristine population draws.
- **Independence weighting.** Maximise independent information (the plan's organizing
  principle): weight acquisition toward distinct typed sources (Bio2RDF sub-datasets) for (T)
  and *new domains* (bibliographic, geographic, music) for (N), over re-adding the
  encyclopedic/biomedical lineage already represented.

## Recommended acquisition order (feeds plan §"Recommended pipeline" step 2)

1. **Bio2RDF** — several mid-sized sub-datasets → satisfies the (T) gate in one pass.
2. **DBpedia** — a few language chapters → distinct encyclopedic draws, (T)+(N).
3. **DBLP-RDF** + **GeoNames** → new domains for (N) spread (DBLP also helps (T); both live).
4. **YAGO 4.5** split → richest taxonomy, encyclopedic.
5. **PrimeKG** → extra typed-biomedical draw.
6. Everything else (OGB, Freebase/Wikidata subsets, PharmKG) only to fill specific gaps,
   with the caveats above. (LinkedGeoData and MusicBrainz-RDF are dead — do not pursue.)

## Sources

- Bio2RDF — <https://bio2rdf.org/> · stats:
  <https://github.com/bio2rdf/bio2rdf-scripts/wiki/Bio2RDF-Dataset-Summary-Statistics>
- YAGO 4 / 4.5 — <https://yago-knowledge.org/downloads/yago-4-5> ·
  <https://arxiv.org/pdf/2308.11884>
- OGB link property prediction (ogbl-biokg, ogbl-wikikg2) —
  <https://ogb.stanford.edu/docs/linkprop/>
- PrimeKG — <https://zitniklab.hms.harvard.edu/projects/PrimeKG/> ·
  <https://www.nature.com/articles/s41597-023-01960-3>
- DBpedia — <https://www.dbpedia.org/resources/> · DBLP-RDF — <https://dblp.org/rdf/> ·
  GeoNames — <https://www.geonames.org/ontology/>
