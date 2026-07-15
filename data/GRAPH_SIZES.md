# Graph Sizes

Base RDF graphs in [`graphs/data/`](data/), ranked by triple count (line count of the raw graph file). The `*_queries` / `*_ranking` subdirectories are query workloads, not graphs, and are excluded.

Measurement status below refers to the **reduced signature** (`sig_out_reduced/`, Blocks
A,B,C,D,F). See [developer_docs/plan/stage1_population_sampler.md](../developer_docs/plan/stage1_population_sampler.md)
§"Reality check" for why only ~4 of these are usable population draws, and
[developer_docs/notes/data_source_evaluation.md](../developer_docs/notes/data_source_evaluation.md) for the
acquisition plan.

| Graph | Triples | File size | Raw file | Measured? | Status / why |
|---|---:|---:|---|:--:|---|
| **wn18rr_v4** | **9,842** | 976 K | `wn18rr_v4/raw/59622641` | ❌ | split of WN18RR; relation-sparse (R=9) → would be degenerate like aids. Optional lexical-domain candidate (measure v4 **only**) |
| fb237_v4_ind | 14,554 | 1.9 M | `fb237_v4_ind/raw/59621825` | ❌ | inductive **split of FB237** — already represented by `fb237_v4`; skip (duplicate KG) |
| wn18rr_v4_ind | 15,157 | 1.5 M | `wn18rr_v4_ind/raw/59622656` | ❌ | inductive **split of WN18RR** (same KG as `wn18rr_v4`); skip |
| fb237_v4 | 33,916 | 4.5 M | `fb237_v4/raw/fb237_v4.nt` | ✅ | **usable draw** (FB237; a Freebase benchmark extract). Untyped, T=0 |
| swdf | 242,256 | 15 M | `swdf/raw/59320634` | ✅ | **usable draw** — relation-rich (R=170), untyped (T=0), scholarly domain. V=76,711, mean-deg 3.16 |
| codex_l | 612,437 | 73 M | `codex_l/raw/codex_l.nt` | ✅ | **usable draw** (Wikidata extract). Untyped, T=0 |
| dbpedia100k | 697,572 | 59 M | `dbpedia100k/raw/59622674` | ✅ | **usable draw** — relation-rich (R=470), untyped (no `rdf:type`), encyclopedic. V=99,604, mean-deg 7.00 |
| aids | 802,066 | 58 M | `aids/raw/AIDS.nt` | ✅ | **usable but degenerate** — only typed graph (T=51) yet R=5 → ~24 relation/type features NaN |
| hetionet | 2,250,197 | 302 M | `hetionet/raw/hetionet.nt` | ✅ | **usable draw** (biomedical). Untyped, T=0 |
| lubm | 2,688,849 | 202 M | `lubm/raw/59410577.ttl` | ⚠️ measured, **excluded** | **synthetic** (LUBM generator) — excluded from the population fit by decision |
| wikidata | 21,354,359 | 2.5 G | `wikidata/raw/59320361` | ❌ | **too large** — OOMs the `rdflib → igraph` loader; HPC-only. Wikidata lineage already covered by `codex_l` |
| yago | 58,276,870 | 5.0 G | `yago/raw/Yago.nt` | ❌ | **too large** — explicitly skipped for RAM in `measure_all_raw.py`; HPC-only |

Not in this table: `graphs/data/raw/59621618` was measured but is a **byte-identical
duplicate** of `fb237_v4`, so it is dropped.

**Tally:** 8 reduced signatures exist (`fb237_v4`, `codex_l`, `aids`, `hetionet`, `swdf`,
`dbpedia100k`, `lubm`, and the `5
9621618` duplicate); of those, `lubm` (synthetic) and
`59621618` (duplicate) drop out → **6 usable real draws** (`fb237_v4`, `codex_l`, `aids`,
`hetionet`, `swdf`, `dbpedia100k`), with `aids` degenerate on the relation side. `swdf` +
`dbpedia100k` were added from already-on-disk graphs (no download). Next cheapest lift is
acquiring external sources — see
[developer_docs/notes/data_source_evaluation.md](../developer_docs/notes/data_source_evaluation.md).

- **Smallest:** `wn18rr_v4` — 9,842 triples (~976 KB)
- **Largest:** `yago` — 58,276,870 triples (~5.0 GB)
