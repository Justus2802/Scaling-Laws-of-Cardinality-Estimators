# Per-relation reciprocity: how bidirectionality (and CS symmetry) is structured

Follow-on to `motif_reachability_and_edge_multiplicity.md`, which found Stage-2
produces ~zero **bidirectional** pairs (a↔b) whereas real graphs have many, and that
this is the main driver of the simple-edge inflation / off-manifold degree sequence.
This note characterises *where* bidirectionality comes from, to decide how to encode
and generate it. Survey tool: `scripts/relation_reciprocity.py`
(outputs in `experiments/relation_reciprocity/`).

## Question

Bidirectionality needs the same entities to be **both subject and object** of a
relation — i.e. a relation's forward CS (who emits it) and inverse CS (who receives
it) must coincide. Is that coincidence an entity-level correlation (hard to encode),
or a **per-relation** property (a relation is "symmetric" or not)?

## Finding — it is per-relation, and nearly bimodal

For each relation `r`, over entity–entity content edges:

- `reciprocity[r]` = fraction of r's directed edges `a→b` whose reverse `b→a` also
  exists via `r`;
- `cs_symmetry[r]` = `|S_r ∩ O_r| / |S_r ∪ O_r|` (entities both emitting and receiving r).

**Relations split cleanly into symmetric (reciprocity ≈ 0.8–1.0) and asymmetric
(≈ 0.0), with almost nothing in between**, and `cs_symmetry` tracks `reciprocity`
tightly (symmetric relations' forward and inverse CS coincide). Examples:

| graph | relation | edges | reciprocity | cs_symmetry |
|---|---|---|---|---|
| wn18rr | derivationally_related | 5787 | **0.95** | 0.96 |
| wn18rr | verb_group | 223 | **0.93** | 0.90 |
| wn18rr | similar_to | 18 | **1.00** | 1.00 |
| wn18rr | hypernym | 2765 | 0.00 | 0.32 |
| wn18rr | has_part | 406 | 0.00 | 0.23 |
| fb237 | award_nominee | 2002 | **0.82** | 0.90 |
| fb237 | award_winner | 943 | **0.79** | 0.85 |
| fb237 | film_release_region | 1871 | 0.00 | 0.00 |
| fb237 | profession / gender / genre / … | ~1000 each | 0.00 | 0.00 |

The "middle band" (reciprocity 0.1–0.5) holds **~0 % of edges** on every graph
surveyed — the split is essentially binary.

## Corpus survey (entity–entity content edges)

| graph | overall recip. | any-rel bidir pair frac | symmetric-edge frac | mid-band edge frac | CS∩invCS Jaccard |
|---|---|---|---|---|---|
| aids | **1.00** | 0.50 | 1.00 | 0.00 | 1.00 |
| wn18rr_v4_ind | 0.66 | 0.33 | 0.70 | 0.00 | 0.60 |
| wn18rr_v4 | 0.61 | 0.31 | 0.66 | 0.00 | 0.62 |
| fb237_v4_ind | 0.14 | 0.14 | 0.17 | 0.00 | 0.08 |
| fb237_v4 | 0.12 | 0.13 | 0.15 | 0.00 | 0.07 |
| dbpedia100k | 0.05 | 0.06 | 0.01 | **0.17** | 0.06 |
| codex_l | 0.03 | 0.02 | 0.03 | 0.00 | 0.01 |
| swdf | **0.00** | **0.18** | 0.00 | 0.00 | 0.00 |
| hetionet | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |

- `overall recip.` = edge-weighted same-relation reciprocity; `symmetric-edge frac` =
  edges in relations with reciprocity > 0.5; `mid-band edge frac` low ⇒ bimodal.
- The entity-level CS↔inv-CS Jaccard is a *consequence* of which symmetric relations
  an entity uses, not an independent signal — it rises exactly with `overall recip.`
- Span: **aids fully symmetric** (every relation reciprocity 1.0 — molecular bonds),
  **hetionet fully asymmetric** (0.0). Both perfectly bimodal (mid-band 0).
- **dbpedia100k is the partial exception** (mid-band 0.17): its relations are not as
  cleanly split — a fraction sit at intermediate reciprocity. A quantile fit over
  per-relation reciprocity (rather than a binary flag) handles this gracefully.

## The one exception — swdf is cross-relational

swdf has **0.00 same-relation reciprocity but 0.18 any-relation bidirectionality**:
its reverse edges use a *different* relation than the forward edge (`a →r1 b`,
`b →r2 a`). So per-relation reciprocity captures bidirectionality on
fb237/wn18rr/codex_l but **not** swdf's. A complete mechanism needs both:
- **same-relation reciprocity** (the dominant, tractable case) → a per-relation
  feature + a shared entity pool for symmetric relations;
- **cross-relation bidirectionality** (swdf) → the residual, caught by the
  graph-level `bidirectional_ratio` (already in Block C) via opportunistic reverse
  matching, not by the per-relation feature.

## Implemented mechanism — four independent factors, all had to be fixed

Stage-2 is a **product-of-independent-marginals generator**: entities, CS
membership, stub allocation and degree targets are each drawn independently, but a
mutual pair `a↔b` under relation `r` requires **four** of these to correlate at once.
Each was diagnosed and fixed in turn (`src/generator/stage2.py`,
`src/generator/stage1.py`, `src/signature/block_b.py`):

1. **CS membership** (`S_r ∩ O_r` too small) — forward CS (§3a, who emits `r`) and
   inverse CS (§3b, who receives `r`) are built independently, so even a relation
   marked symmetric has almost no entities eligible for both. **Fix:** a new §3b2
   pass adds `r` to a `ρ_r` fraction of `r`'s emitters' inverse CS (a size-preserving
   swap, so `inv_cs_size_q` is untouched).
2. **Stub allocation** (out-stubs ⊥ in-stubs) — even CS-eligible-both entities rarely
   get a stub on *both* sides from two independent multinomials. **Fix:** explicit
   reservation — force `round(ρ_r·edges_r/2)` entities from `S_r ∩ O_r` to have ≥1
   out-stub *and* ≥1 in-stub, stealing from the max-count entity on each side
   (budget-neutral).
3. **Pair construction reuse** — the first version of the mutual-pair builder
   consumed each eligible entity **once**, even if it had several stubs (typical
   `edges_r/|S_r|` is ≫1). **Fix:** draw from the eligible pool *with replacement*,
   removing an entity only once it is exhausted on either side — this alone roughly
   doubled attainment in testing.
4. **Reciprocity → relation assignment** — assigning reciprocity to synthetic
   relations independently of frequency put it on the *wrong* relations (the biggest
   relation drawing ρ≈0 despite being symmetric in the original). **Fix:** Block B
   now stores reciprocity as `P(symmetric)` binned by each relation's own
   cumulative-**edge**-frequency rank (6 fixed bins, edge-mass-weighted so a few huge
   relations aren't diluted by a long tail of tiny ones) + one scalar for the
   symmetric-mode magnitude (`recip_symmetric_frac`/`recip_symmetric_value`,
   replacing an earlier frequency-blind `recip_q`). Stage 1 looks up each synthetic
   relation's *own* frequency-rank bin rather than drawing independently. An empty
   bin (common when R is small — aids has only 5 relations over 6 fixed bins)
   borrows its nearest non-empty neighbour rather than defaulting to "asymmetric",
   which was found to silently zero out a fully-symmetric graph's reciprocity.

A same-relation double-edge swap **cannot** fix factors 2–3 after the fact: it
preserves each entity's per-relation in/out-degree exactly (only *who* connects to
whom changes), so it can never give an entity its first in-edge (or out-edge) of a
relation it currently has zero of. The correlation has to be injected where degree is
actually decided — inside Stage-2's construction, not in a later pass.

## Measured result — substantial improvement, not full attainment

Re-measured after all four fixes (`scripts/edge_multiplicity.py`,
`experiments/edge_multiplicity/`):

| graph | orig ρ | synth ρ before | synth ρ after | simple-edge inflation before → after |
|---|---|---|---|---|
| fb237_v4 | 1.268 | 1.007 | 1.057 | +26% → +20% |
| wn18rr_v4 | 1.451 | 1.000 | 1.167 | +45% → +24% |
| aids | 2.000 | ~1.0 | 1.303 | +~100% → +53% |

Bidirectional-pair attainment lands around **45–50% of target** on all three, not
full attainment. Root cause of the residual (verified by direct instrumentation on
wn18rr's biggest relation): reaching `ρ_r` needs each shared-pool entity to supply
`~ρ_r·edges_r / |S_r∩O_r|` stub-pairs on average — on wn18rr's dominant relation
that works out to **~3.4 stubs/entity needed vs ~2.85 available** (`edges_r/n_sr`).
Even with every fix above, `S_r ∩ O_r` and the average degree impose a genuine
capacity ceiling; hitting it fully would require either growing the shared pool
further (§3b2 more aggressively) or accepting non-uniform stub demand across the
pool (currently unweighted). Left as an open item, not chased further this round —
the ~20–50% reduction in edge inflation is a real, substantial improvement in
exactly the quantity that drives the paw/c5 motif overshoot
(`motif_reachability_and_edge_multiplicity.md`), and per-relation reciprocity now
lands on the *correct* relations (verified: the biggest/most-frequent relation in
each graph gets the right symmetric/asymmetric call), even though its magnitude
undershoots.

## Future work — joint distributions between per-relation features (deferred)

Per-relation reciprocity is now conditioned on frequency (the one joint that
mattered most for correct relation targeting), but it is still independent of the
*other* per-relation / per-entity features — CS size, co-occurrence group,
multiplicity. These are almost certainly correlated too, and marginals do not pin
the joint (the same failure mode that produced the whole edge-multiplicity gap).

Concretely worth capturing later, a **joint of reciprocity × CS size**: do symmetric
relations concentrate in particular CS templates / CS sizes (e.g. small tightly-knit
symmetric neighbourhoods vs large asymmetric fan-outs)? If so, conditioning §3b2's
pool-overlap injection on CS size too (not just frequency) could also help close the
residual capacity gap above. Other candidate joints: reciprocity × co-occurrence
group, CS-size × multiplicity. Deferred; the right encoding (conditional quantile
fits, or a low-rank joint like the co-occurrence spectrum) is itself an open design
question.
