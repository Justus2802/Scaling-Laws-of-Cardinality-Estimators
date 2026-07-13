# Plan — rdf:type out of the degree measurement, and a two-sided degree-sum invariant

## Context

Stage 2 currently ends `_sample_target_degrees` (`stage2.py:490-494`) with a multinomial "top-up" that
raises `Σ targets` until it reaches `content_E`. It reads as a rounding patch. It is not. Measured
across all 9 corpus signatures, it is covering a **27–57% shortfall** in the Stage-1 out-degree
sequence, and it pays for that by inflating the p90/max degree targets by **1.2–2.3×** — undoing the
extreme-value matching in `_adapters.py:122-125` that exists specifically to land the max on target.

Two root causes, both upstream of the top-up:

1. **rdf:type edges and class nodes pollute the degree fits.** Block B measures degrees over *all
   non-literal vertices* with *all* edges included (`block_b.py:160-163`), so rdf:type class nodes are
   counted as entities and their type edges as degree. aids' `in_degree_max` = **184,493** is literally
   `<http://class0>`'s instance count (verified against `data/graphs/aids/AIDS.nt`), not any entity's
   in-degree. Stage 1 then extreme-value-matches a tail exponent to it and spreads a power law up to
   184k across 10% of the *entities* — so aids' in-targets sum to **2.47× the edge budget** and the
   in-degree quota never binds. In-degree steering on aids is effectively off. Stage 2's
   `samples_out - 1` hack (`stage2.py:501-505`) exists only to paper over the out-side half of this.

2. **The top-up is one-sided.** `if shortfall > 0` — it only ever *adds*, never trims. It does achieve
   stub balance (`Σout = Σin = content_E`) for 8 of 9 graphs, which is its real and undersold job. But
   it cannot touch a side that *overshoots*, which is exactly aids' failure mode.

Stage-1 sums today, as a fraction of `content_E`:

| graph | Σout / E | Σin / E | in:out |
|---|---|---|---|
| aids | 0.43 | **2.47** | 5.8× |
| codex_l | 0.50 | 0.78 | 1.6× |
| dbpedia100k | 0.63 | 0.95 | 1.5× |
| hetionet | 0.60 | 0.40 | 0.7× |
| swdf | 1.00 | 1.00 | 1.0× |

**Intended outcome:** degree fits that describe entities and content edges only; a single explicit,
two-sided sum invariant on both the in- and out-side; and the deletion of both the top-up and the `-1`
hack. Degree-*shape* steering (the α-vs-p90-vs-max trade) is explicitly **out of scope** and stays as-is.

### Scope boundary (decided)

Fix **degrees + the edge budget only**. Block A's `num_relations` also counts rdf:type as a relation,
and Block D's characteristic sets include it as a member — real contamination, but out of scope here.
Record as a follow-up, do not touch. Literal-object triples are a third theoretical contaminant of
out-degree; all 9 corpus graphs have **zero** literals, so ignore.

### Working on top of in-flight edits

`_adapters.py`, `block_b.py`, `schema.py`, `stage1.py`, `stage2.py` all have uncommitted changes
(`sample_powerlaw` → `sample_powerlaw_trunc`, `obj/subj_mult_max` plumbing). The tree imports fine.
Rebase these steps onto whatever is there rather than reverting anything.

---

## Step 1 — Block B: measure entity content degrees

`src/kgsynth/signature/block_b.py`, in `calculate` (~`:159-163`).

Reuse the existing `RDF_TYPE` constant from `signature/_utils.py:12` — there is already precedent for
exactly this exclusion at `block_b.py:201` (the reciprocity guard) and `block_c.py:170`
(`edge_multiplicity` / `bidirectional_ratio`).

- Derive the class-node set: the targets of rdf:type edges (`{e.target for e in g.es if
  e["predicate"] == RDF_TYPE}`). There is no `is_class` vertex attribute and no class vocabulary —
  confirmed in `kg_io.py`; this is the only way to identify them.
- Degree node set = non-literal vertices **minus** class nodes.
- Degrees = over non-rdf:type edges only (`g.es.select(predicate_ne=RDF_TYPE)`).
- This changes `out_degree_fit`, `in_degree_fit`, `out/in_degree_max`, `out/in_degree_p90` only.
  Leave every other Block-B structure (multiplicity fits, Zipf, `a_obj`/`a_subj`) untouched — those
  are contaminated too, but that's the out-of-scope follow-up.
- Docstring: state that degrees are *entity content degrees* — class nodes and rdf:type edges excluded.

## Step 2 — Block A: make the type-edge budget derivable

`src/kgsynth/signature/block_a.py`. There is currently **no feature anywhere** recording the number of
rdf:type triples or typed entities (`num_classes` and `class_size_fit` in Block C do not let you
recover the sum). Without one, `content_E` cannot be computed correctly from a signature.

- Add one scalar — `type_edge_frac` = (rdf:type triples) / `num_triples` — to Block A's measured state,
  `as_vector()`, `feature_names()` and `_state_from_features`.
- Then `content_E = round(E · (1 − type_edge_frac))` is exact, replacing Stage 2's assumption that
  every entity carries exactly one type edge.
- Note the feature vector grows by one; check `transform/_surface.py` and `dataset/worker.py`, which
  both consume Block A features.

**Why this is load-bearing, not bookkeeping.** `mean_degree = E / V` (`block_a.py:70`) is the *only*
mean stored in the whole signature — Block B has none of its own. That is correct as it stands: each
edge contributes one out-stub and one in-stub, so mean-out = mean-in = E/V by identity, and a single
scalar legitimately serves both sides. But Step 1 changes what the degree fits describe: their mean
becomes `content_E / V_entities`, while the stored mean is still `E / V_all_non_literal`. Without
`type_edge_frac` the fit and the mean would describe **different populations** — the same category
error this plan exists to remove, merely relocated. With it, Stage 1 can reconstruct the content mean
that matches the new fits (Step 3).

Residual wrinkle, accepted knowingly: `num_entities` still counts class nodes (the scope boundary
above), so the derived content mean divides by a slightly-too-large V. Immaterial in practice — aids
has ~10 class nodes against 254k entities, ~0.004% — but it is a real inconsistency and disappears
only when the out-of-scope follow-up removes class nodes from `num_entities`.

Also note `mean_degree` is the **fourth** target constraint alongside α, p90 and max. All four are
scored by the distance metric, and they over-determine the truncated-power-law family — see
"Out of scope" below. This plan makes the *sum* (i.e. the mean) the hard constraint and leaves the
resulting shape trade untouched.

## Step 3 — Stage 1: content mean, both sides, exact sum

`src/kgsynth/generator/_adapters.py` and `stage1.py`.

- `stage1.py:308` currently passes `mean_deg = num_triples / n_ent` — the *total* budget, including
  type edges — to **both** sides. Pass the **content** mean (`content_E / n_ent`) instead. This is the
  in-side's missing counterpart to the out-side `-1` hack, and a large part of why the in column sits
  above the out column in the table above.
- In `sample_degree_sequence`, replace the one-directional zero-inflation (`_adapters.py:131-139`)
  with a **two-sided, body-confined sum repair** so the returned sequence sums to the requested budget
  *exactly*:
  - surplus → zero-inflate body entries (today's behaviour, unchanged);
  - shortfall → raise body entries;
  - never touch the tail — that is what carries p90 and max, and `tail[argmax] = round(hi)` must
    survive.
  Confining the repair to the body is what stops this from re-inflating max the way the top-up does.
- Factor it as a shared helper (e.g. `_repair_degree_sum(seq, target_sum, rng, floor=None)`) because
  Step 4 needs the same primitive with a floor.

## Step 4 — Stage 2: delete the top-up and the `-1` hack

`src/kgsynth/generator/stage2.py`.

- Delete `samples_out = np.maximum(samples_out - 1, 0)` (`:501-505`). Measured out-degrees no longer
  include the type edge, so the correction is now wrong, not merely ugly.
- `n_type_edges` (`:198`): derive from `type_edge_frac` rather than `actual_V`. If the target graph
  types only a fraction of its entities, assign types to that many (highest CS→type score first) and
  leave the rest untyped, instead of typing every entity unconditionally.
- `_sample_target_degrees` (`:473-495`): keep the rank-matching and the `floor=cs_sizes_all`; replace
  the multinomial top-up with the Step-3 repair, targeting `round(DEGREE_QUOTA_SLACK · content_E_target)`
  **exactly**, two-sided, with the floor as a lower bound when trimming.
- Add `DEGREE_QUOTA_SLACK` to `generator/_constants.py`, **default 1.0**.

**Why slack and not exact equality:** `tgt_out` is a *hard* per-node cap (`stage2.py:636-637`), and
`tgt_in`'s weight going to zero can make `_cap_redistribute` drop edges or skip a relation entirely
(`:651-652`). The `≥` in today's top-up is load-bearing headroom, not sloppiness. Keep a headroom knob;
tune it against the observable in Verification rather than guessing.

## Step 5 — Tests

`tests/test_generator_stage2.py` — two existing tests hard-code the old budget rule
(`target = schema.num_triples - schema.num_entities`) and **will fail by construction**; update both to
derive the budget from `type_edge_frac`:
- `test_content_edges_near_budget` (`:112-120`)
- `test_budget_conserved_with_reciprocity` (`:198-205`)

New coverage:
- **Block B excludes types**: fixture graph with rdf:type edges — a typed entity's measured out-degree
  excludes its type edge, and the class node is absent from the degree node set entirely.
- **The sum invariant**: `Σ tgt_out == Σ tgt_in == round(slack · content_E)` exactly, on both a typed
  and an untyped fixture. This is the property the top-up never had.
- **The aids regression**: a synthetic Block B whose `in_degree_max` is a class-node-scale outlier no
  longer yields in-targets summing to multiples of `content_E` (today: 2.47×).
- **max/p90 not inflated**: the realised `tgt_out.max()` stays at the signature's `out_degree_max`
  rather than 1.2–2.3× it.

## Verification

1. **Unit tests** — `.venv/bin/python -m pytest tests/ -q` (312 passed / 254 subtests is the current
   baseline).
2. **Two throwaway diagnostics produced the numbers in the Context section**; re-run them as the
   primary acceptance check (they live in the session scratchpad — keep them under `scripts/` if they
   prove useful beyond this change). Each loads every `data/signatures/*.json` via
   `Signature.from_features`, runs `sample_schema`, and replays Stage 2's target construction:
   - **stub balance** — `Σ tgt_out / content_E` and `Σ tgt_in / content_E` per graph. Expect both to
     equal `slack` on **all 9** graphs, including aids (today: 0.43 / 2.47).
   - **max inflation** — the signature's `out_degree_max` versus `tgt_out.max()` after the top-up.
     Expect the ratio to collapse to ~1.00× (today: 1.2–2.3×).
3. **End-to-end**, watching the Stage-2 logs, which already expose everything needed:
   - `Stage 2: degree targets — out(max=…) in(max=…)` (`:516-520`) — max should now match the
     signature's, not double it.
   - `Stage 2: deficit recovery placed %d/%d missing edges` (`:923`) — **this is the metric that sets
     `DEGREE_QUOTA_SLACK`.** It counts edges falling through to the dumb quota-weighted random
     placement. Run at slack 1.0 on wn18rr_v4 (small) and aids (the pathological case); if the deficit
     volume climbs materially versus today, raise the slack until it doesn't.
4. **Corpus regen** — the signatures in `data/signatures/` are already stale (their `out_degree_xmin`
   values are 2/4/5/7/9/172, but the current fitter pins `xmin=1` and always reports 1.0), and Steps 1–2
   invalidate them again. Regenerate before any downstream numbers are trusted, but **do not run it as
   part of this work** — it is a long job on aids (254k entities), and the CHANGELOG already tracks a
   deferred corpus regen. Fold it into that.

## Docs & changelog (per CLAUDE.md)

- `docs/generator.md` §5c (the "multinomial top-up" paragraph, lines ~245-252) and the §213 budget
  split line (`content_E = E − n_type_edges`, "one rdf:type edge per entity").
- `docs/signature.md` — Block A row (new `type_edge_frac`) and Block B row (degrees are entity content
  degrees).
- `CHANGELOG.md` under `## Unreleased`, dated 2026-07-12, `Fixed` + `Changed`. Note that this changes
  seeded output and requires the deferred corpus regen.

## Out of scope — record as follow-ups

- Block A `num_relations` counts rdf:type as a relation; Block D characteristic sets include it as a
  member (and class nodes appear as inv-CS-bearing entities). Same class of bug, wider blast radius.
- Degree-shape steering: α, p90, max and mean over-determine the truncated-power-law family (a
  PL(α=2.68) on [1,265] has mean ≈2.4 against codex_l's actual 7.86), so no sampler can honour all
  four. Today's realised α is ~1.5 against targets of 2.2–2.9. Worth a soft-target / constrained-fit
  treatment later; deliberately untouched here.
