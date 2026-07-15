# Plan — CS templates ask for more edges than the budget has (Σ|CS(v)| > content_E)

Status: **diagnosed, not designed**. Companion to
[`cs_freq_concentration.md`](cs_freq_concentration.md) (a different bug in the same
neighbourhood — that one governs *which* templates exist and how subjects are
distributed across them; this one governs whether an entity's *assigned* template
survives intact once wiring has to live within the edge budget).

## Symptom

On a fresh `signature_roundtrip.py wn18rr_v4` (seed 42), Stage 2's own log shows the
CS-template *construction* landing almost exactly on target:

```
Stage 1: schema ready — ... cs_num_templates=43
Stage 2: group forward CS (target 43 templates, realised 42)
Stage 2: the ≥1-edge-per-CS-relation floor (Σ|CS|=12782) does not fit the edge
         budget (content_E=9842) alongside the degree tail — dropping it;
         fit_stubs still applies it per relation where affordable
```

But the graph Stage 2 actually **wires** — measured directly, before Stage 3
refinement even runs — only realises **26** distinct characteristic sets, not 42:

```python
g = Generator(target).sample_pre_refine(seed=42)   # the wired Stage-2 graph
# CS measured from g's actual edges: 26 distinct sets (target 43)
```

So the *intended* assignment (`entity_cs[v]`, before any edge is placed) is close to
correct; what's realised in the graph is not. The gap is entirely inside Stage 2 —
confirmed by measuring the pre-refinement graph directly, so this is not a
Stage-3 effect.

## Root cause

Every entity's assigned characteristic set is a **demand**: at least one edge per
relation in that set. Summed across all entities, that demand (`Σ|CS(v)|`) can exceed
the actual content-edge budget (`content_E`) once the degree tail also has to be paid
for out of the same budget. When it does, `_sample_target_degrees`
([stage2.py:528-538](../../src/kgsynth/generator/stage2.py#L528-L538)) drops the
floor as a **hard per-entity degree-target constraint** and falls back to the
unfloored degree law (`repair_degree_sum` without `floor=eff_floor`) so the sum still
lands on `content_E` — logged at [stage2.py:531-536](../../src/kgsynth/generator/stage2.py#L531-L536).
`fit_stubs` (§4, the IPF stage) still tries to honour the floor **per relation, where
it's still affordable**, but "where affordable" is the crux: it isn't affordable
everywhere, so for many entities one or more of their *assigned* relations never gets
an edge wired.

That silent per-entity, per-relation drop is what collapses the distinct-CS count.
If relation `r` fails to wire for many entities that were assigned CS templates
differing only in whether they include `r`, all of those templates collapse onto the
same smaller *realised* relation-set once `r` is gone from their actual edges — so
several originally-distinct templates end up indistinguishable in the graph Block D
measures. This is a **collapse**, not a reduction proportional to how much demand was
dropped: on wn18rr_v4 the assigned-template count (42) barely differs from target
(43), but the realised count (26) is 40% lower than the assigned count.

This is the general form of a bug already hit and worked around once before, for a
different symptom: swdf's characteristic sets asked for 606,500 CS relations against
a content-edge budget of 242,256 (2.5×over), the floor was dropped there too, and the
realised **CS *size*** collapsed from a target of 6 down to 1 (fixed separately via
`subject_frac`/`object_frac` — see the CHANGELOG's 2026-07-14 swdf entry — which
reduced *how many entities carry a CS at all*, not this budget conflict directly).
wn18rr_v4's version of the same conflict is milder in ratio (12,782 / 9,842 ≈ 1.3×)
but still large enough to more than halve the *distinct-count* fidelity that the
cross-group pool-overlap fix (this session) otherwise restored.

## Why this is a different bug from the concentration/count fix already landed

The pool-overlap fix (`_build_distinct`'s shared `seen` set, landed 2026-07-15) fixes
how many *distinct* templates exist and get assigned in `entity_cs` — the intended
assignment. It has no visibility into the edge budget at all; that's a downstream
concern of §3c/§4. Confirmed empirically: the intended assignment (42/43) is already
close to correct on wn18rr_v4 with that fix in place. The remaining loss (42 → 26)
happens entirely in the **budget-constrained wiring** stage that assignment feeds
into. Fixing the assignment stage further (e.g. the hub-reservation design in
`cs_freq_concentration.md`) will not touch this bug — it improves what's *asked for*,
not whether the budget can *deliver* it.

## Open questions for a fix (not designed here)

- **Which (entity, relation) pairs get dropped is currently an emergent side-effect
  of `repair_degree_sum`'s unfloored fallback and `fit_stubs`'s per-relation
  affordability**, not a deliberate choice. A fix likely needs to decide *which*
  relations to sacrifice per entity when `Σ|CS(v)| > content_E`, rather than letting
  it fall out of whichever repair pass happens to run.
- Options sketched, none evaluated yet:
  1. **Shrink CS templates to fit** — deflate template sizes (not just which templates
     exist) so `Σ|CS(v)|` is constrained to ≤ `content_E` at construction time, before
     any degree-target repair runs. Interacts with `cs_size_q` fidelity — shrinking
     changes the CS-*size* distribution Block D also measures, so this trades one
     fidelity for another rather than obviously fixing it.
  2. **Prioritise which relations survive** — e.g. keep the higher cs_freq-weighted /
     more frequently-reused relations per entity when a CS must be trimmed, so the
     relations that collapse most templates together are the ones least likely to be
     dropped.
  3. **Raise the effective budget** — revisit how much of `content_E` the degree tail
     reserves before the CS floor gets a claim on it; not obviously safe, since the
     degree tail's own fidelity (`max`/`p90`) was a hard-won fix earlier in this
     project (see `degree_budget_and_type_edges.md`) and re-opening that trade needs
     its own measurement pass.
- Whichever direction is chosen, validate on **both** wn18rr_v4 (mild ratio, 1.3×) and
  swdf (severe ratio, 2.5×) — a fix tuned only to the mild case may not generalise.

## Out of scope here

- The CS-frequency concentration/count fix (`cs_freq_concentration.md`) — separate,
  already diagnosed, not yet implemented.
- swdf's specific `subject_frac`/`object_frac` fix — already landed, addressed a
  different manifestation (CS *size* collapse via entity count, not this edge-count
  conflict directly).

## Validation

Re-run `signature_roundtrip.py wn18rr_v4 --seed 42` and compare Stage 2's own two log
lines (assigned-template count vs the floor-drop message) against the *realised*
`num_distinct_cs` measured post-hoc from the wired graph. Success = the gap between
"assigned" and "realised" shrinks (ideally to ~0) without regressing edge count or
degree fidelity, on both wn18rr_v4 and swdf.
