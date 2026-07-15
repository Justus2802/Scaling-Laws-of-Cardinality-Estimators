# Plan — Stage 3's tracked clustering/assortativity diverge from ground truth

Status: **diagnosed, not designed**. Root cause confirmed and reproduced; fix
sketched but not implemented.

## Symptom

On a wn18rr_v4 signature roundtrip, the final measured `clustering_coefficient`
and `degree_assortativity` sit well off target (the motivating report: ~70% on
both), even though the Stage-3 convergence-log plots (`tri_err`, ...,
`assort_err`, `cc_err` columns) show these two specifically converging to a
tiny tracked error within a few thousand swaps. The two views disagree, and
by a lot.

## Root cause 1 (clustering coefficient) — confirmed, dominant

`CC_avg` is tracked as `Σ_v t_node[v] / denom[v] / n`, where `t_node[v]` is v's
per-node triangle count (kept exactly up to date via `_triangle_node_delta` on
every accepted swap) and `denom[v] = C(sim_deg[v], 2)`. **`sim_deg`/`denom` are
computed once at the very start of `refine()`
([stage3.py:437-439](../../src/kgsynth/generator/stage3.py#L437-L439)) and are
never updated again for the rest of the run** — confirmed by grep: those two
arrays appear nowhere else except the one read at line 1005.

The module docstring's claim — *"C(k_v, 2) denominators are invariant under
degree-preserving swaps"* — is true for each entity's **per-relation** directed
degree (which a same-relation double-edge swap trivially preserves), but
`sim_deg` is the **simple, relation-collapsed** undirected degree (distinct
neighbours, across all relations) that `denom` actually needs. A same-relation
swap changes which (subject, object) *pairs* carry that relation, and whether a
given pair is simple-graph-adjacent afterward depends on whether they already
share some *other* relation — which is not tracked or excluded here. So
`sim_deg` silently drifts from the frozen initial value as the walk progresses.

**Measured directly** (wn18rr_v4, seed 0, comparing the tracked value at the
best-seen snapshot against a fresh, independent recompute on that exact same
edge list — so this is not a snapshot-timing artifact):

| | clustering_coefficient |
|---|---|
| target | 0.0955 |
| **tracked** (`best_state.cc`, what the loss/logs report) | 0.0971 (1.7% off — looks converged) |
| **true** (fresh `igraph` recompute on the same graph) | 0.0643 (32.7% off) |
| nodes whose true `sim_deg` no longer matches the frozen initial value | **2448 / 3861 (63%)** |

63% of nodes have drifted by the end of a run — this isn't an edge case, it's
the majority of the graph. The tracked value's apparent tight convergence is
real *only relative to the stale denominators it's been dividing by the whole
time*; a fresh measurement (exactly what Block F's `calculate()` and the
signature roundtrip do) sees a materially different value.

## Root cause 2 (degree assortativity) — confirmed, different mechanism

Assortativity tracks the cross-product sum `Q = Σ_e d_u·d_v` over `und_deg`,
which the code's own comment states counts **"directed/multi edges"**
([stage3.py:436](../../src/kgsynth/generator/stage3.py#L436) and the
`und_deg` construction at
[stage3.py:503-506](../../src/kgsynth/generator/stage3.py#L503-L506)) — i.e.
every directed content edge instance, including **both** directions of a
bidirectional/reciprocal pair and any parallel same-pair edges carried by a
different relation. This *is* exactly invariant under a same-relation swap (an
entity's total incident-edge count, regardless of target, doesn't change when
only *which* object it points to changes) — so there is no drift bug here, `Q`
is tracked exactly relative to its own degree definition.

The problem is that definition is **not** the one Block F measures:
`BlockF.calculate()` computes `assortativity_degree` on
`g.as_undirected(combine_edges="first").simplify()` — a **deduplicated simple
graph**, using the distinct-neighbour degree (the same `sim_deg` CC_avg's setup
already computes correctly, just doesn't reuse). Whenever a node participates
in a bidirectional pair or a parallel multi-relation edge, `und_deg` overcounts
it relative to `sim_deg`. wn18rr_v4's target `bidirectional_ratio` is
~1.45 — a substantial fraction of pairs are reciprocal — so this is not a
corner case for this corpus.

Measured on the same run: tracked assortativity −0.0802 vs true −0.0843
(target −0.0791) — a smaller gap than clustering's on this particular seed
(6.6% true relative error vs clustering's 32.7%). Two things keep this from
being a clean, always-large number: (a) assortativity values sit close to
zero for this corpus, so a *relative*-error percentage is inherently noisy —
a modest absolute deviation swings the percentage a lot, and pre-refinement
(Stage-2-only) assortativity was measured varying `-0.0325 / -0.0046 / +0.0189`
across just 3 seeds, even changing sign; (b) the `und_deg` vs `sim_deg` gap's
*size* depends on how much bidirectionality/multiplicity the specific run
happens to realise. Averaged over the paper's 10 seeds, this combination of a
real formula mismatch plus a fragile near-zero-target relative-error metric is
a plausible explanation for a large reported average even though any single
seed's absolute deviation isn't dramatic.

## What isn't the cause (ruled out)

- **Post-hoc component bridging** (`_connect_components`, called after the SA
  walk to hit `num_components`/`largest_component_fraction`) was the first
  hypothesis — it adds edges outside the tracked incremental system, which
  looked like a strong candidate. Directly instrumented and ruled out on this
  run: the Stage-2 graph was already a single component, so it added **zero**
  edges. It remains a theoretical risk on a graph that genuinely needs
  bridging (added edges there are placed randomly, degree-blind, and equally
  untracked by CC/assortativity) but is not what's happening here.
- **Floating-point accumulation drift** — already guarded against: `candidate.cc`
  is recomputed fresh from `t_node`/`denom` on every accepted swap rather than
  repeatedly incremented
  ([stage3.py:1129-1132](../../src/kgsynth/generator/stage3.py#L1129-L1132)),
  specifically to avoid this. It doesn't touch the *denominator* staleness,
  which is the actual issue.

## Fix directions (sketched, not implemented)

- **CC_avg**: update `sim_deg`/`denom` exactly, incrementally, alongside
  `t_node`. `_adj_inc`/`_adj_dec` already distinguish a *count* change from a
  *key* change (a neighbour dict entry being created or deleted) — `sim_deg`
  only needs to move when a swap causes such a key-level change (i.e. a pair
  becomes newly simple-graph-adjacent, or stops being adjacent because no
  relation connects it anymore). This is O(1) extra bookkeeping per swap, no
  periodic full recompute needed.
- **Assortativity**: switch `und_deg`/`Q`/`S`/`T` to the same `sim_deg` basis
  once it's correctly maintained, and recompute the cross-product sum over
  **distinct pairs** rather than per-directed-edge-instance. This is a bigger
  change than CC's fix since `Q`'s delta formula currently assumes per-edge
  additivity that a dedup step breaks; would need its own delta derivation.
- Either fix should be validated by comparing tracked-vs-fresh-recompute (the
  same instrumentation used to diagnose this) before/after, on both wn18rr_v4
  and a higher-bidirectionality/higher-multiplicity graph, since assortativity's
  fix matters more exactly where this corpus's other reciprocity issues
  (`reciprocity_stub_correlation.md`) are also most visible.

## Out of scope here

- The bidirectionality/reciprocity generation shortfall itself
  (`reciprocity_stub_correlation.md`) — a Stage-2 issue, unrelated mechanism,
  though it interacts: fixing it would *increase* `bidirectional_ratio`,
  which would make the `und_deg`/`sim_deg` gap in root cause 2 larger, not
  smaller, until that's also fixed.
- `_connect_components`'s degree-/clustering-blind random bridging — a real,
  separate risk on graphs that need bridging, not exercised by this
  investigation's test run. Worth a follow-up check on a graph with multiple
  natural components.

## Validation

Re-run the same tracked-vs-fresh-recompute comparison (`best_state.cc` /
`_assort_from_Q(best_state.Q)` against an independent `igraph` recompute on
`best_content`) before and after either fix, across several seeds. Success =
the two views agree to within measurement noise, not just that the tracked
value looks converged.
