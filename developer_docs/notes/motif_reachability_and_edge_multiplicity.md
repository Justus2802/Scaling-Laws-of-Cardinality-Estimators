# Motif reachability, the ~1D swap coupling, and the edge-multiplicity gap

Why Stage-3 rewiring cannot bring the fb237-class motif errors to zero, traced to
its real root cause: **Stage-2 produces a graph with essentially no edge
multiplicity (no pair overlap), which the signature never asked it to, so its
simple-graph degree sequence is off the manifold the motif targets live on.**

This started as "design biased Stage-3 proposals to steer motifs," but the
diagnosis moved the problem upstream twice — first to the swap operator's
dimensionality, then to a signature under-specification. The bias-proposal idea is
a *third-order* lever; it can't help until the degree sequence is right.

Tools built: `scripts/edge_multiplicity.py` (the corpus survey below).

---

## 1. The per-swap motif coupling is nearly one-dimensional

Correlation of the seven per-swap motif deltas across 300 fb237 proposals
(`experiments/swap_delta_logs/`):

- **Clustering group** — triangle, paw, diamond, K4 — pairwise **+0.96…+0.99**.
- **Chordless-cycle group** — C4, 5-cycle, 6-cycle — pairwise **+0.78…+1.0**.
- **Between the groups: −0.95…−1.0.** triangle vs C4 move the same direction only
  31 % of the time; triangle vs c5/c6 only 28–31 %.

Structurally exact: the clustering motifs all contain triangles, the chordless
motifs (C4/c5/c6, induced) contain none, and a triangle is created by adding a
**chord** — which simultaneously destroys the induced cycle it cuts. So one edge
move trades one group for the other: a double-edge swap moves the graph along a
single *clustering ↔ chordless* axis. (Robust same-sign fractions show ~70 %
antagonism, not 100 %, so there is ~30 % off-axis freedom — real but small.)

**Consequence for fb237's demand.** It wants *more* triangle/C4/K4 and *fewer*
paw/c5. But triangle/K4 are clustering and C4 is chordless (−0.96 from triangle),
so no swap raises both; "fewer paw" (clustering) and "fewer c5" (chordless) also
pull opposite ends. The demand has components on **both** ends of the one axis the
operator can move — which is why uniform-proposal SA stalls at a compromise.

---

## 2. Reachability: the target lives on a different degree sequence

The motif targets were **measured on the original graph**, i.e. on the *original
degree sequence*. Maslov–Sneppen swaps are degree-preserving, so Stage-3 only
explores graphs sharing **Stage-2's** degree sequence. Those differ materially:

| fb237 simple graph | original | Stage-2 |
|---|---|---|
| distinct undirected edges | 26,757 | **33,686 (+26 %)** |
| max degree | 1050 | 1383 |
| top-5 degrees | 1050, 1028, 936, 884, 392 | 1383, 798, 601, 520, 498 |
| sorted-degree L1 vs original | — | **27.5 % of total** |

So the target is realized by a graph **outside** Stage-3's reachable set, and motif
counts are strongly degree-constrained — a target realized on sequence A is
generally not reachable on sequence B. The signature *is* realizable (the original
proves it); the guarantee just doesn't transfer, because Stage-2 hands Stage-3 the
wrong degree sequence to begin with.

---

## 3. Root cause: Stage-2 produces ~zero edge multiplicity

Both graphs carry the same directed content-edge budget (33,916), but distribute
it oppositely. The original packs edges onto shared pairs; Stage-2 scatters them:

| fb237, directed→simple | original | Stage-2 |
|---|---|---|
| ρ = directed / distinct-undirected | **1.268** | **1.007** |
| parallel (same ordered pair, ≥2 relations) | 1.102 (~10 %) | 1.006 (~0.6 %) |
| bidirectional (both a→b and b→a) | 1.150 (~15 %) | 1.001 (~0.1 %) |

Stage-2 is essentially a *simple* graph. The ~6k extra distinct edges are exactly
what inflates paw (+2×) and c5 (+2.2×) and moves the degree sequence off-manifold.
Nothing in Stage-3 can remove them — the excess edges *are* the degree sequence it
must preserve.

**It is universal across the corpus** (`scripts/edge_multiplicity.py --orig-only`;
synth ρ confirmed ≈1 on fb237, wn18rr, wn18rr_v4_ind):

| graph | orig ρ | parallel | bidir | ⇒ synth edge inflation |
|---|---|---|---|---|
| aids | **2.000** | 1.000 | 2.000 | ~2.0× |
| wn18rr_v4_ind | 1.496 | 1.002 | 1.493 | ~1.50× |
| swdf | 1.488 | 1.217 | 1.222 | ~1.49× |
| wn18rr_v4 | 1.451 | 1.002 | 1.448 | ~1.45× |
| fb237_v4_ind | 1.288 | 1.102 | 1.169 | ~1.29× |
| fb237_v4 | 1.268 | 1.102 | 1.150 | ~1.27× |
| dbpedia100k | 1.204 | 1.138 | 1.058 | ~1.20× |
| hetionet | 1.068 | 1.066 | 1.001 | ~1.07× |
| codex_l | 1.034 | 1.019 | 1.016 | ~1.03× |

Every graph overlaps; severity ranges from negligible (codex_l 1.03) to 2× (aids).
The *flavor* differs: symmetric-relation graphs (aids, wn18rr — molecular bonds,
WordNet symmetric relations) are almost purely **bidirectional**; fb237 / swdf /
dbpedia mix in **multi-relational parallel** edges. There is no "clean" corpus
graph where Stage-2's degree sequence matches — so no on-manifold test bed exists
for the bias-proposal idea until this is fixed.

---

## 4. Why: the signature never encodes pair overlap

Audit of every edge-related feature — **nothing pins the pair-level overlap:**

- **Block A** — `mean_degree = directed_content_edges / V`; the reduced block
  *dropped* `density`. Fixes the directed budget, nothing about the simple graph.
- **Block B** — degrees are `g.degree(mode="out"/"in")` on the **directed
  multigraph** (they count parallel edges); plus per-relation object/subject
  multiplicity. All directed/per-relation — a node with 5 relations to one
  neighbour has out-degree 5 here but is *one* simple edge; B pins the former.
- **Block C / D** — co-occurrence and characteristic sets describe which relations
  share an **entity**, never which share a **pair**.
- **Block E** — motifs are measured on `g.as_undirected(...).simplify()`, i.e.
  *downstream* of the simple graph; the simple edge count is never itself a target.
- **Block F** — connectivity/clustering/paths: emergent, no edge-count handle.

The simple-edge count is set by two statistics — the fraction of pairs joined by
≥2 relations, and the fraction of bidirectional pairs — and **neither is in the
signature**. The entity marginals do not pin the pair joint (knowing subject *a*
emits {r1,r2} and object *b* receives {r1,r2} does not say whether both land on the
*same* pair). So Stage-2 has no target, wires each relation to near-disjoint pairs,
and gets ρ≈1.

**The chain:** signature omits pair-overlap → Stage-2 has no target → produces ~0
multiplicity → +26 % distinct edges → wrong simple-degree sequence + inflated
paw/c5 → Stage-3 (degree-preserving) can never fix it.

---

## 5. Proposed fix — a third, pair-level co-occurrence

Pair overlap is naturally a **third kind of relation co-occurrence**, orthogonal to
the two the signature already has:

- `subj_cooc` — relations sharing a **subject**.
- `obj_cooc` — relations sharing an **object**.
- **new: `pair_cooc`** — relations sharing a **(subject, object) pair** (directed →
  multi-relational parallel; undirected → bidirectional).

It reuses Block C's machinery unchanged: build a `pair × relation` incidence and
run the same `_cooc_stats` reduction (spectrum + **density** + row-entropy). Its
*density* is the multi-relational overlap fraction; bidirectionality is a companion
scalar (distinct mechanism, so keep it separate).

Two design facts:

- **For motifs specifically, the load-bearing part is a scalar.** The simple graph
  only cares *whether* a pair is connected, not by which relation — so the total
  collapse ratio ρ (or the simple-edge count) is what removes the inflation and
  fixes paw/c5. The full pair-co-occurrence spectrum buys broader multi-relational
  fidelity but little extra for motif counts. Minimal fix = add ρ (+ a bidirectional
  fraction); principled fix = the full `pair_cooc`.
- **Measuring is trivial; generating to it is the work.** Stage-2 must correlate the
  object choices *across a subject's relations* — deliberately route a ρ-calibrated
  fraction of a subject's relations to shared targets / reverse directions. There is
  precedent: the existing subj/obj co-occurrence is already used generatively via the
  group-prototype path (`subj_group_probs`/`obj_group_probs`), so a pair-level
  analogue fits, though the mechanism is more of a shared-target assignment.

---

## 6. Fix ordering (this inverts the original plan)

1. **Add the pair-overlap target to the signature** (minimally ρ + bidirectional
   fraction; ideally the full `pair_cooc` block feature).
2. **Teach Stage-2 to hit it** (shared-target / reverse-direction wiring).
3. **Only then** are the motif targets on-manifold, most of the paw/c5 overshoot
   dissolves at the source, and Stage-3 proposal-biasing (§1's clustering↔chordless
   axis, exploiting the ~30 % off-axis moves) becomes the relevant *fine-tuning*
   lever — not the primary one.

Corollary: the earlier per-swap motif-delta work (guards, MITM, the approximate
hub delta, SA retune — see `stage3_steering_analysis.md`) optimises a stage that is
starting from the wrong graph. Worthwhile, but second-order to the multiplicity fix.

## Open threads

- Measure whether motif error tracks the edge inflation directly (regress
  paw/c5 overshoot on ρ-gap across the corpus) to confirm the causal chain
  quantitatively.
- Decide scalar-ρ vs full `pair_cooc`: is the relation identity of overlaps needed
  for any target beyond motifs (e.g. multi-relational path/tree templates)?
- Design the Stage-2 shared-target wiring and its interaction with CS/multiplicity
  (Block D / B) — the overlap must not double-count against the directed budget.
