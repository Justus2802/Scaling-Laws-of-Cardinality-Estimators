# Plan — per-relation stub balance: replace the greedy quota with a joint allocation

Status: proposed, not implemented. Successor to `degree_budget_and_type_edges.md`, which made
`Σ tgt_out == Σ tgt_in == content_E` hold *globally* and explicitly left the deficit-recovery
blow-up as a follow-up. This plan closes that follow-up by enforcing the stronger constraint the
wiring actually needs, and deletes the deficit pass rather than tuning it.

---

## 1. The finding

**Stage 2 §5c does not enforce per-relation stub balance, and structurally cannot.** It produces
exactly two numbers per entity — `tgt_out[v]`, `tgt_in[v]` — aggregated over *all* relations. Its
only guarantee is the global `Σ_v tgt_out == Σ_v tgt_in == quota_budget`. The degree targets have
no per-relation decomposition anywhere in the Schema.

Per-relation balance is *attempted* in the wiring loop (`stage2.py` §4): `m_obj` and `m_in` are both
drawn as `Multinomial(edges_r, ·)`, so at draw time both sides sum to `edges_r`. It is then **broken
immediately after** by `_cap_redistribute(..., hard_cap=…)`. Each side is independently truncated
against the *global* remaining quota of its own pool, and the closing `np.minimum(m, caps, out=m)`
drops whatever the 8 redistribution passes could not rehome. The two sides carry different caps, so
they end up with different sums, and the pairing can place at most the smaller of the two. Everything
that goes missing lands in §5a deficit recovery.

### Measured (seed 1, `sample_pre_refine`, instrumented wiring loop)

| graph | per-rel budget | out-stubs | in-stubs | Σ per-rel \|imbalance\| | placed | deficit |
|---|---|---|---|---|---|---|
| `fb237_v4` | 33 908 | 33 740 | 31 858 | 1 843 | 31 632 | 2 284 |
| `wn18rr_v4` | 9 842 | 9 229 | 9 563 | 790 | 9 001 | 841 |
| `aids` | 547 910 | 547 910 | 362 918 | **184 992** | 362 918 | ~185 000 |

The realised stub counts equal the available quota capacity of each pool to within a handful of
edges, so **the hard cap is the binding constraint** — nothing else is. For scale: pairing failures
(`MAX_PAIR_RETRY` exhaustion on duplicate `(s,o)` / self-loops) cost 226 edges on `fb237_v4` (0.7%),
and the `|S_r|·|O_r|` capacity bound cost 8. Essentially the entire deficit is stub imbalance.

### Two distinct root causes

**(a) `fb237_v4` / `wn18rr_v4` — a processing-order artifact.** Relations are iterated in *index*
order and eat the global quota greedily. Of `fb237_v4`'s 219 relations, 46 starve, and they are
almost all high-index (177–218); the first ~170 have generous headroom. Relation index is unrelated
to relation weight — `r=177` is the *largest* relation (5 682 edges) and it starves. So which
relations get their degree law honoured is currently a function of array position.

**(b) `aids` — a hard infeasibility, and it is one relation.** `r=1` is allocated 374 355 edges, but
its object pool `O_1` (72 851 objects, from the sampled inverse CS) holds a total in-degree quota of
189 363. The gap, 374 355 − 189 363 = **184 992**, is *exactly* the whole imbalance. That relation
would need a mean in-multiplicity of 5.14 from a degree law whose max is 11 and mean is 2.4.

Two things compound to produce it:

- `relation_zipf` is NaN on `aids` (R=5 — too few points to fit), so Stage 1 falls back to the
  hard-coded `DEFAULT_ZIPF_EXPONENT = 2.0` (`stage1.py:237-243`), handing the top relation **68%**
  of content edges. The real `aids` top relation has **49%** (≈ Zipf s=1.2). Measured real
  per-relation content edges / object-pool sizes:

  | predicate | E_r | \|O_r\| | mean in-mult |
  |---|---|---|---|
  | `edge0` | 269 960 | 156 894 | 1.72 |
  | `edge3` | 227 284 | 108 517 | 2.09 |
  | `edge1` | 48 010 | 46 811 | 1.03 |
  | `edge2` | 2 656 | 2 656 | 1.00 |

  (`num_relations = 5` also still counts `rdf:type` as a content relation, so a phantom fifth
  relation is being wired — the known follow-up.)
- Nothing calibrates `|O_r|` against `edge_budget[r]`. The real top relation has 156 894 objects; the
  sampled inverse CS gave `r=1` 72 851.

### What is *not* the problem

`repair_degree_sum` inside §5c is, measured, nearly a no-op. Stage 1's `sample_degree_sequence`
already returns sequences summing exactly to `content_E` (verified: `Σ out == Σ in == content_E`
exactly on `fb237_v4`, `wn18rr_v4`, `codex_l`). So Stage 2's **in-side repair delta is 0**, and the
out-side repair only trims back what `floor=cs_sizes_all` added. Rank-matching is a permutation:
sum-neutral, `O(V log V)`. Neither is expensive; neither is where the damage is.

The expensive and harmful pass is **§5a deficit recovery**. On `aids` it runs ~185 000 iterations,
each an `rng.choice` over a 70–200k pool *with a freshly normalised probability vector* — `O(deficit
× |pool|)`, which is where `aids`' Stage-2 slowness comes from. It is harmful because once the quota
is exhausted its weight `max(quota, 0) + 1e-3` degenerates to **uniform**: a third of `aids`' content
edges get placed with no multiplicity law, no preferential attachment and no degree law.

---

## 2. Design

### 2.1 Relation frequency: replace the Zipf with a log-space quantile fit

Replace `relation_zipf: ZipfFit` with `rel_freq_logq: QuantileFit` — the empirical quantile function
of `log(E_r / Σ E_r)` over the R predicates, via `fit_quantiles(log_shares, min_samples=2)`. Stage 1
rebuilds `relation_weights` by evaluating it at R evenly-spaced levels (reconstructing the rank curve
directly), exponentiating, and renormalising.

No goodness-of-fit gate, no Zipf fallback. A gate was built and **it degenerates to a constant** — the
quantile fit wins on all 9 corpus graphs, on both log-share RMSE and top-share error:

| graph | R | current (Zipf) | top err | OLS rank exp. | top err | **log-quantile** | top err |
|---|---|---|---|---|---|---|---|
| `aids` | 5 | 1.199 | 103% | 1.185 | 122% | **0.000** | 0.0% |
| `codex_l` | 69 | 2.511 | 32% | 2.435 | 145% | **0.534** | 28% |
| `dbpedia100k` | 470 | 2.429 | 97% | 4.091 | 708% | **0.480** | 37% |
| `fb237_v4` | 219 | 0.597 | 184% | 0.798 | 455% | **0.152** | 6.7% |
| `fb237_v4_ind` | 200 | 0.581 | 145% | 0.743 | 378% | **0.189** | 9.4% |
| `hetionet` | 24 | 1.804 | 6.5% | 1.326 | 184% | **0.252** | 11% |
| `swdf` | 170 | 2.806 | 94% | 3.839 | 740% | **0.149** | 6.0% |
| `wn18rr_v4` | 9 | 1.371 | 11% | 1.029 | 45% | **0.287** | 7.3% |
| `wn18rr_v4_ind` | 9 | 0.804 | 0.0% | 0.400 | 21% | **0.119** | 4.0% |

Two findings underneath this.

**There is a unit bug in the current path.** `fit_zipf` calls `_fit_powerlaw(counts)`, which fits the
**count distribution** `P(count = x) ∝ x^(−α)` with `xmin` pinned to 1. Stage 1 consumes that α as a
**rank-frequency** exponent — `ranks ** (-exponent)` (`stage1.py:243`). Those are different laws (a
rank-Zipf with exponent `s` has `α = 1 + 1/s`). And the fitted α is **pinned at ≈1.0** on all six
graphs where it fits at all (1.000, 1.068, 1.000, 1.000, 1.000, 1.014), so `relation_weights ≈ rank⁻¹`
on *every* corpus graph regardless of its actual shape. The measured exponent carries essentially no
information — which is why `fb237_v4`'s top relation is handed 16.8% of the budget against a real
share of 5.9%.

**An OLS rank-exponent fix is worse, not better.** It is the natural "just fix the units" repair, and
it loses on 8 of 9 graphs (top-share error up to 740%). The reason is that these curves are simply not
Zipf-shaped: `aids`' shares are `.337 / .284 / .230 / .060 / .003` — a flat head, then a cliff — and no
single exponent reproduces that (Zipf(2.33) at R=5 puts 0.75 on the top relation). A non-parametric
quantile function is the right tool, and it is the *only* one of the three that is right at both
R=5 and R=470.

Feature-vector impact: −2 (`relation_zipf_exponent`, `relation_zipf_xmin`) +7 → **127 → 132**; the
surface trades one scalar knob for one coupled quantile knob → **79 → 85**. Touch points:
`signature/block_b.py` (fit, `as_features`, `from_features`, `feature_names`, `_distance` entry,
`summary`, plot overlay), `transform/_surface.py` (`COUPLED` += `_q_group("rel_freq_logq")`,
`_SURFACE_B`), `_domains.py` (drop `relation_zipf_xmin` from `INTEGER_FEATURES` / `MIN_ONE` /
`WEAKLY_EXTENSIVE`), `stage1.py:237-243`, and the `TestSurface` count assertions.

#### Do not select this on deficit

Measured deficit under the log-quantile weights: `aids` 184 992 → **34 354** (−81%), `wn18rr_v4`
800 → **457**, but `fb237_v4` 2 284 → **3 265** (*up*). That last number is not evidence against the
fit. `relation_weights` feeds *back* into the CS pools (`subj_group_probs` / `obj_group_probs` are
built by multiplying through by it), so a more faithful — flatter, longer-tailed — weight vector
spreads the budget across many small relations with small pools, which the **greedy sequential
allocator** then starves. The deficit rose because the allocator is broken (§1, cause (a)), not
because the weights got worse.

Deficit measures wiring feasibility, not fidelity. Select the relation-frequency law on reconstruction
error (the table above) and let §2.2 absorb the feasibility. Concretely: **§2.1 and §2.2 must land
together, or §2.2 first.** Landing §2.1 alone would improve relation-frequency fidelity while making
`fb237_v4`'s deficit worse.

### 2.2 Joint stub allocation by IPF (replaces §5c's caps and §4's per-relation multinomials)

Solve the allocation **once, jointly**, instead of greedily per relation. Find integer matrices
`X[v,r]` (out-stubs) and `Y[v,r]` (in-stubs) with:

- **row margins** `Σ_r X[v,r] = tgt_out[v]`, `Σ_r Y[v,r] = tgt_in[v]` — the degree law, hit exactly,
  with no post-hoc repair;
- **column margins** `Σ_v X[v,r] = Σ_v Y[v,r] = e'_r` — the **per-relation stub balance**, satisfied
  by construction on both sides;
- **support** `X[v,r] = 0` unless `r ∈ CS(v)`; `Y[v,r] = 0` unless `r ∈ invCS(v)`;
- **floor** `X[v,r] ≥ 1` on support (the ≥1-edge-per-CS-relation rule — now a margin precondition
  rather than a clamp applied afterwards).

Seed the matrices with **exactly the weights the loop already uses** —
`powerlaw(α_obj_r) · cs_size^a_obj` on the out side, `powerlaw(α_subj_r) · inv_cs_size^a_subj` on the
in side — then run **Sinkhorn/IPF**: alternately rescale rows to the degree targets and columns to
`e'`. By Sinkhorn's theorem this converges to the unique matrix with those margins in the same
cross-ratio class as the seed, i.e. it **preserves the weighting law while forcing both margins**.

Handle the floor by pre-subtracting: `X = 1_support + X'`, with row margins `tgt_out[v] − |CS(v)|`
(non-negative — `floor=cs_sizes_all` already guarantees it) and column margins `e'_r − |S_r|`. Note
`e'_r ≥ |S_r|` is a real precondition; the current code already has this problem and takes a separate
branch when `edges_r < n_sr`. Prototype must confirm how often it binds.

Cost: the support is sparse, `nnz = Σ_v |CS(v)|` ≈ 20k on `fb237_v4`, ~500k on `aids`. Fifty IPF
iterations is two scatter-adds over `nnz` — single-digit milliseconds, against the deficit pass it
deletes. Use `scipy.sparse` CSR + `np.add.reduceat` / `bincount`; no hand-rolled math.

### 2.3 Infeasibility policy: shrink `e'_r`, redistribute — which IPF gives for free

The chosen policy when the margins are inconsistent (the `aids` case) is to **shrink the
over-subscribed relation's edge budget and redistribute the surplus to relations with slack**. This
falls out of the IPF formulation rather than needing a separate mechanism:

- Row normalisation enforces `Σ_r Y[v,r] = tgt_in[v]` *exactly*, so every entity's in-quota is
  conserved and pushed into whichever of its eligible relations has capacity.
- Read the achieved column margins after the final row step. They **automatically** sum to
  `Σ_v tgt_in[v] = content_E`, so **there is no deficit, ever** — the budget is fully placed by
  construction.
- Where a relation's pool cannot absorb its Zipf/quantile budget, its achieved column is smaller, and
  the difference has already flowed to relations whose pools have room.

The two sides must agree on `e'`, and *which* side binds differs by graph (`aids`/`fb237_v4`: the in
side; `wn18rr_v4`: the out side). So run an **outer coordinate-descent loop on `e'`**, initialised at
the quantile-derived `edge_budget`:

1. Row-normalised IPF on each side → achieved columns `a_out`, `a_in`.
2. `e'_r ← min(a_out_r, a_in_r)`.
3. Redistribute the shortfall `content_E − Σ e'_r` across relations in proportion to their residual
   headroom `min(out_head_r, in_head_r)`.
4. Repeat to a fixpoint (expect ~10 iterations; each is milliseconds).

Log the final `e'` against the target `edge_budget` so a shrunk relation is **visible**, not silent —
this is the `aids` diagnosis surfacing as a first-class output instead of 185k edges quietly
disappearing into a uniform draw.

Note this deliberately spends the relation-weight law to buy the degree law. On `aids` that is the
right trade: the relation weights there come from a hard-coded fallback for a fit that did not
converge, i.e. they are the least trustworthy of the three laws in play. §2.1 should reduce how often
the trade is needed at all.

### 2.4 Rounding

The IPF solution is fractional. Round with column margins preserved exactly (largest-remainder down
each column gives `Σ_v X[v,r] = e'_r` exactly, so **stub balance survives rounding**), leaving row
residuals of at most ±1 per entity. Fix those with a bounded ±1 transportation repair along
alternating paths, or accept ±1 on the degree target — prototype should measure whether the repair is
worth its complexity.

### 2.5 Pairing

With balanced stubs, the residual loss is the configuration-model endgame: ~0.7% (226/33 916 on
`fb237_v4`) lost to `MAX_PAIR_RETRY` exhaustion on duplicate `(s,o)` and self-loops. Replace the
shuffled-reservoir draw with **decreasing-remaining-stub order** (bipartite Havel–Hakimi), which
provably realises any bipartite-graphical degree sequence; shuffle within equal-remaining ties to keep
the randomisation. Phase A (mutual pairs) and the parallel-overlap bias run first, unchanged.

### 2.6 What gets deleted

- **§5a deficit recovery** — the entire pass. With §2.3 the budget is placed by construction.
- **`_cap_redistribute`'s `hard_cap` path** — the per-node quota is now a margin, not a clamp. The
  `|O_r|` / `|S_r|` side cap becomes a box constraint (clip after each column step); it is nearly
  never binding (8 edges on `fb237_v4`, 0 elsewhere), so a clip is enough.
- **`DEGREE_QUOTA_SLACK`** — it exists only to trade deficit volume against degree fidelity. With no
  deficit there is nothing to trade, and the edge-conservation cliff at slack > 1.0 goes with it.
- **The per-relation `m_obj`/`m_in` multinomials** — subsumed by the IPF columns.
- Stage 2's `repair_degree_sum` call in §5c *may* survive, but only to enforce
  `tgt_out[v] ≥ |CS(v)|` budget-neutrally. It is cheap and correct; leave it.

### 2.7 Rank-matching

Keep it, but the IPF framing makes it much less load-bearing. Its real cost was never the repair —
it is that `floor=cs_sizes_all` forces a *perfect* rank correlation between CS size and out-degree
(Spearman ≈ 1), far stronger than anything measured. `a_obj` / `a_subj` (G2b) already carry the
CS-size↔degree coupling *inside the IPF weight matrix*, where IPF will honour it while still hitting
both margins. So once IPF is in, the out-side rank match can be softened to match the in-side (which
already adds a uniform tiebreak) — or dropped to a random assignment — and the correlation should
survive. **Measure this in the prototype before changing it**; do not change it blind.

---

## 3. Verification

- Assert `Σ_v X[v,r] == Σ_v Y[v,r] == e'_r` for every relation — the constraint this plan exists to
  establish, and a cheap unit test on synthetic margins.
- Deficit → 0 on all 9 corpus graphs (currently 2 284 / 841 / ~185 000).
- Realised degree fidelity must not regress: `max-out`, `max-in`, `p90` against target on every
  graph, versus the numbers recorded in `degree_budget_and_type_edges.md`.
- `aids`: `e'` for the top relation should land near its pool capacity, and the log should say so.
  The realised in-degree max must stay at ~10–11 (target 11) — no return of the phantom mega-hub.
- Stage-2 wall time on `aids` should drop sharply (the deficit pass is the hot loop).
- Motif / clustering / assortativity deltas on all 9 graphs: a third of `aids`' edges move from a
  uniform draw to the multiplicity/PA law, so these *should* change. Verify they move toward target.

## 4. Out of scope

- Block A's `num_relations` still counts `rdf:type` as a relation, and Block D's characteristic sets
  still include it as a member. §2.1 reduces the *consequences* on `aids` but does not fix the cause.
- Degree *shape*: α, p90, max and mean still over-determine the truncated power-law family. The sum
  remains the hard constraint; the shape trade is unchanged.
- Stage 3 refinement is untouched.
