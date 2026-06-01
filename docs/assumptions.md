# Implementation Assumptions

## Block A — Size and Density

### Literals are excluded from |V|
`num_entities` counts only vertices whose `is_literal` attribute is `False`. RDF literals (strings, numbers, dates) are objects in triples but are not knowledge-graph entities in the graph-theoretic sense. The spec defines |V| as "distinct entities (subjects ∪ objects, **excluding literals**)".

### |E| counts all edges, including rdf:type triples
No predicate is special-cased. `g.ecount()` includes `rdf:type`, `owl:sameAs`, and all other predicates. This is consistent with the spec definition "|E| = number of triples" and avoids a decision that would need to be replicated in every other block.

### |R| counts distinct predicate URIs, including rdf:type
`len(set(g.es["predicate"]))` treats `rdf:type` as an ordinary relation. Excluding it would change relation-reuse and density in hard-to-predict ways. Block C is the designated place for type-specific measurement.

### Density uses |V|² (not |V|(|V|−1)) as denominator
`density = |E| / |V|²` follows the standard definition for directed graphs where self-loops are permitted. The off-by-one correction `|V|(|V|−1)` is only conventional for simple undirected graphs without self-loops.

### Division-by-zero falls back to 0.0, not NaN
When |V| = 0 or |R| = 0, derived ratios (density, triples_per_entity, relation_reuse) return `0.0`. An empty graph is a degenerate but valid input; returning `0.0` keeps `as_vector()` free of NaN entries for downstream numeric use.

---

## Block C — Schema and Relation Correlation

### Subject-side and object-side matrices are built independently
`M_subj[i,j]` counts subjects that use both r_i and r_j; `M_obj[i,j]` counts objects that appear as the target of both r_i and r_j. These capture different aspects of schema correlation (who tends to emit relation pairs vs. who tends to receive them) and are not interchangeable.

### Object-side co-occurrence includes literal targets
`obj_to_rels[e.target]` is populated for every edge target, including literals. A literal object "receives" a relation the same way as a URI — its relation-multiplicity on the object side is 1 per incoming predicate. Excluding literals would silently drop all datatype-property co-occurrences (e.g. `label` and `comment` always co-occurring on the same object string).

### Co-occurrence matrix counts entities, not edge occurrences
For each entity, only the **set** of relations it uses is recorded (`subj_to_rels[e.source].add(ri)`). If a subject has five triples with predicate r_i and three with r_j, M[i,j] is incremented by 1, not 5 or 3. This matches the spec definition: M[i,j] = |{s : ∃o₁o₂. (s,r_i,o₁) ∈ G ∧ (s,r_j,o₂) ∈ G}|.

### Top-k singular values are padded to exactly _TOP_K_SV = 10
When |R| < 11, `svds` is called with `k = min(10, |R|−1)` and the result is zero-padded to length 10. This keeps `as_vector()` length constant regardless of how few relations the graph has.

### Row entropy treats the co-occurrence row as a probability distribution
Each row M[i, :] is normalised to sum to 1 before computing Shannon entropy. This measures how spread out relation i's co-occurrences are, independent of the raw co-occurrence counts. A row of all zeros (relation never co-occurs with anything) is assigned entropy 0.

### Type statistics require rdf:type triples; absence yields zero classes
If no edges carry the predicate `RDF_TYPE`, `class_sizes` is empty, `num_classes = 0`, and `class_size_zipf_exponent = nan`. Downstream consumers should treat `nan` as "no type information available" rather than as an error.

### P(r | type) counts relation uses with repetition across all typed subjects
`type_rel_counts[t][r]` is incremented once per outgoing edge of each typed subject. If entity A has type Person and has three `name` edges, it contributes 3 to `type_rel_counts["Person"]["name"]`. This weights relations by how heavily they are used, not merely whether they appear.

---

## Block E — Motifs

### All motif counts use the undirected simplification of the directed KG
`g.as_undirected(combine_edges="first").simplify()` is computed once and reused for triangle counting, 4-node RANDESU (diamond, K4, tailed-triangle), star counting, and 4/5/6-cycle estimation. Multi-edges (multiple predicates between the same node pair) are collapsed to one; self-loops are removed. BGP query shapes in the spec are drawn without direction arrows, so the undirected structure is the correct substrate for motif counting.

### Triangle count uses the sparse A ⊙ A² identity
`A.multiply(A @ A).sum() // 6` counts triangles without materialising A³. Each triangle appears exactly 6 times in trace(A³) (3 vertices × 2 traversal directions), so dividing by 6 gives the exact count. The sparse element-wise multiply avoids the O(n²) dense materialisation.

### 4-node motif indices are discovered at runtime, not hardcoded
`_4node_motif_index_map()` creates one canonical 4-node example for each named motif (diamond, K4, tailed triangle) and runs `motifs_randesu(size=4)` on it to find which list position holds count = 1. The result is cached with `@lru_cache`. This avoids a dependency on igraph's internal canonical ordering, which is not guaranteed stable across versions. The 4-cycle is excluded here because it is estimated by the same random-walk closure sampler as 5- and 6-cycles (see below).

### 4-, 5-, and 6-cycle counts are sampled estimates, not exact values
Exact enumeration of simple cycles of length ≥ 4 is intractable for large KGs, and even the RANDESU 4-cycle count must fall back to probabilistic branch cuts above `_RANDESU_EXACT_LIMIT` vertices. Using sampling for all three cycle lengths keeps their accuracy and runtime behaviour consistent. `_estimate_k_cycle` samples simple random walks of length k−1 and checks closure, then scales the closure rate by `n × avg_deg^(k−1) / (2k)`. The three calls share the per-block `sample_budget`, with each cycle length receiving `sample_budget // 3` walks. This is accurate in order of magnitude for sparse graphs; the scaling degrades for dense or highly heterogeneous degree distributions.

### Star counts use the exact combinatorial formula C(deg(v), k)
`_count_stars` computes `Σ_v C(deg(v), k)` for k = 2..10 using `math.comb`. This is exact: every k-subset of a vertex's neighbours is a distinct k-star subgraph centred at that vertex. Degrees are taken from the undirected simplification so that multi-edges do not artificially inflate star counts.

### Path templates follow directed edges and stop at literal targets
`_build_out_adj` only adds edges whose **target** is not a literal. A walk that would enter a literal node is a dead end (literals have no outgoing edges), so stopping there avoids wasted samples. The walk still records the relation that led to the literal.

### Path template sampling uses a fixed seed for reproducibility
`np.random.default_rng(1)` seeds the sampler for path/tree templates. Motif-count sampling uses seed 0. Fixing seeds makes signatures deterministic across runs on the same graph, which is required for signature comparison and regression testing.

### Depth-2 tree template is the sorted tuple of all (r1, r2) pairs from a root
A sampled root's template is `tuple(sorted([(r1, r2) for child, r1 in adj1 for _, r2 in adj2]))`. Sorting makes the template order-independent (the root's children are not ordered). Roots with no two-hop paths contribute nothing to the distribution.


---

## Block D — Characteristic Sets

### CS definition includes edges to literals
`CS(s) = { p : ∃o. (s,p,o) ∈ G }` is evaluated over **all** outgoing edges of `s`, including those whose object is an RDF literal. The alternative would be to restrict to non-literal objects, but the original Neumann & Moerkotte definition makes no such distinction.

### Subjects are never literals
`_compute_cs` applies no `is_literal` guard on the source vertex. This relies on the contract enforced by `kg_io.load_kg`: the RDF data model does not permit literals as triple subjects, so every edge source is guaranteed to be a non-literal node.

### Inverse CS excludes literal targets
`_compute_inv_cs` only records entries for non-literal target vertices. A literal node (e.g., an integer or a string value) cannot meaningfully act as the "object-side star centre" that the inverse CS is intended to characterise.

### Two-step pairs require a non-literal bridge entity
In `_two_step_pair_stats`, `in_preds[e.target]` is only populated when the target is not a literal, consistent with the inverse CS assumption above. A bridge entity `x` in `s →[q]→ x →[p]→ o` must be an IRI or blank node that can itself have outgoing edges.

### Entities with no outgoing edges are absent from `cs_of`
`_compute_cs` builds `cs_of` by accumulating predicates from `g.es`; vertices with zero outgoing edges never appear as keys. Such vertices therefore do not contribute to CS size statistics or the frequency distribution. This is intentional: an entity with no outgoing predicates has an empty CS, which carries no signal for star-query cardinality estimation.

### `_fit_powerlaw` used for all frequency fits
All three Zipf/power-law fits (forward CS frequency, inverse CS frequency, two-step pair frequency) use Block B's `_fit_powerlaw` helper (backed by `powerlaw.Fit`) rather than Block C's Hill-estimator `_fit_zipf_mle`. This gives KS distances for distribution comparison and consistent NaN-guarding via `MIN_SAMPLES_FOR_FIT`.

### `as_vector()` exposes only `.alpha` from each `PowerLawStats`
The full `PowerLawStats` (alpha, xmin, KS distance, and three alternative-distribution distances) is stored in the dataclass for downstream inspection, but only `.alpha` enters the fixed-length vector. Including all six fields per fit would triple the pair-related vector entries without clear benefit for cross-KG comparison.

### Top-k pair frequencies are normalised by total pair count
`top_pair_freqs` stores counts divided by the total number of (bridge-entity, in_pred, out_pred) combinations, not raw counts. Normalisation makes the vector comparable across KGs of different sizes.

### `rdf:type` edges are treated as ordinary predicates in CS
No special handling is applied to `rdf:type` triples in Block D. A subject's type assertions contribute their predicate to its CS the same as any other relation. Block C handles type-specific statistics separately.

---

## Block F — Connectivity

### Weak (not strong) connectivity for components and LCC
`g.connected_components(mode="weak")` is used instead of `mode="strong"`. Real KGs are almost never strongly connected but are typically one large weakly connected component. Weak connectivity matches the standard graph-database notion of reachability used by the spec.

### LCC fraction denominator includes literal vertices
`largest_component_fraction = lcc.vcount() / g.vcount()` counts all vertices (including RDF literals) in both numerator and denominator. The LCC is computed on the full graph, so literals are members of components and belong in the count. This differs from Block A's `num_entities`, which excludes literals.

### Shortest-path sampling uses undirected BFS within the LCC
`lcc.distances(source=..., target=..., mode="all")` treats edges as undirected. This guarantees that every (src, tgt) pair within the weakly connected LCC has a finite distance. Directed BFS (`mode="out"`) would leave many pairs unreachable in a typical KG, making the average meaningless.

### Sampling: 10^k independent pairs with replacement via a deduplicated matrix call
`n_samples = 10 ** sample_k` (src, tgt) pairs are drawn independently and with replacement from the pool of non-literal vertices in the LCC. Sources and targets are then deduplicated so that a single `distances()` call returns the full result matrix; individual pair distances are looked up by index. This avoids `n_samples` separate distance calls while keeping the pairs statistically independent.

### Self-pairs (distance == 0) are excluded from the average
When src == tgt is drawn (possible because sampling is with replacement), the distance is 0. These entries are filtered with `pair_dists > 0` before computing the mean, so the average reflects only actual between-entity distances.

### Literal vertices are excluded from the sampling pool
Only `lcc.vs` entries with `is_literal == False` are eligible as sources or targets. Literals always have in-degree > 0 and out-degree 0 in RDF, so including them as sources would produce only self-pairs or dead ends, biasing the average upward.

### Clustering coefficient uses the undirected simplification
`g.as_undirected(combine_edges="first").simplify()` is computed before calling `transitivity_avglocal_undirected(mode="zero")`. The undirected simplification is shared with Block E and is the standard definition of local clustering. `mode="zero"` assigns 0 (rather than NaN) to degree-0 and degree-1 vertices so the average is always finite.

### Degree assortativity uses total (undirected) degree
`g_und.assortativity_degree(directed=False)` computes the Pearson correlation between the total degrees of the two endpoints of each undirected edge. The directed alternative (`directed=True`) correlates out-degree of source with in-degree of target, which conflates structural assortativity with the subject/object role split of RDF. The undirected version is used for now.

### Bootstrap SE is configurable via `n_bootstrap`
`block_f` accepts `n_bootstrap: int = _N_BOOTSTRAP` (default 999). `scipy.stats.bootstrap` is called with `n_resamples=n_bootstrap` and a fixed `rng=42` for reproducibility. Warnings from scipy (e.g. the BCa degeneracy warning that fires when all sampled distances are equal) are suppressed with `warnings.catch_warnings` — consistent with how Block B silences powerlaw output. When `finite` has fewer than 2 values, the SE is set to NaN without calling bootstrap.
