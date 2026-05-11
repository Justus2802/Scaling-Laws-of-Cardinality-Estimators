# Implementation Assumptions

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
