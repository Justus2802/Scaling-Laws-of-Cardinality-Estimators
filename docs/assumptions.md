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
