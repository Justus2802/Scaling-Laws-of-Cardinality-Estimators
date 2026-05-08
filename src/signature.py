"""Graph signature measurement for KGs loaded via kg_io.load_kg."""

from collections import defaultdict
from dataclasses import dataclass

import igraph
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_TOP_K_SV = 10  # number of singular values to keep


@dataclass
class BlockA:
    """Block A — Size and density features of a KG."""
    num_entities: int       # |V|  distinct non-literal nodes
    num_triples: int        # |E|  total triples
    num_relations: int      # |R|  distinct predicates
    density: float          # |E| / |V|^2
    triples_per_entity: float  # |E| / |V|
    relation_reuse: float   # |E| / |R|

    def as_vector(self) -> list[float]:
        return [
            float(self.num_entities),
            float(self.num_triples),
            float(self.num_relations),
            self.density,
            self.triples_per_entity,
            self.relation_reuse,
        ]


def block_a(g: igraph.Graph) -> BlockA:
    """Compute Block A (size and density) of the graph signature.

    Literals are excluded from the entity count, matching the definition
    |V| = distinct subjects ∪ objects excluding RDF literals.
    """
    num_entities = sum(1 for v in g.vs if not v["is_literal"])
    num_triples = g.ecount()
    num_relations = len(set(g.es["predicate"])) if num_triples > 0 else 0

    density = num_triples / (num_entities ** 2) if num_entities > 0 else 0.0
    triples_per_entity = num_triples / num_entities if num_entities > 0 else 0.0
    relation_reuse = num_triples / num_relations if num_relations > 0 else 0.0

    return BlockA(
        num_entities=num_entities,
        num_triples=num_triples,
        num_relations=num_relations,
        density=density,
        triples_per_entity=triples_per_entity,
        relation_reuse=relation_reuse,
    )


# ---------------------------------------------------------------------------
# Block C — Schema and relation correlation
# ---------------------------------------------------------------------------

@dataclass
class BlockC:
    """Block C — Schema and relation correlation features of a KG."""
    # Subject-side co-occurrence matrix: M[i,j] = #subjects using both r_i and r_j
    subj_singular_values: np.ndarray      # top-k SVs, shape (_TOP_K_SV,), zero-padded
    subj_cooc_density: float              # fraction of nonzero entries
    subj_row_entropies: np.ndarray        # per-relation row entropy, shape (|R|,)

    # Object-side co-occurrence matrix: M[i,j] = #objects appearing with both r_i and r_j
    obj_singular_values: np.ndarray
    obj_cooc_density: float
    obj_row_entropies: np.ndarray

    # Type statistics (from rdf:type triples)
    num_classes: int                      # |T|
    class_size_zipf_exponent: float       # MLE power-law exponent of class-size distribution
    class_sizes: dict                     # class URI -> entity count
    # P(r | type): for each type, probability distribution over outgoing relations
    type_relation_conditional: dict       # {type_uri: {relation_uri: probability}}

    def as_vector(self) -> list[float]:
        subj_ent = float(np.mean(self.subj_row_entropies)) if self.subj_row_entropies.size else 0.0
        subj_ent_std = float(np.std(self.subj_row_entropies)) if self.subj_row_entropies.size else 0.0
        obj_ent = float(np.mean(self.obj_row_entropies)) if self.obj_row_entropies.size else 0.0
        obj_ent_std = float(np.std(self.obj_row_entropies)) if self.obj_row_entropies.size else 0.0

        # mean entropy of P(r | type) across all types
        type_rel_entropies = []
        for dist in self.type_relation_conditional.values():
            p = np.array(list(dist.values()), dtype=float)
            p = p[p > 0]
            if p.size:
                type_rel_entropies.append(-float(np.sum(p * np.log(p))))
        mean_type_rel_ent = float(np.mean(type_rel_entropies)) if type_rel_entropies else 0.0

        return (
            list(self.subj_singular_values)
            + [self.subj_cooc_density, subj_ent, subj_ent_std]
            + list(self.obj_singular_values)
            + [self.obj_cooc_density, obj_ent, obj_ent_std]
            + [float(self.num_classes), self.class_size_zipf_exponent, mean_type_rel_ent]
        )


def _build_cooc_matrix(
    entity_to_rels: dict,
    num_relations: int,
) -> scipy.sparse.csr_matrix:
    """Build a relation co-occurrence count matrix from a mapping entity -> {rel_idx}."""
    rows, cols, data = [], [], []
    for rel_set in entity_to_rels.values():
        rel_list = list(rel_set)
        for ri in rel_list:
            for rj in rel_list:
                rows.append(ri)
                cols.append(rj)
                data.append(1)
    if not rows:
        return scipy.sparse.csr_matrix((num_relations, num_relations), dtype=np.int32)
    return scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_relations, num_relations),
        dtype=np.int32,
    )


def _cooc_stats(
    M: scipy.sparse.csr_matrix,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (top-k singular values padded to _TOP_K_SV, density, row entropies)."""
    n_rows, n_cols = M.shape
    total_cells = n_rows * n_cols
    density = M.nnz / total_cells if total_cells > 0 else 0.0

    # Top-k singular values
    k = min(_TOP_K_SV, min(n_rows, n_cols) - 1)
    svs = np.zeros(_TOP_K_SV)
    if k > 0 and M.nnz > 0:
        computed = scipy.sparse.linalg.svds(
            M.astype(float), k=k, return_singular_vectors=False
        )
        computed = np.sort(computed)[::-1]
        svs[:len(computed)] = computed

    # Row entropies — iterate row by row to stay memory-efficient
    row_entropies = np.zeros(n_rows)
    for i in range(n_rows):
        row = np.asarray(M.getrow(i).todense(), dtype=float).ravel()
        s = row.sum()
        if s > 0:
            p = row / s
            p = p[p > 0]
            row_entropies[i] = -np.sum(p * np.log(p))

    return svs, density, row_entropies


def _fit_zipf_mle(sizes: np.ndarray) -> float:
    """MLE power-law exponent via the Hill estimator (α = 1 + n / Σ ln(x_i/x_min))."""
    sizes = sizes[sizes > 0]
    if sizes.size < 2:
        return float("nan")
    x_min = float(sizes.min())
    return 1.0 + sizes.size / float(np.sum(np.log(sizes / x_min)))


def block_c(g: igraph.Graph) -> BlockC:
    """Compute Block C (schema and relation correlation) of the graph signature.

    Builds subject-side and object-side relation co-occurrence matrices, summarises
    them by top-10 singular values, density, and per-row entropy, then extracts
    type statistics from rdf:type triples.
    """
    predicates = g.es["predicate"] if g.ecount() > 0 else []
    unique_rels = sorted(set(predicates))
    rel_idx = {r: i for i, r in enumerate(unique_rels)}
    num_relations = len(unique_rels)

    # Build entity -> relation-index sets for subject and object sides
    subj_to_rels: dict = defaultdict(set)
    obj_to_rels: dict = defaultdict(set)
    for e in g.es:
        ri = rel_idx[e["predicate"]]
        subj_to_rels[e.source].add(ri)
        obj_to_rels[e.target].add(ri)

    M_subj = _build_cooc_matrix(subj_to_rels, num_relations)
    M_obj = _build_cooc_matrix(obj_to_rels, num_relations)

    subj_svs, subj_density, subj_ents = _cooc_stats(M_subj)
    obj_svs, obj_density, obj_ents = _cooc_stats(M_obj)

    # --- Type statistics ---
    # Collect type assignments: subject -> set of types
    subj_types: dict = defaultdict(set)
    for e in g.es:
        if e["predicate"] == RDF_TYPE:
            type_name = g.vs[e.target]["name"]
            subj_types[e.source].add(type_name)

    # Class sizes: count distinct subjects per type
    class_counts: dict = defaultdict(int)
    for types in subj_types.values():
        for t in types:
            class_counts[t] += 1

    class_sizes = dict(class_counts)
    num_classes = len(class_sizes)
    zipf_exp = _fit_zipf_mle(np.array(list(class_sizes.values()), dtype=float))

    # P(r | type): for each type, distribution over outgoing relations of its subjects
    type_rel_counts: dict = defaultdict(lambda: defaultdict(int))
    for subj_vid, types in subj_types.items():
        rels_used = [g.es[eid]["predicate"] for eid in g.incident(subj_vid, mode="out")]
        for t in types:
            for r in rels_used:
                type_rel_counts[t][r] += 1

    type_relation_conditional: dict = {}
    for t, rel_counts in type_rel_counts.items():
        total = sum(rel_counts.values())
        type_relation_conditional[t] = {r: cnt / total for r, cnt in rel_counts.items()}

    return BlockC(
        subj_singular_values=subj_svs,
        subj_cooc_density=subj_density,
        subj_row_entropies=subj_ents,
        obj_singular_values=obj_svs,
        obj_cooc_density=obj_density,
        obj_row_entropies=obj_ents,
        num_classes=num_classes,
        class_size_zipf_exponent=zipf_exp,
        class_sizes=class_sizes,
        type_relation_conditional=type_relation_conditional,
    )
