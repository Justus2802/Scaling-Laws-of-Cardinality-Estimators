"""Graph signature measurement for KGs loaded via kg_io.load_kg.

Block B (degree structure) uses the `powerlaw` package to fit heavy-tailed
degree distributions and compare against alternative distributions via
Kolmogorov–Smirnov goodness-of-fit. References:

- Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-Law
  Distributions in Empirical Data. SIAM Review, 51(4), 661–703.
  https://doi.org/10.1137/070710111
- Alstott, J., Bullmore, E., & Plenz, D. (2014). powerlaw: A Python Package
  for Analysis of Heavy-Tailed Distributions. PLoS ONE 9(1): e85777.
  https://doi.org/10.1371/journal.pone.0085777
"""

import contextlib
import functools
import io
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, NamedTuple

import igraph
import numpy as np
import powerlaw
import scipy.sparse
import scipy.sparse.linalg

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_TOP_K_SV = 10  # number of singular values to keep
MIN_SAMPLES_FOR_FIT = 10  # below this, powerlaw.Fit results are dominated by noise


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
    num_entities = len(g.vs.select(is_literal_eq=False))
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
# Block B — Degree structure
# ---------------------------------------------------------------------------


class PowerLawStats(NamedTuple):
    """Six-number summary of a power-law fit via `powerlaw.Fit`.

    Used uniformly for the two aggregate degree distributions and for every
    per-relation multiplicity distribution. An all-NaN instance means the fit
    was skipped (too few samples) or raised internally.
    """
    alpha: float          # power-law exponent α from P(x) ∝ x^(-α)
    xmin: float           # lower-bound of the tail (KS-optimized by powerlaw)
    ks: float             # KS distance of the power-law fit itself
    D_lognormal: float    # KS distance for the alternative lognormal fit
    D_exponential: float  # KS distance for the alternative exponential fit
    D_truncated: float    # KS distance for the alternative truncated_power_law fit


def _nan_power_law_stats() -> PowerLawStats:
    """Return an all-NaN PowerLawStats — the canonical 'fit unavailable' value."""
    return PowerLawStats(*([float("nan")] * 6))


def _fit_powerlaw(data: np.ndarray) -> PowerLawStats:
    """Fit a power-law to a 1-D non-negative integer array and report KS distances.

    Filters to strictly positive samples (the `powerlaw` package rejects zeros).
    If fewer than MIN_SAMPLES_FOR_FIT positive samples remain, short-circuits
    to all-NaN — Clauset/Shalizi/Newman (2009, Sec. 3.3) show that fitted α
    has prohibitively wide confidence intervals on small samples, so the fit
    would produce noise. Skipping also avoids the package's stdout chatter,
    division-by-zero warnings, and per-call overhead on long-tail relations.

    Returns a PowerLawStats with:
      - alpha, xmin, ks from `fit.power_law` (the power-law fit itself)
      - D_lognormal, D_exponential, D_truncated from each alternative's own KS
        distance (`fit.<dist>.D`). Smaller D ⇒ that distribution fits better.

    Any exception inside the fitter is swallowed and yields all-NaN.
    """
    positive = data[data > 0]
    if positive.size < MIN_SAMPLES_FOR_FIT:
        return _nan_power_law_stats()
    try:
        with warnings.catch_warnings(), \
             np.errstate(divide="ignore", invalid="ignore"), \
             contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            fit = powerlaw.Fit(positive, discrete=True, verbose=False)
            return PowerLawStats(
                alpha=float(fit.power_law.alpha),
                xmin=float(fit.power_law.xmin),
                ks=float(fit.power_law.D),
                D_lognormal=float(fit.lognormal.D),
                D_exponential=float(fit.exponential.D),
                D_truncated=float(fit.truncated_power_law.D),
            )
    except Exception:
        return _nan_power_law_stats()


def _summarize_values(values: Iterable[float]) -> tuple[float, float, float, float]:
    """Return NaN-safe (mean, std, min, max) over an iterable of floats.

    Returns four NaNs when the iterable is empty or all values are NaN
    (guards the `nanmin`/`nanmax` "all-NaN slice" warning).
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan"), float("nan"), float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            float(np.nanmean(arr)),
            float(np.nanstd(arr)),
            float(np.nanmin(arr)),
            float(np.nanmax(arr)),
        )


def _per_relation_features(
    g: igraph.Graph,
) -> tuple[dict, dict, dict, dict]:
    """Build per-relation multiplicity / functionality features in one edge pass.

    For each predicate r, computes:
      - object_multiplicity[r]: PowerLawStats over (#distinct objects per subject)
      - subject_multiplicity[r]: PowerLawStats over (#distinct subjects per object)
      - functionality[r]: fraction of subjects whose object-multiplicity == 1
      - inverse_functionality[r]: fraction of objects whose subject-multiplicity == 1

    Assumes `kg_io.load_kg`'s contract: at most one edge per (subject, predicate,
    object) triple. Under that invariant, the count of edges with fixed (r, s)
    equals the number of distinct objects, so plain integer counters suffice.
    Per-relation `_fit_powerlaw` calls short-circuit to all-NaN for relations
    with fewer than MIN_SAMPLES_FOR_FIT distinct subjects (or objects).
    """
    subj_obj_count: dict = defaultdict(lambda: defaultdict(int))
    obj_subj_count: dict = defaultdict(lambda: defaultdict(int))
    for e in g.es:
        r = e["predicate"]
        subj_obj_count[r][e.source] += 1
        obj_subj_count[r][e.target] += 1

    object_multiplicity: dict = {}
    subject_multiplicity: dict = {}
    functionality: dict = {}
    inverse_functionality: dict = {}

    for r, subj_map in subj_obj_count.items():
        obj_counts = np.fromiter(subj_map.values(), dtype=int, count=len(subj_map))
        object_multiplicity[r] = _fit_powerlaw(obj_counts)
        functionality[r] = float(np.mean(obj_counts == 1)) if obj_counts.size else float("nan")

    for r, obj_map in obj_subj_count.items():
        subj_counts = np.fromiter(obj_map.values(), dtype=int, count=len(obj_map))
        subject_multiplicity[r] = _fit_powerlaw(subj_counts)
        inverse_functionality[r] = float(np.mean(subj_counts == 1)) if subj_counts.size else float("nan")

    return object_multiplicity, subject_multiplicity, functionality, inverse_functionality


@dataclass
class BlockB:
    """Block B — Degree structure features of a KG.

    Aggregate features fit the in/out-degree distributions (over non-literal
    vertices) with the `powerlaw` package. Per-relation features quantify how
    multi-valued each predicate is, distinguishing functional relations like
    `bornIn` from many-to-many ones like `friend`/`type` — they have very
    different join selectivities.
    """
    out_degree_fit: PowerLawStats   # over d_out(v) for non-literal v with d_out>0
    in_degree_fit: PowerLawStats    # over d_in(v)  for non-literal v with d_in>0

    object_multiplicity: dict       # relation_uri -> PowerLawStats
    subject_multiplicity: dict      # relation_uri -> PowerLawStats
    functionality: dict             # relation_uri -> fraction in [0, 1]
    inverse_functionality: dict     # relation_uri -> fraction in [0, 1]

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 68-vector for cross-KG comparison.

        Layout (in order):
          - out_degree_fit fields (6 floats: alpha, xmin, ks, D_lognormal,
            D_exponential, D_truncated)
          - in_degree_fit fields (6 floats, same order)
          - For object_multiplicity: for each of the 6 PowerLawStats fields,
            (mean, std, min, max) over the per-relation values → 24 floats
          - Same for subject_multiplicity → 24 floats
          - (mean, std, min, max) of functionality.values() → 4 floats
          - (mean, std, min, max) of inverse_functionality.values() → 4 floats

        Per-relation dicts are summarized rather than emitted directly so the
        vector length stays fixed across KGs with any number of predicates.
        """
        vec: list[float] = []
        vec.extend(self.out_degree_fit)
        vec.extend(self.in_degree_fit)

        for stat_dict in (self.object_multiplicity, self.subject_multiplicity):
            for field in PowerLawStats._fields:
                vec.extend(_summarize_values(getattr(v, field) for v in stat_dict.values()))

        vec.extend(_summarize_values(self.functionality.values()))
        vec.extend(_summarize_values(self.inverse_functionality.values()))
        return vec


def block_b(g: igraph.Graph) -> BlockB:
    """Compute Block B (degree structure) of the graph signature.

    Degree distributions are taken over non-literal vertices only (matching
    Block A's |V| definition); literals can only appear as RDF objects and
    would always have d_out=0. Self-loops contribute 1 to each side, which is
    the RDF-correct count of triples-as-subject and triples-as-object.

    The aggregate power-law fits use `powerlaw.Fit` (KS-optimized x_min,
    discrete-aware, with alternative-distribution KS distances); per-relation
    fits reuse the same helper. See `_fit_powerlaw` for the short-circuit on
    small samples.
    """
    non_lit_vs = g.vs.select(is_literal_eq=False)
    if len(non_lit_vs):
        subject_multiplicity_overall = np.array(g.degree(non_lit_vs, mode="out"), dtype=int)
        object_multiplicity_overall = np.array(g.degree(non_lit_vs, mode="in"), dtype=int)
    else:
        subject_multiplicity_overall = np.array([], dtype=int)
        object_multiplicity_overall = np.array([], dtype=int)

    subject_multiplicity_overall_fit = _fit_powerlaw(subject_multiplicity_overall)
    object_multiplicity_overall_fit = _fit_powerlaw(object_multiplicity_overall)

    object_multiplicity, subject_multiplicity, functionality, inverse_functionality = (
        _per_relation_features(g)
    )

    return BlockB(
        out_degree_fit=subject_multiplicity_overall_fit,
        in_degree_fit=object_multiplicity_overall_fit,
        object_multiplicity=object_multiplicity,
        subject_multiplicity=subject_multiplicity,
        functionality=functionality,
        inverse_functionality=inverse_functionality,
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


# ---------------------------------------------------------------------------
# Block E — Motifs (controllable shape distribution)
# ---------------------------------------------------------------------------

_SAMPLE_BUDGET = 100_000  # default walk samples for path/tree templates


@dataclass
class BlockE:
    """Block E — Motif shape distribution of a KG."""
    # Exact counts on the undirected simplification
    triangle_count: int           # 3-cycle (K3)
    four_cycle_count: int         # C4 (exact)
    five_cycle_count: int         # C5 (sampled estimate)
    six_cycle_count: int          # C6 (sampled estimate)
    diamond_count: int            # K4 minus one edge (5 edges on 4 nodes)
    k4_count: int                 # complete graph on 4 nodes
    tailed_triangle_count: int    # triangle + one pendant (paw)

    # Exact star counts: number of k-star subgraphs = Σ_v C(deg(v), k), k=2..10
    star_counts: dict             # int k -> int

    # Sampled directed-walk path templates, k=2..10
    path_template_zipf: dict      # int k -> float
    path_template_entropy: dict   # int k -> float

    # Sampled depth-2 rooted tree templates
    tree_template_zipf: float
    tree_template_entropy: float

    def as_vector(self) -> list[float]:
        vec = [
            float(self.triangle_count),
            float(self.four_cycle_count),
            float(self.five_cycle_count),
            float(self.six_cycle_count),
            float(self.diamond_count),
            float(self.k4_count),
            float(self.tailed_triangle_count),
        ]
        for k in range(2, 11):
            vec.append(float(self.star_counts.get(k, 0)))
        for k in range(2, 11):
            vec.append(self.path_template_zipf.get(k, float("nan")))
        for k in range(2, 11):
            vec.append(self.path_template_entropy.get(k, float("nan")))
        vec.extend([self.tree_template_zipf, self.tree_template_entropy])
        return vec  # length 7 + 9 + 9 + 9 + 2 = 36


@functools.lru_cache(maxsize=1)
def _4node_motif_index_map() -> dict:
    """Discover which index in motifs_randesu(size=4) maps to each named 4-node motif.

    Creates a canonical 4-node example for each motif, runs full RANDESU enumeration,
    and finds the unique index with count == 1.  Cached so it runs only once per process.
    """
    specs = {
        "four_cycle":      igraph.Graph(n=4, edges=[(0,1),(1,2),(2,3),(3,0)]),
        "diamond":         igraph.Graph(n=4, edges=[(0,1),(0,2),(0,3),(1,2),(1,3)]),
        "k4":              igraph.Graph(n=4, edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]),
        "tailed_triangle": igraph.Graph(n=4, edges=[(0,1),(1,2),(0,2),(2,3)]),
    }
    index_map: dict = {}
    for name, pattern in specs.items():
        for i, count in enumerate(pattern.motifs_randesu(size=4)):
            if not math.isnan(count) and int(count) == 1:
                index_map[name] = i
                break
    return index_map


def _template_stats(counts: dict) -> tuple[float, float]:
    """Return (Zipf exponent, Shannon entropy) from a {template: count} dict."""
    if not counts:
        return float("nan"), float("nan")
    freqs = np.array(list(counts.values()), dtype=float)
    zipf = _fit_zipf_mle(freqs)
    p = freqs / freqs.sum()
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    return zipf, entropy


def _build_out_adj(g: igraph.Graph) -> tuple[dict, list]:
    """Build adjacency list for directed walks, skipping literal targets.

    Returns (out_edges dict: vertex_id -> [(neighbor_id, predicate)],
             start_verts: non-literal vertices that have at least one outgoing edge).
    """
    out_edges: dict = defaultdict(list)
    for e in g.es:
        if not g.vs[e.target]["is_literal"]:
            out_edges[e.source].append((e.target, e["predicate"]))
    start_verts = [v for v, adj in out_edges.items() if not g.vs[v]["is_literal"]]
    return dict(out_edges), start_verts


def _sample_path_templates(
    out_edges: dict,
    start_verts: np.ndarray,
    k: int,
    n_samples: int,
    rng: np.random.Generator,
) -> dict:
    """Sample n_samples directed walks of length k; count relation-sequence tuples."""
    counts: dict = defaultdict(int)
    for _ in range(n_samples):
        v = int(rng.choice(start_verts))
        rels: list = []
        for _ in range(k):
            adj = out_edges.get(v)
            if not adj:
                break
            nb, rel = adj[int(rng.integers(len(adj)))]
            rels.append(rel)
            v = nb
        if len(rels) == k:
            counts[tuple(rels)] += 1
    return dict(counts)


def _sample_tree_depth2_templates(
    out_edges: dict,
    start_verts: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> dict:
    """Sample depth-2 rooted trees; template = sorted tuple of (r1, r2) pairs."""
    counts: dict = defaultdict(int)
    for _ in range(n_samples):
        root = int(rng.choice(start_verts))
        adj1 = out_edges.get(root)
        if not adj1:
            continue
        pairs: list = []
        for child, r1 in adj1:
            adj2 = out_edges.get(child)
            if adj2:
                for _, r2 in adj2:
                    pairs.append((r1, r2))
        if pairs:
            counts[tuple(sorted(pairs))] += 1
    return dict(counts)


def _count_stars(g_und: igraph.Graph) -> dict:
    """Exact k-star counts for k=2..10 via the degree distribution.

    A k-star subgraph centred at v exists for every k-subset of v's neighbours.
    Count = Σ_v C(deg(v), k).
    """
    from math import comb
    degrees = g_und.degree()
    return {k: sum(comb(d, k) for d in degrees if d >= k) for k in range(2, 11)}


def _estimate_k_cycle(
    g_und: igraph.Graph,
    k: int,
    n_samples: int,
    rng: np.random.Generator,
) -> int:
    """Estimate k-cycle count via random walk closure sampling.

    Samples simple walks of length k-1 from random starting vertices and checks
    whether the endpoint connects back to the start, forming a simple k-cycle.
    Scales the closure rate by n * avg_degree^(k-1) / (2k) — a first-order
    approximation that is accurate in order of magnitude for sparse graphs.
    """
    n = g_und.vcount()
    if n < k:
        return 0
    adj = [list(g_und.neighbors(v)) for v in range(n)]
    degrees = np.array([len(a) for a in adj], dtype=float)
    avg_deg = float(degrees.mean()) if n > 0 else 0.0

    n_closed = 0
    n_valid = 0
    for _ in range(n_samples):
        start = int(rng.integers(n))
        if not adj[start]:
            continue
        v = start
        visited = {start}
        ok = True
        for _ in range(k - 1):
            candidates = [nb for nb in adj[v] if nb not in visited]
            if not candidates:
                ok = False
                break
            nb = candidates[int(rng.integers(len(candidates)))]
            visited.add(nb)
            v = nb
        n_valid += 1
        if ok and len(visited) == k and start in adj[v]:
            n_closed += 1

    if n_valid == 0 or n_closed == 0:
        return 0
    return int((n_closed / n_valid) * n * (avg_deg ** (k - 1)) / (2 * k))


def block_e(g: igraph.Graph, sample_budget: int = _SAMPLE_BUDGET) -> BlockE:
    """Compute Block E (motif distribution) of the graph signature.

    Exact counts for 3- and 4-node motifs on the undirected simplification.
    Path and tree templates are estimated by random walk sampling.
    """
    # --- Exact motif counts on undirected simple graph ---
    g_und = g.as_undirected(combine_edges="first").simplify()

    # Triangles: A ⊙ A² summed and divided by 6 (each triangle counted 6 times)
    A = scipy.sparse.csr_matrix(g_und.get_adjacency_sparse())
    tri_count = int(A.multiply(A @ A).sum() // 6)

    # 4-node motifs: full RANDESU enumeration with runtime index discovery
    idx_map = _4node_motif_index_map()
    four_motifs = g_und.motifs_randesu(size=4)

    def _get_motif(name: str) -> int:
        i = idx_map.get(name)
        if i is None or i >= len(four_motifs):
            return 0
        v = four_motifs[i]
        return 0 if math.isnan(v) else int(v)

    # Stars (exact) and 5/6-cycles (sampled)
    star_counts = _count_stars(g_und)
    motif_rng = np.random.default_rng(0)
    n_cycle = max(1, sample_budget // 2)
    five_cycle = _estimate_k_cycle(g_und, 5, n_cycle, motif_rng)
    six_cycle  = _estimate_k_cycle(g_und, 6, n_cycle, motif_rng)

    # --- Path and tree templates from directed graph ---
    out_edges, start_verts_list = _build_out_adj(g)
    start_verts = np.array(start_verts_list)

    path_template_zipf: dict = {}
    path_template_entropy: dict = {}
    tree_zipf = float("nan")
    tree_ent = float("nan")

    if start_verts.size > 0:
        rng = np.random.default_rng(1)
        n_per_k = max(1, sample_budget // 9)  # spread budget evenly across k=2..10
        for k in range(2, 11):
            counts = _sample_path_templates(out_edges, start_verts, k, n_per_k, rng)
            path_template_zipf[k], path_template_entropy[k] = _template_stats(counts)

        tree_counts = _sample_tree_depth2_templates(
            out_edges, start_verts, sample_budget, rng
        )
        tree_zipf, tree_ent = _template_stats(tree_counts)

    return BlockE(
        triangle_count=tri_count,
        four_cycle_count=_get_motif("four_cycle"),
        five_cycle_count=five_cycle,
        six_cycle_count=six_cycle,
        diamond_count=_get_motif("diamond"),
        k4_count=_get_motif("k4"),
        tailed_triangle_count=_get_motif("tailed_triangle"),
        star_counts=star_counts,
        path_template_zipf=path_template_zipf,
        path_template_entropy=path_template_entropy,
        tree_template_zipf=tree_zipf,
        tree_template_entropy=tree_ent,
    )
